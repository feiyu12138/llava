#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union
import warnings
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image
from llava.constants import IGNORE_INDEX,IMAGE_TOKEN_INDEX

from transformers.generation.beam_search import BeamScorer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList,validate_stopping_criteria
from transformers.generation.utils import  GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    def __init__(self, 
                 grouping=None, 
                 cot_decoding=None,
                 num_branch=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.grouping = grouping
        self.cot_decoding = cot_decoding
        self.num_branch = num_branch


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_branch = config.num_branch
        self.cot_decoding = config.cot_decoding

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            # init values
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
            if max_length is not None:
                warnings.warn(
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                    UserWarning,
                )
                stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
            pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
            eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
            output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
            output_attentions = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )
            return_dict_in_generate = (
                return_dict_in_generate
                if return_dict_in_generate is not None
                else self.generation_config.return_dict_in_generate
            )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

            this_peer_finished = False  # used by synced_gpus only
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            # argmax
            if not self.cot_decoding:
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            else:
                sorted_scores, next_tokens = torch.topk(torch.softmax(next_tokens_scores,dim=-1), self.num_branch, dim=-1)
                delta_list = torch.tensor([sorted_scores[:,0] - sorted_scores[:,1]] + [0] * (self.num_branch - 1))

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if self.cot_decoding:
                input_ids = torch.cat([input_ids.repeat(1,next_tokens.shape[1]), next_tokens], dim=-2).permute(1,0)
                input_ids_list = input_ids.split(1, dim=0)
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                input_ids_list = [input_ids]
            model_kwargs = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                    )
            finished_inpiut_ids = []
            finished_delta = []
            past_key_values_snapshot = model_kwargs.get("past_key_values")
            attention_mask_snapshot = model_kwargs.get("attention_mask")
            for input_ids,delta in zip(input_ids_list,delta_list):
                model_kwargs["past_key_values"] = past_key_values_snapshot
                model_kwargs["attention_mask"] = attention_mask_snapshot
                this_peer_finished = False
                synced_gpus = False
                unfinished_sequences = 1 - unfinished_sequences
                while True:
                    if synced_gpus:
                        # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                        # The following logic allows an early break if all peers finished generating their sequence
                        this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                        # send 0.0 if we finished, 1.0 otherwise
                        dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                        # did all peers finish? the reduced sum will be 0.0 then
                        if this_peer_finished_flag.item() == 0.0:
                            break

                    # prepare model inputs
                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                    # forward pass to get next token
                    outputs = self(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )

                    if synced_gpus and this_peer_finished:
                        continue  # don't waste resources running the code we don't need

                    next_token_logits = outputs.logits[:, -1, :]

                    # pre-process distribution
                    next_tokens_scores = logits_processor(input_ids, next_token_logits)

                    # Store scores, attentions and hidden_states when required
                    if return_dict_in_generate:
                        if output_scores:
                            scores += (next_tokens_scores,)
                        if output_attentions:
                            decoder_attentions += (
                                (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                            )
                            if self.config.is_encoder_decoder:
                                cross_attentions += (outputs.cross_attentions,)

                        if output_hidden_states:
                            decoder_hidden_states += (
                                (outputs.decoder_hidden_states,)
                                if self.config.is_encoder_decoder
                                else (outputs.hidden_states,)
                            )

                    # argmax
                    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                    top1 = torch.softmax(next_tokens_scores, dim=-1).max()
                    top2_idx = torch.topk(next_tokens_scores, 2, dim=-1)[1][:,-1]
                    next_delta = top1 - torch.softmax(next_tokens_scores, dim=-1)[:,top2_idx]
                    
                    # finished sentences should have their next token be a padding token
                    if eos_token_id is not None:
                        if pad_token_id is None:
                            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                    # update generated ids, model inputs, and length for next step
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    delta = torch.cat([delta.view(1,-1), next_delta.to(delta.device)], dim=-1)
                    
                    if streamer is not None:
                        streamer.put(next_tokens.cpu())
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                    # if eos_token was found in one sentence, set sentence to finished
                    if eos_token_id_tensor is not None:
                        unfinished_sequences = unfinished_sequences.mul(
                            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                        )

                        # stop when each sentence is finished
                        if unfinished_sequences.max() == 0:
                            this_peer_finished = True

                    # stop if we exceed the maximum length
                    if stopping_criteria(input_ids, scores):
                        this_peer_finished = True

                    if this_peer_finished and not synced_gpus:
                        finished_inpiut_ids.append(input_ids)
                        finished_delta.append(delta[:,:-1])
                        break

                if streamer is not None:
                    streamer.end()
            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                if self.cot_decoding:
                    return finished_inpiut_ids, finished_delta
                return input_ids

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
