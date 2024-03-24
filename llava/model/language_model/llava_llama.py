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


from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer,LlamaFlashAttention2,apply_rotary_pos_emb, repeat_kv
from llava.constants import MAPPINGX, MAPPINGY
from llava.model.multimodal_projector.visual_plugin import CAbstractor

logger = logging.get_logger(__name__)
def adjust_attention_mask(attention_mask, q_len, kv_seq_len):
    if len(attention_mask.shape) == 2:
        batch_size, seq_length = attention_mask.shape
        new_attention_mask = attention_mask[:,:q_len]
    else:
        batch_size, _, seq_length, _ = attention_mask.shape
        new_attention_mask = attention_mask[:, :, :q_len, :kv_seq_len]
    return new_attention_mask

def unflatten_image_features(image_features, position_ids):
    B,C,Q = image_features.shape
    image_features = image_features.view(B, C, 24, 24).contiguous()
    position_ids = position_ids.view(B, 24, 24).unsqueeze(1)
    start_ids = position_ids.min()
    mappingx = MAPPINGX.to(position_ids.device)
    mappingy = MAPPINGY.to(position_ids.device)
    x_ids = mappingx[position_ids-start_ids]
    y_ids = mappingy[position_ids-start_ids]
    
    return image_features, x_ids, y_ids

def flatten_image_features(image_features,x_ids,y_ids,position_ids):
    B,C,H,W = image_features.shape
    start_ids = position_ids.min()
    image_features = image_features.view(B,C,H*W).permute(0,2,1).contiguous()
    position_ids_2d = torch.floor(x_ids) + 24 * torch.floor(y_ids)
    position_ids = position_ids_2d.view(B,H*W) + start_ids
    
    return image_features,position_ids

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    def __init__(self, 
                 grouping=None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.grouping = grouping
class MyFlashAttention2(LlamaFlashAttention2):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max()+1)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class MyLlamaSdpaAttention(LlamaSdpaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max()+1)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

MY_LLAMA_ATTENTION_CLASSES = {
    "flash_attention_2": MyFlashAttention2,
    "sdpa": MyLlamaSdpaAttention,
}   
class MyLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # self.self_attn = MyLlamaSdpaAttention(config=config, layer_idx=layer_idx)
        self.self_attn = MY_LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        self.images_idx = None
        self.grouping = None
        self.groupingLayer = None
        self.stride = None
        self.layers = nn.ModuleList(
            [MyLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.label_ids = None
        self.CAbstractor = None
        self.hidden_size = config.hidden_size
        
    def create_CAbstractor(self, num_pre_layers, num_post_layers,stride):
        self.CAbstractor = CAbstractor(hidden_dim=self.hidden_size, 
                                       num_pre_layers=num_pre_layers, 
                                       num_post_layers=num_post_layers, 
                                       pool_stride=stride)
    def get_CAbstractor(self):
        return self.CAbstractor
    
    def visual_operating(self, hidden_states, position_ids, operator):
        if self.images_idx is not None:
            i = 0
            # copy position ids for batch size time
            position_ids = position_ids.repeat(hidden_states.shape[0], 1)
            # cat hidden states with position ids
            new_hidden_states = []
            new_position_ids = []
            FLAG = False
            for image_idx in self.images_idx:
                if image_idx[0].shape[0] == 0:
                    continue
                FLAG = True
                states_segment = []
                position_segment = []
                for vi in range(image_idx[0].shape[0]):
                    if vi == 0:
                        states_segment.append(hidden_states[i:i+1,0: image_idx[vi]])
                        position_segment.append(position_ids[i:i+1,0: image_idx[vi]])
                    else:
                        states_segment.append(hidden_states[i:i+1,image_idx[vi-1] + 576: image_idx[vi]])
                        position_segment.append(position_ids[i:i+1,image_idx[vi-1] + 576: image_idx[vi]])
                    visual_states = hidden_states[i:i+1,image_idx[vi]: image_idx[vi] + 576].permute(0,2,1).contiguous()
                    visual_positions = position_ids[i:i+1,image_idx[vi]: image_idx[vi] + 576].unsqueeze(1).contiguous()
                    visual_states,visual_positions = operator(visual_states,visual_positions)
                    states_segment.append(visual_states)
                    position_segment.append(visual_positions.to(position_ids.dtype))
                    if vi == image_idx[0].shape[0] - 1:
                        states_segment.append(hidden_states[i:i+1,image_idx[vi] + 576: ])
                        position_segment.append(position_ids[i:i+1,image_idx[vi] + 576: ])
                states_segment = torch.cat(states_segment, dim=1)
                position_segment = torch.cat(position_segment, dim=1)
                new_hidden_states.append(states_segment) 
                new_position_ids.append(position_segment)
                i += 1
            # filling the position ids so that the model can use them
            if FLAG:
                hidden_states = torch.cat(new_hidden_states, dim=0)
                new_position_ids = torch.cat(new_position_ids, dim=0).to(position_ids.device).to(position_ids.dtype)
            else:
                new_position_ids = position_ids
        else:
            new_position_ids = position_ids
            
        return hidden_states, new_position_ids

    def visual_avg_pool1d(self,visual_states, visual_positions):
        visual_states = torch.nn.functional.avg_pool1d(visual_states, kernel_size=self.stride, stride=self.stride).permute(0,2,1).contiguous()
        visual_positions = torch.nn.functional.avg_pool1d(visual_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride).squeeze(1).contiguous()
        return visual_states, visual_positions
    
    def visual_avg_pool2d(self,visual_states, visual_positions):
        visual_states, visual_x_positions, visual_y_positions = unflatten_image_features(visual_states, visual_positions)
        visual_states = torch.nn.functional.avg_pool2d(visual_states, kernel_size=self.stride, stride=self.stride)
        visual_x_positions = torch.nn.functional.avg_pool2d(visual_x_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_y_positions = torch.nn.functional.avg_pool2d(visual_y_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_states, visual_positions = flatten_image_features(visual_states, visual_x_positions, visual_y_positions,visual_positions)
        return visual_states, visual_positions
    
    def apply_CAbstractor(self, visual_states, visual_positions):
        visual_states, visual_x_positions, visual_y_positions = unflatten_image_features(visual_states, visual_positions)
        visual_states = self.CAbstractor(visual_states)
        visual_x_positions = torch.nn.functional.avg_pool2d(visual_x_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_y_positions = torch.nn.functional.avg_pool2d(visual_y_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_states, visual_positions = flatten_image_features(visual_states, visual_x_positions, visual_y_positions,visual_positions)
        return visual_states, visual_positions
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        layer_idx = 0
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if layer_idx == self.groupingLayer and self.grouping != 'none':
                if self.grouping == 'avgpool1d':
                    hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.visual_avg_pool1d)
                    self.label_ids = position_ids
                elif self.grouping == 'avgpool2d':
                    hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.visual_avg_pool2d)
                    self.label_ids = position_ids
                elif self.grouping == 'cabstractor':
                    hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.apply_CAbstractor)
                    self.label_ids = position_ids
                else:
                    raise ValueError(f"Grouping {self.grouping} is not supported")
                if attention_mask is not None:
                    q_len = hidden_states.size(1)
                    if past_key_values is not None:
                        kv_seq_len = q_len + past_key_values.get_usable_length(q_len, layer_idx)
                    else:
                        kv_seq_len = q_len
                    attention_mask = adjust_attention_mask(attention_mask,q_len,kv_seq_len)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            layer_idx += 1

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.images_idx = None
        self.model.grouping = config.grouping
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
                labels,
                images_idx
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            self.model.images_idx = images_idx
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if self.model.label_ids is not None:
                labels = torch.gather(labels, 1, self.model.label_ids)
            # Shift so that tokens < n predict n
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
        images_idx = None
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                images_idx
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
        self.model.images_idx = images_idx
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

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
