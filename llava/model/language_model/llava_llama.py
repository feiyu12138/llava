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
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput,GenerateBeamEncoderDecoderOutput
from llava.model.language_model.attention_viz import generate_attention_map, calc_qkvs_std

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from llava.model.vcc.coarser import Coarser
from llava.model.vcc.finer import Finer
from llava.model.vcc.selector import Selector
from llava.model.vcc.formatter import Formatter

GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]



from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer,LlamaFlashAttention2,apply_rotary_pos_emb, repeat_kv,rotate_half
from llava.constants import MAPPINGX, MAPPINGY
from llava.model.multimodal_projector.visual_plugin import Abstractor
from llava.mm_utils import get_anyres_image_grid_shape

logger = logging.get_logger(__name__)
DEBUG=True
def apply_rotary_pos_emb_for_msa(q, k, cos, sin, position_ids, source_position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos_target = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin_target = sin[position_ids].unsqueeze(unsqueeze_dim)
    if source_position_ids is not None:
        cos_source = cos[source_position_ids].unsqueeze(unsqueeze_dim)
        sin_source = sin[source_position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos_source = cos_target
        sin_source = sin_target
    q_embed = (q * cos_target) + (rotate_half(q) * sin_target)
    k_embed = (k * cos_source) + (rotate_half(k) * sin_source)
    return q_embed, k_embed

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
                 cot_decoding=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.grouping = grouping
        self.cot_decoding = cot_decoding
        
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
    
class AdaptiveFlashAttention2(LlamaFlashAttention2):
    def __init__(self, viz, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viz = viz
        self.attn_map = []
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        source_states: torch.Tensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        source_position_ids: Optional[torch.LongTensor] = None,
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
        if source_states is not None:
            kv_len = source_states.size(1)
        else:
            kv_len = q_len
            source_states = hidden_states

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(source_states)
        value_states = self.v_proj(source_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max()+1)
        query_states, key_states = apply_rotary_pos_emb_for_msa(query_states, key_states, cos, sin, position_ids,source_position_ids)
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
    
class AdaptiveLlamaSdpaAttention(LlamaSdpaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, viz: bool = False):
        super().__init__(config, layer_idx)
        self.viz = viz
        self.attn_map = None
        self.std = {}
        self.text_std = {}
        self.user_std = {}
        self.images_idx = None
    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        source_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        source_position_ids: Optional[torch.LongTensor] = None,
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
        if source_states is not None:
            kv_len = source_states.size(1)
        else:
            kv_len = q_len
            source_states = hidden_states

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(source_states)
        value_states = self.v_proj(source_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max()+1)
        query_states, key_states = apply_rotary_pos_emb_for_msa(query_states, key_states, cos, sin, position_ids, source_position_ids)
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
        if self.viz:
            self.attn_map = generate_attention_map(query_states, key_states)
            # self.std.update(
            #     calc_qkvs_std(query_states[:,:,self.images_idx:self.images_idx+576], 
            #                   key_states[:,:,self.images_idx:self.images_idx+576], 
            #                   value_states[:,:,self.images_idx:self.images_idx+576], 
            #                   hidden_states[:,self.images_idx:self.images_idx+576])
            # )
            # self.text_std.update(
            #     calc_qkvs_std(query_states[:,:,0:self.images_idx],
            #                     key_states[:,:,0:self.images_idx],
            #                     value_states[:,:,0:self.images_idx],
            #                     hidden_states[:,0:self.images_idx])
            # )
            # self.user_std.update(
            #     calc_qkvs_std(query_states[:,:,self.images_idx+576:],
            #                   key_states[:,:,self.images_idx+576:],
            #                   value_states[:,:,self.images_idx+576:],
            #                     hidden_states[:,self.images_idx+576:])
            # )
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
    "ada_flash_attention_2": MyFlashAttention2,
    "flash_attention_2": AdaptiveFlashAttention2,
    "sdpa": AdaptiveLlamaSdpaAttention,
    "adaptive_sdpa": MyLlamaSdpaAttention,
}  
class MyLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, viz: bool = False):
        super().__init__(config, layer_idx)
        # self.self_attn = MyLlamaSdpaAttention(config=config, layer_idx=layer_idx)
        self.self_attn = MY_LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx, viz=viz)
        
        
class AdaptiveLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, viz: bool = False):
        super().__init__(config, layer_idx)
        # self.self_attn = MyLlamaSdpaAttention(config=config, layer_idx=layer_idx)
        self.self_attn = MY_LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx, viz=viz)
    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        if compressed_hidden_states is not None and compressed_position_ids is None:
            raise ValueError("compressed_position_ids must be provided when compressed_hidden_states is provided")
        if compressed_hidden_states is None and compressed_position_ids is not None:
            raise ValueError("compressed_hidden_states must be provided when compressed_position_ids is provided")
        if compressed_hidden_states is not None:
            target_states = compressed_hidden_states
            target_position_ids = compressed_position_ids
            source_states = hidden_states
            source_position_ids = position_ids
        else:
            target_states = hidden_states
            target_position_ids = position_ids
            source_states = None
            source_position_ids = None
        residual = target_states
        target_states = self.input_layernorm(target_states)
        if source_states is not None:
            source_states = self.input_layernorm(source_states)
        # Self Attention
        target_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=target_states,
            source_states=source_states,
            attention_mask=attention_mask,
            position_ids=target_position_ids,
            source_position_ids=source_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        target_states = residual + target_states

        # Fully Connected
        residual = target_states
        target_states = self.post_attention_layernorm(target_states)
        target_states = self.mlp(target_states)
        target_states = residual + target_states

        outputs = (target_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
        

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        self.images_idx = None
        self.grouping = None
        self.groupingLayer = None
        self.groupingLayerList = []
        self.stride = None
        self.strideList = []
        self.layers = nn.ModuleList(
            [AdaptiveLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.label_ids = None
        self.Abstractor = None
        self.hidden_size = config.hidden_size
        self.halfpool = False
        self.viz = False
        self.attention_maps = []
        self.std_layers = []
        self.text_std_layers = []
        self.user_std_layers = []
        self.unified_vpe = False
        self.citer=1
        self.viz_assign = False
        self.assignment = None
        self.progressive = False
        self.step = 0
        self.pivot = 0
        self.pivotList = []
        self.OPT_DICT = {
            "avgpool1d": self.visual_avg_pool1d,
            "avgpool2d": self.visual_avg_pool2d,
            "withdrawn": self.visual_withdrawn,
            "detach_hard_k_means": self.apply_detach_soft_k_means,
            "detach_soft_k_means": self.apply_detach_soft_k_means,
            "hard_k_means": self.apply_hard_k_means,
            "soft_k_means": self.apply_soft_k_means,
            "block_random_drop": self.apply_block_random_drop
        }
    def set_lists(self, strideList, pivotList, groupingLayerList):
        '''
        strides, pivots, layers
        '''
        self.strideList = strideList
        self.pivotList = pivotList
        self.groupingLayerList = groupingLayerList
        self.stride = self.strideList.pop(0)
        self.pivot = self.pivotList.pop(0)
        self.groupingLayer = self.groupingLayerList.pop(0)
        print(f"Stride reduction, present stride is {self.stride}, present grouping layer is {self.groupingLayer}, present pivot is {self.pivot}")
        
    def step_stride_and_layer(self):
        if self.step % self.pivot == 0 and self.step != 0 and self.stride > 1:
            self.stride = self.strideList.pop(0)
            self.groupingLayer = self.groupingLayerList.pop(0)
            self.pivot = self.pivotList.pop(0) if len(self.pivotList) > 0 else 10000
            print(f"Stride reduction, present stride is {self.stride}, present grouping layer is {self.groupingLayer}, present pivot is {self.pivot}")
        self.step += 1

    def create_Abstractor(self, num_pre_layers, num_post_layers,stride,kernel_size,rel_pos_spatial):
        self.Abstractor = Abstractor(hidden_dim=self.hidden_size, 
                                       num_pre_layers=num_pre_layers, 
                                       num_post_layers=num_post_layers, 
                                       pool_stride=stride,
                                       rel_pos_spatial=rel_pos_spatial,
                                       grouping=self.grouping,
                                       kernel_size=kernel_size)

    def get_Abstractor(self):
        return self.Abstractor
    
    def visual_operating(self, hidden_states, position_ids, operator):
        if self.images_idx is not None:
            i = 0
            # copy position ids for batch size time
            if position_ids.shape[0] == 1:
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
                    visual_states = hidden_states[i:i+1,image_idx[vi]: image_idx[vi] + 576].permute(0,2,1)
                    visual_positions = position_ids[i:i+1,image_idx[vi]: image_idx[vi] + 576].unsqueeze(1)
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
    
    def visual_withdrawn(self,visual_states, visual_positions):
        visual_states = visual_states.permute(0,2,1).contiguous()
        B,L,D = visual_states.shape
        visual_positions = visual_positions.squeeze(1).contiguous()
        visual_states = torch.zeros(B,0,D,device=visual_states.device).to(visual_states.dtype)
        visual_positions = torch.zeros(B,0,device=visual_positions.device).to(visual_positions.dtype)
        return visual_states, visual_positions
    
    def visual_avg_pool2d(self,visual_states, visual_positions):
        visual_states, visual_x_positions, visual_y_positions = unflatten_image_features(visual_states, visual_positions)
        visual_states = torch.nn.functional.avg_pool2d(visual_states, kernel_size=self.stride, stride=self.stride)
        visual_x_positions = torch.nn.functional.avg_pool2d(visual_x_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_y_positions = torch.nn.functional.avg_pool2d(visual_y_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_states, visual_positions = flatten_image_features(visual_states, visual_x_positions, visual_y_positions,visual_positions)
        return visual_states, visual_positions
    
    def apply_Abstractor(self, visual_states, visual_positions):
        visual_states, visual_x_positions, visual_y_positions = unflatten_image_features(visual_states, visual_positions)
        visual_states = self.Abstractor(visual_states)
        visual_x_positions = torch.nn.functional.avg_pool2d(visual_x_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_y_positions = torch.nn.functional.avg_pool2d(visual_y_positions.to(torch.float16), kernel_size=self.stride, stride=self.stride)
        visual_states, visual_positions = flatten_image_features(visual_states, visual_x_positions, visual_y_positions,visual_positions)
        return visual_states, visual_positions
    
    def apply_random_drop(self,tokens, position_ids):
        K = tokens.size(2) // self.stride
        position_ids = position_ids.squeeze(1)
        # Randomly keep K tokens
        keep_ids = torch.randperm(tokens.size(2))[:K]    
        tokens = tokens[:,:,keep_ids]
        position_ids = position_ids[:,keep_ids]
        
        return tokens.permute(0,2,1).contiguous(), position_ids.contiguous()
    
    def apply_soft_k_means(self, tokens,positions,iterations=1):
        if self.citer != 1:
            iterations = self.citer
        start_ids = positions.min()
        centroids = tokens[:,:,::self.stride]
        for _ in range(iterations):
            distances = torch.sum(torch.abs((tokens.unsqueeze(3) - centroids.unsqueeze(2))), dim=1)
            weights = torch.softmax(-distances, dim=2)
            weights = weights / (weights.sum(1) + 1e-6)
            centroids = torch.einsum('bcl,blq->bcq', tokens, weights)
        centroids = centroids.permute(0,2,1)
        if self.viz_assign:
            self.assignment = weights
        positions = torch.zeros(centroids.shape[0],centroids.shape[1],device=positions.device).long() + start_ids
        return centroids,positions
    
    def apply_hard_k_means(self, tokens, positions, iterations=1, tau=1.0):
        if self.citer != 1:
            iterations = self.citer
        start_ids = positions.min()
        centroids = tokens[:, :, ::self.stride]  # initial centroids
        for _ in range(iterations):
            # Compute pairwise absolute differences and sum along feature dimension
            distances = torch.sum(torch.abs(tokens.unsqueeze(3) - centroids.unsqueeze(2)), dim=1)
            
            # Use Gumbel softmax for differentiable 'hard' assignment
            weights = torch.nn.functional.gumbel_softmax(-distances, tau=tau, dim=2, hard=True)
            weights = weights / (weights.sum(1) + 1e-6)
            
            # Update centroids: equivalent to weighted average where weights are one-hot encoded
            centroids = torch.einsum('bcl,blq->bcq', tokens, weights)
        if self.viz_assign:
            self.assignment = weights
        centroids = centroids.permute(0, 2, 1)
        positions = torch.zeros(centroids.shape[0], centroids.shape[1], device=positions.device).long() + start_ids
        
        return centroids, positions
        
    
    def apply_detach_soft_k_means(self, tokens,positions,iterations=1):
        if self.citer != 1:
            iterations = self.citer
        start_ids = positions.min()
        detach_tokens = tokens.detach()
        detach_centroids = detach_tokens[:,:,::self.stride]
        with torch.no_grad():
            for i in range(iterations):
                distances = torch.sum(torch.abs((detach_tokens.unsqueeze(3) - detach_centroids.unsqueeze(2))), dim=1)
                weights = torch.softmax(-distances, dim=2)
                weights = weights / (weights.sum(1) + 1e-6)
                detach_centroids = torch.einsum('bcl,blq->bcq', detach_tokens, weights)
                del distances
                if i<iterations-1:
                    del weights
        centroids = torch.einsum('bcl,blq->bcq', tokens, weights)
        centroids = centroids.permute(0,2,1)
        if self.viz_assign:
            self.assignment = weights
        positions = torch.zeros(centroids.shape[0],centroids.shape[1],device=positions.device).long() + start_ids
        del detach_tokens
        del detach_centroids
        del weights
        return centroids,positions
    
    def apply_detach_hard_k_means(self, tokens,positions,iterations=1):
        if self.citer != 1:
            iterations = self.citer
        start_ids = positions.min()
        detach_tokens = tokens.detach()
        detach_centroids = detach_tokens[:,:,::self.stride]
        with torch.no_grad():
            for i in range(iterations):
                distances = torch.sum(torch.abs((detach_tokens.unsqueeze(3) - detach_centroids.unsqueeze(2))), dim=1)
                weights = torch.argmax(-distances, dim=2)
                one_hot_weights = F.one_hot(weights, num_classes=distances.size(2)).to(detach_tokens.dtype)
                one_hot_weights = one_hot_weights / (one_hot_weights.sum(1) + 1e-6)
                detach_centroids = torch.einsum('bcl,blq->bcq', detach_tokens, one_hot_weights)
                del distances
                if i<iterations-1:
                    del weights
                    del one_hot_weights
        centroids = torch.einsum('bcl,blq->bcq', tokens, one_hot_weights).permute(0,2,1)
        if self.viz_assign:
            self.assignment = one_hot_weights
        positions = torch.zeros(centroids.shape[0],centroids.shape[1],device=positions.device).long() + start_ids
        del detach_tokens
        del detach_centroids
        del weights
        return centroids,positions

    def apply_PCA(self,tokens,positions):
        # apply PCA on the third dimension of the tokens
        
        return tokens,positions
    
    def apply_block_random_drop(self, tokens, position_ids):
        K = tokens.size(2) // self.stride
        # Randomly keep K continuous tokens
        start_ids = torch.randint(0,tokens.size(2)-K+1,(1,))
        tokens = tokens[:,:,start_ids:start_ids+K]
        position_ids = position_ids[:,:,start_ids:start_ids+K].squeeze(1)
        
        return tokens.permute(0,2,1), position_ids
    
    def apply_position_average(self, visual_states, visual_positions):
        visual_positions = torch.mean(visual_positions.float(), dim=2).long().repeat(1, 1, visual_states.size(2)).squeeze(1)
        return visual_states.permute(0,2,1), visual_positions
    
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
            if attention_mask is not None and self.groupingLayer == 0:
                attention_mask = adjust_attention_mask(attention_mask,seq_length + past_key_values_length,seq_length + past_key_values_length)
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
            if self.viz and self.images_idx is not None:
                decoder_layer.self_attn.viz = True
                decoder_layer.self_attn.images_idx = self.images_idx[0][0]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.unified_vpe and layer_idx == 0:
                hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.apply_position_average)
            if (layer_idx == self.groupingLayer and self.grouping != 'none'):
                if self.grouping == 'avgpool1d':
                    compressed_hidden_states, compressed_position_ids = self.visual_operating(hidden_states, position_ids, self.visual_avg_pool1d)
                    self.label_ids = compressed_position_ids
                elif self.grouping == 'avgpool2d':
                    compressed_hidden_states, compressed_position_ids = self.visual_operating(hidden_states, position_ids, self.visual_avg_pool2d)
                    self.label_ids = compressed_position_ids
                elif self.grouping.find('abstractor') != -1:
                    hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.apply_Abstractor)
                    self.label_ids = position_ids
                elif self.grouping == 'random_drop':
                    hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.apply_random_drop)
                    self.label_ids = position_ids
                elif self.grouping == 'block_random_drop':
                    hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.apply_block_random_drop)
                    self.label_ids = position_ids
                elif self.grouping == 'soft_k_means':
                    compressed_hidden_states, compressed_position_ids = self.visual_operating(hidden_states, position_ids, self.apply_soft_k_means)
                    self.label_ids = compressed_position_ids
                elif self.grouping == 'hard_k_means':
                    compressed_hidden_states, compressed_position_ids = self.visual_operating(hidden_states, position_ids, self.apply_hard_k_means)
                    self.label_ids = compressed_position_ids
                elif self.grouping == 'detach_soft_k_means':
                    compressed_hidden_states, compressed_position_ids = self.visual_operating(hidden_states, position_ids, self.apply_detach_soft_k_means)
                    self.label_ids = compressed_position_ids
                elif self.grouping == 'detach_hard_k_means':
                    compressed_hidden_states, compressed_position_ids = self.visual_operating(hidden_states, position_ids, self.apply_detach_hard_k_means)
                    self.label_ids = compressed_position_ids
                elif self.grouping == 'withdrawn':
                    compressed_hidden_states, compressed_position_ids = self.visual_operating(hidden_states, position_ids, self.visual_withdrawn)
                    self.label_ids = compressed_position_ids
                else:
                    raise ValueError(f"Grouping {self.grouping} is not supported")
                if attention_mask is not None:
                    q_len = hidden_states.size(1) if compressed_hidden_states is None else compressed_hidden_states.size(1)
                    if past_key_values is not None:
                        kv_seq_len = q_len + past_key_values.get_usable_length(q_len, layer_idx)
                    else:
                        kv_seq_len = q_len
                    attention_mask = adjust_attention_mask(attention_mask,q_len,kv_seq_len)
            elif (layer_idx == 0 and self.unified_vpe):
                hidden_states, position_ids = self.visual_operating(hidden_states, position_ids, self.apply_position_average)
                compressed_hidden_states = None
                compressed_position_ids = None
            else:
                compressed_hidden_states = None
                compressed_position_ids = None
            
            if not self.halfpool and compressed_hidden_states is not None:
                hidden_states = compressed_hidden_states
                position_ids = compressed_position_ids
                compressed_hidden_states = None
                compressed_position_ids = None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    compressed_hidden_states,
                    attention_mask,
                    position_ids,
                    compressed_position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    compressed_hidden_states=compressed_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    compressed_position_ids=compressed_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if compressed_position_ids is not None:
                position_ids = compressed_position_ids
                compressed_position_ids = None

            layer_idx += 1
            if self.viz:
                assert decoder_layer.self_attn.attn_map is not None, "Attention map is not generated. Please set viz=True in the attn layer"
                self.attention_maps.append(decoder_layer.self_attn.attn_map)
                self.std_layers.append(decoder_layer.self_attn.std)
                self.text_std_layers.append(decoder_layer.self_attn.text_std)
                self.user_std_layers.append(decoder_layer.self_attn.user_std)
        
        if self.viz and self.images_idx is not None:
            top_left = [self.images_idx[0][0].item(), self.images_idx[0][0].item()]
            width_height_init = [576,576]
            for idx, map in enumerate(self.attention_maps):
                width_height = width_height_init if idx < self.groupingLayer else [width_height_init[0]//self.stride,width_height_init[1]//self.stride]
                map = map.squeeze(0).cpu().detach().numpy()
                
                plt.figure()
                plt.imshow(map,cmap='coolwarm',interpolation='none')
                plt.colorbar()
                rect = patches.Rectangle((top_left[0], 0), width_height[0], map.shape[0],
                         linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.ylabel('Query ID')
                plt.xlabel('Key ID')
                plt.tight_layout()
                plt.savefig(f'{self.savedir}/attention_map_{idx}.png',dpi=300)
                # x label is query id
                # y label is key id
                plt.close()
                plt.figure()
                visual_map = map[:,top_left[0]:top_left[0]+width_height[0]]
                visual_attention_max = map[top_left[1]:top_left[1]+width_height[1],top_left[0]:top_left[0]+width_height[0]].max()
                visual_attention_min = map[top_left[1]:top_left[1]+width_height[1],top_left[0]:top_left[0]+width_height[0]].min()
                visual_map = (visual_map - visual_attention_min) / (visual_attention_max - visual_attention_min)
                plt.ylabel('Query ID')
                plt.xlabel('Visual Key ID')
                # set x range to be the same as the visual key range: [top_left[0],top_left[0]+width_height[0]](just for visualization purpose)
                # plt.xticks(np.arange(0, width_height[0],50), np.arange(top_left[0], top_left[0]+width_height[0],50))
                # plt.imshow(visual_map,cmap='coolwarm',interpolation='none')
                # plt.tight_layout()
                # plt.savefig(f'{self.savedir}/attention_map_{idx}_visual_key.png',dpi=300)
                # plt.close()
            # state_std_layers = [std['state'].cpu() for std in self.std_layers]
            # query_std_layers = [std['query'].cpu() for std in self.std_layers]
            # key_std_layers = [std['key'].cpu() for std in self.std_layers]
            # value_std_layers = [std['value'].cpu() for std in self.std_layers]
            # plt.figure()
            # plt.title('Standard Deviation of QKVS, segment length = 4')
            # plt.plot(state_std_layers,label='state std')
            # plt.plot(query_std_layers,label='query std')
            # plt.plot(key_std_layers,label='key std')
            # plt.plot(value_std_layers,label='value std')
            # plt.xlabel('Layer')
            # plt.ylabel('Standard Deviation')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('tempt/visual_std_layers.png',dpi=300)
            # plt.close()
            # text_state_std_layers = [std['state'].cpu() for std in self.text_std_layers]
            # text_query_std_layers = [std['query'].cpu() for std in self.text_std_layers]
            # text_key_std_layers = [std['key'].cpu() for std in self.text_std_layers]
            # text_value_std_layers = [std['value'].cpu() for std in self.text_std_layers]
            # plt.figure()
            # plt.title('Standard Deviation of QKVS(system), segment length = 4')
            # plt.plot(text_state_std_layers,label='state std')
            # plt.plot(text_query_std_layers,label='query std')
            # plt.plot(text_key_std_layers,label='key std')
            # plt.plot(text_value_std_layers,label='value std')
            # plt.xlabel('Layer')
            # plt.ylabel('Standard Deviation')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('tempt/system_std_layers.png',dpi=300)
            # plt.close()
            # user_state_std_layers = [std['state'].cpu() for std in self.user_std_layers]
            # user_query_std_layers = [std['query'].cpu() for std in self.user_std_layers]
            # user_key_std_layers = [std['key'].cpu() for std in self.user_std_layers]
            # user_value_std_layers = [std['value'].cpu() for std in self.user_std_layers]
            # plt.figure()
            # plt.title('Standard Deviation of QKVS(user), segment length = 4')
            # plt.plot(user_state_std_layers,label='state std')
            # plt.plot(user_query_std_layers,label='query std')
            # plt.plot(user_key_std_layers,label='key std')
            # plt.plot(user_value_std_layers,label='value std')
            # plt.xlabel('Layer')
            # plt.ylabel('Standard Deviation')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('tempt/user_std_layers.png',dpi=300)
            # plt.close()
            from ipdb import set_trace; set_trace()
        self.attention_maps = []

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
        self.cot_decoding = config.cot_decoding
        # Initialize weights and apply final processing
        self.post_init()
        self.latency = []

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
        sample_ids = None, # added by jieneng
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
        
        if self.model.progressive:
            self.model.step_stride_and_layer()

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
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels,None
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            images = images.permute(1, 0, 2, 3, 4)
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        
        if self.model.groupingLayer == -1:
            if self.model.grouping == 'avgpool1d':
                image_features = torch.nn.functional.avg_pool1d(image_features.permute(0,2,1), kernel_size=self.model.stride, stride=self.model.stride).permute(0,2,1).contiguous()
            elif self.model.grouping == 'avgpool2d':
                B,Q,C = image_features.shape
                image_features,p_ids1,p_ids2 = unflatten_image_features(image_features.permute(0,2,1), torch.zeros(B,Q).to(dtype=torch.int))
                p_ids1 = torch.nn.functional.avg_pool2d(p_ids1, kernel_size=self.model.stride, stride=self.model.stride)
                p_ids2 = torch.nn.functional.avg_pool2d(p_ids2, kernel_size=self.model.stride, stride=self.model.stride)
                image_features = torch.nn.functional.avg_pool2d(image_features, kernel_size=self.model.stride, stride=self.model.stride)
                
                image_features,_ = flatten_image_features(image_features,  p_ids1, p_ids2,torch.zeros(B,Q))
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        images_idx = [torch.where(cur_input_ids == IMAGE_TOKEN_INDEX) for cur_input_ids in _input_ids]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, images_idx
    

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


    
