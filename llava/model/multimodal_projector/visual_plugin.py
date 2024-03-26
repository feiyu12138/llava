import torch
import torch.nn as nn
from functools import partial
from timm.models.regnet import RegStage
from timm.layers import LayerNorm2d
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
)
from einops import rearrange

def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Abstractor(nn.Module):
    def __init__(self, hidden_dim, num_pre_layers, num_post_layers, pool_stride, grouping):
        super(Abstractor, self).__init__()
        self.type = grouping.split('_')[0] # option: cabstractor, dabstractor
        self.is_gate = grouping.find('gate')!=-1
        
        if self.type == 'cabstractor':
            RegBlock = partial(
                RegStage,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
            s1 = RegBlock(
                num_pre_layers,
                hidden_dim,
                hidden_dim,
            )
            s2 = RegBlock(
                num_post_layers,
                hidden_dim,
                hidden_dim,
            )
            sampler = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
            self.net = nn.Sequential(s1, sampler, s2)
        elif self.type == 'dabstractor':
            self.net = nn.Identity()
        elif self.type == 'DWConvabstractor':
            depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=pool_stride+1, stride=pool_stride, padding=pool_stride//2, groups=hidden_dim, bias=False)
            norm = LayerNorm2d(hidden_dim)
            act = nn.GELU()
            self.net = nn.Sequential(depthwise, norm, act)
        elif self.type == 'DWKSabstractor':
            depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=pool_stride, stride=pool_stride, padding=0, groups=hidden_dim, bias=False)
            norm = LayerNorm2d(hidden_dim)
            act = nn.GELU()
            self.net = nn.Sequential(depthwise, norm, act)
        else:
            self.net = nn.Identity()

        if self.is_gate:
            self.pooler = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
            self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self,x):
        if self.is_gate:
            x = self.net(x) * self.gate.tanh() + self.pooler(x)
        else:
            x = self.net(x)
        return x


class DAbstractor(nn.Module):
    def __init__(self,config, num_feature_levels,decoder_layers ):
        super(Abstractor, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(config) for _ in range(decoder_layers)]
        )
    def _get_query_reference_points(self, spatial_shapes, valid_ratios):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        steps = int(self.num_queries**0.5)
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, steps, dtype=torch.float32),
                torch.linspace(0.5, width - 0.5, steps, dtype=torch.float32),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points.squeeze(2)

    def _forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        intermediate = ()
        intermediate_reference_points = ()

        for _, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _process_v_features(self, visual_feat):
        # visual_feat: [B, len, dim] or [B, lvls, len, dim]
        if self.except_cls:
            visual_feat = visual_feat[:, :, 1:] if self.isMs else visual_feat[:, 1:]

        if self.isMs:
            visual_feats = []
            for level in range(self.num_feature_levels):
                visual_feats.append(self.input_proj[level](visual_feat[:, level]))
            visual_feat = torch.stack(visual_feats, 1)

            # add pos emb [1, len, dim]
            if self.v_pos_emb is not None:
                visual_feat = visual_feat + self.v_pos_emb.unsqueeze(1)

            # add lvl emb [1, lvls, 1, dim]
            visual_feat = visual_feat + self.level_emb
            visual_feat = visual_feat.flatten(1, 2)  # [B, lvls, v_len, dim] -> [B, lvls*v_len, dim]
        else:
            visual_feat = self.input_proj[0](visual_feat)
            if self.v_pos_emb is not None:
                visual_feat = visual_feat + self.v_pos_emb

        return visual_feat

    def _convert_dtype_device(self, tgt_feat, dtype=None, device=None):
        # tgt_feat: target tensor to be converted
        _dtype = tgt_feat.dtype if dtype is None else dtype
        _device = tgt_feat.device if device is None else device

        tgt_feat = tgt_feat.type(_dtype).to(_device)

        return tgt_feat

    def _prepare_ddetr_inputs(self, batch_size, seq_len, lvls, dtype=None, device=None):
        # assume there are no paddings in a feature map
        valid_ratios = torch.ones(batch_size, lvls, 2)

        # assume all feature maps have the same sequence length (i.e., the same shape)
        spatial_shapes = torch.tensor([int(seq_len**0.5), int(seq_len**0.5)]).repeat(lvls, 1)
        level_start_index = torch.arange(0, seq_len * lvls, seq_len)

        if dtype is not None and device is not None:
            valid_ratios = self._convert_dtype_device(valid_ratios, dtype=dtype, device=device)
            spatial_shapes = self._convert_dtype_device(
                spatial_shapes, dtype=torch.long, device=device
            )
            level_start_index = self._convert_dtype_device(
                level_start_index, dtype=torch.long, device=device
            )

        return valid_ratios, spatial_shapes, level_start_index

    def _make_pooled_queries(self, visual_feat):
        assert (
            self.num_feature_levels == 1
        )  # currently do not support multi-scale features for the v-pooled Q

        batch_size, seq_len, h_dim = visual_feat.shape
        query_embeds = self.query_position_embeddings.weight
        if self.pooled_v_target != "none":
            hw_v = int(seq_len**0.5)
            hw_q = int(self.num_queries**0.5)
            visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw_v, w=hw_v)
            if self.pooled_v_target == "tgt":
                query_embed = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                target = self.downsampler(visual_feat)
                target = rearrange(target, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
            else:
                target = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                query_embed = self.downsampler(visual_feat)
                query_embed = rearrange(query_embed, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
        else:
            query_embed, target = torch.split(query_embeds, h_dim, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embed, target

    def forward(self, visual_feat):
        
        # deformable attention only supports fp32
        original_dtype = visual_feat.type()
        visual_feat = visual_feat.type(torch.cuda.FloatTensor)
        visual_feat = self._process_v_features(visual_feat)

        batch_size, seq_len, h_dim = visual_feat.shape
        seq_len /= self.num_feature_levels

        query_embed, target = self._make_pooled_queries(visual_feat)
        reference_points = self.reference_points.expand(batch_size, -1, -1)

        valid_ratios, spatial_shapes, level_start_index = self._prepare_ddetr_inputs(
            batch_size, seq_len, self.num_feature_levels, visual_feat.dtype, visual_feat.device
        )

        decoder_outputs_dict = self._forward(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=visual_feat,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            return_dict=True,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        decoder_outputs = decoder_outputs_dict.last_hidden_state

        if self.eos_tokens is not None:
            decoder_outputs = torch.cat(
                [decoder_outputs, self.eos_tokens.expand(batch_size, -1, -1)], dim=1
            )

        decoder_outputs = self.output_proj(decoder_outputs)
        decoder_outputs = decoder_outputs.type(original_dtype)
        return decoder_outputs
    
if __name__ == '__main__':
    model = Abstractor(32, 3, 3, 4, 'DWConvabstractor_gate')
    x = torch.randn(1, 32, 24, 24)
    y = model(x)
    print(y.shape)
        