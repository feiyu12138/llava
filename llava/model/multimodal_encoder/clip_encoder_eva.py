import os

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from llava.model.processor.blip_processors import Blip2ImageTrainProcessor, Blip2ImageEvalProcessor

from llava.dist_utils import download_cached_file, is_url
from llava.model.multimodal_encoder.Qformer import BertConfig, BertLMHeadModel
import logging
import contextlib

from .eva_vit import create_eva_vit_g, LayerNorm
import logging

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class CLIPVisionTowerEva(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.is_train = getattr(args, 'train_processor', False)
        self.args = args
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        
    
        
    def init_vision_encoder(
        self, model_name='eva_clip_g', img_size=224, drop_path_rate=0.0, use_grad_checkpoint=False, precision="fp16", freeze=True
    ):
        logging.info('Loading VIT')

        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        if not freeze:
            precision = "fp32"  # fp16 is not for training

        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )

        ln_vision = LayerNorm(visual_encoder.num_features)

        if freeze:
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        visual_encoder.config = self.load_config(visual_encoder,img_size)
        

        logging.info('Loading VIT Done')
        return visual_encoder, ln_vision
    
    def load_config(self,visual_encoder,img_size):
        patch_size = visual_encoder.patch_embed.patch_size[0]
        hidden_size = visual_encoder.num_features
        image_size = img_size
        intermediate_size = visual_encoder.blocks[0].mlp.fc1.out_features
        projection_dim = visual_encoder.num_features
        num_hidden_layers = len(visual_encoder.blocks)
        num_attention_heads = 16 #TODO: check
        num_channels = 3
        initializer_range = 0.01 #TODO: check
        initializer_factor = 1.0 #TODO: check
        attention_dropout = 0.0 #TODO: check
        layer_norm_eps = visual_encoder.blocks[0].norm1.eps
        hidden_act = "gelu" #TODO: check
        clipConfig = CLIPVisionConfig(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      num_hidden_layers=num_hidden_layers,
                                      num_channels=num_channels,
                                      hidden_act=hidden_act,
                                      intermediate_size=intermediate_size,
                                      hidden_dropout_prob=0.0,
                                      attention_probs_dropout_prob=attention_dropout,
                                      initializer_range=initializer_range,
                                      layer_norm_eps=layer_norm_eps,
                                      initializer_factor=initializer_factor,
                                      image_size=image_size,
                                      patch_size=patch_size,
                                      projection_dim=projection_dim)
        return clipConfig
        
        

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        Processor = Blip2ImageTrainProcessor if self.is_train else Blip2ImageEvalProcessor
        self.image_processor = Processor.from_config(self.args)
        self.vision_tower, self.ln_vision = self.init_vision_encoder()
        vision_tower_old = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.vision_tower.to(dtype=vision_tower_old.dtype, device=vision_tower_old.device)
        self.vision_tower.device = vision_tower_old.device
        self.vision_tower.dtype = vision_tower_old.dtype
        if self.args.has_qformer: #TODO: load after vision tower
            self.Qformer, self.query_tokens = self.init_Qformer(
                    self.args.num_query_token, self.vision_tower.num_features, self.args.freeze_qformer
                )
            q_former_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
            self.load_from_pretrained(url_or_filename=q_former_model)
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs): #TODO: Not sure if this is correct
        image_features = image_forward_outs.last_hidden_state
        # if self.select_feature == 'patch':
        #     image_features = image_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        return image_features
    
    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
    
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens
    
    def encode_img_qformer(self, image):
        device = image.device
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast(self.dtype):
            image_embeds = self.ln_vision(self.vision_tower(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            inputs_before_proj = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        return inputs_before_proj

    @torch.no_grad()
    def forward(self, images):
        self.call = self.vision_tower.__call__ if not self.args.has_qformer else self.encode_img_qformer
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.call(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.call(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features  
    
     
    
    def preprocess(self, images): #TODO: must implement
        return self.image_processor(images)
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.blocks[0].mlp.fc1.weight.dtype

    @property
    def device(self):
        # return self.vision_tower.device
        return self.vision_tower.blocks[0].mlp.fc1.weight.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


'''
class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
'''
