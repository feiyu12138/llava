"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from llava.model.processor.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import ChannelDimension, infer_channel_dimension_format


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, image_mean=None, image_std=None):
        if image_mean is None:
            image_mean = (0.48145466, 0.4578275, 0.40821073)
        if image_std is None:
            image_std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(image_mean, image_std) 


class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = getattr(cfg,"prompt", "")
        max_words = getattr(cfg,"max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption

class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, image_mean=None, image_std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(image_mean=image_mean, image_std=image_std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size,image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)
    
    def preprocess(self, item,return_tensors='pt',
                   data_format=ChannelDimension.FIRST):
        item = self.transform(item)
        item = item.permute(0, 3, 1, 2)
        data = {"pixel_values":item}
        return BatchFeature(data,tensor_type=return_tensors)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = getattr(cfg,"image_size", 224)

        image_mean = getattr(cfg,"image_mean", None)
        image_std = getattr(cfg,"image_std", None)

        min_scale = getattr(cfg,"min_scale", 0.5)
        max_scale = getattr(cfg,"max_scale", 1.0)

        return cls(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, image_mean=None, image_std=None):
        super().__init__(image_mean=image_mean, image_std=image_std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)
    
    def preprocess(self, item,return_tensors='pt',
                   data_format=ChannelDimension.FIRST):
        item = self.transform(item).unsqueeze(0).bfloat16()
        
        data = {"pixel_values":item}
        return data

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = getattr(cfg,"image_size", 224)

        image_mean = getattr(cfg,"image_mean", None)
        image_std = getattr(cfg,"image_std", None)

        return cls(image_size=image_size, image_mean=image_mean, image_std=image_std)
