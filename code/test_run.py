import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

# import datasets
# import evaluate
import torch
# from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from llm_trainer import LLMTrainer, inference_generation
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version
# xxx: 2023-03-21
import copy

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, LlamaConfig, WhisperConfig, WhisperModel, LlamaModel, LlamaTokenizer
from transformers import AutoConfig, AutoModel
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import numpy as np
import json
import pickle
import codecs
from PIL import Image
# from peft import PeftModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm, trange
from sklearn.metrics import top_k_accuracy_score
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from model import MM_LLMs, MM_LLMs_Config
# import clip
# import whisper


##########Set up the model#############
clip_config = CLIPConfig()
whisper_config = WhisperConfig()
llm_config = LlamaConfig()

model_config = MM_LLMs_Config(
    n_frames=6, attention_heads=8, image_conv_kernel=48, image_conv_stride=36, 
    video_conv_kernel=36, video_conv_stride=30, audio_conv_kernel=240, audio_conv_stride=220,
clip_config=clip_config, whisper_config=whisper_config, llm_config=llm_config)

# load model separately 
model = MM_LLMs(config=model_config)
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

#these are open source weights from hugginface"
model.video_encoder.from_pretrained("openai/clip-vit-base-patch32")
model.audio_encoder.from_pretrained("openai/whisper-tiny.en")
model.llm.from_pretrained("decapoda-research/llama-7b-hf")

model.llm.resize_token_embeddings(len(tokenizer))

meta_data = 'meta_data.json'

##function to generate outputs
output_path = 'generated_responses'
inference_generation(model, tokenizer, meta_data,output_path)
