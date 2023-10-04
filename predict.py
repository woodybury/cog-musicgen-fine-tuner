# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH


import shutil

from tempfile import TemporaryDirectory
# from pathlib import Path
from distutils.dir_util import copy_tree
from typing import Optional
from cog import BasePredictor, Input, Path
import torch
import datetime

# Model specific imports
import torchaudio
import subprocess
import typing as tp
import numpy as np

from audiocraft.models import MusicGen
from audiocraft.models.musicgen import _HF_MODEL_CHECKPOINTS_MAP as HF_MODEL_CHECKPOINTS_MAP
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from audiocraft.data.audio import audio_write

from audiocraft.models.builders import get_lm_model, get_compression_model, get_wrapped_compression_model
from omegaconf import OmegaConf

from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
import re
import time
import subprocess
import logging

def _delete_param(cfg, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)

def load_ckpt(path, device):
    loaded = torch.hub.load_state_dict_from_url(str(path))
    cfg = OmegaConf.create(loaded['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_chord.cache_path')
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    lm = get_lm_model(loaded['xp.cfg'])
    lm.load_state_dict(loaded['model']) 
    lm.eval()
    lm.cfg = cfg
    compression_model = CompressionSolver.wrapped_model_from_checkpoint(cfg, cfg.compression_model_checkpoint, device=device)
    return MusicGen(f"{os.getenv('COG_USERNAME')}/musicgen-finetuned", compression_model, lm)

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if str(weights) == "weights":
            weights = None
        
        self.melody_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-melody",
        )

        self.large_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-large",
        )

        self.medium_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-medium",
        )

        self.small_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-small",
        )

        if weights is not None:
            # self.my_model = MusicGen.get_pretrained(weights)
            # self.my_model = self.load_tensorizer(weights, model_version)
            self.my_model = load_ckpt(weights, self.device)

    def load_tensorizer(self, weights, model_version):
        # st = time.time()
        # weights = str(weights)
        # print("loadin")
        # print(weights)
        # pattern = r"https://pbxt\.replicate\.delivery/([^/]+/[^/]+)"
        # match = re.search(pattern, weights)
        # if match:
        #     weights = f"gs://replicate-files/{match.group(1)}"

        # print(f"deserializing weights")
        # local_weights = "/src/musicgen_tensors"
        # command = f"/gc/google-cloud-sdk/bin/gcloud storage cp {weights} {local_weights}".split()
        # res = subprocess.run(command)
        # if res.returncode != 0:
        #     raise Exception(
        #         f"gcloud storage cp command failed with return code {res.returncode}: {res.stderr.decode('utf-8')}"
        #     )

        logging.disable(logging.WARN)
        model = no_init_or_tensor(
            lambda: MusicGen.get_pretrained(f'facebook/musicgen-{model_version}')
        )
        logging.disable(logging.NOTSET)

        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model.lm)
        print(f"weights loaded in {time.time() - st}")
        return model

    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device

        compression_model = load_compression_model(
            model_id, device=device, cache_dir=model_path
        )
        lm = load_lm_model(model_id, device=device, cache_dir=model_path)

        return MusicGen(model_id, compression_model, lm)

    def predict(
        self,
        model_version: str = Input(
            description="Model to use for generation. If set to 'encode-decode', the audio specified via 'input_audio' will simply be encoded and then decoded.",
            default="finetuned",
            choices=["melody", "small", "medium", "large", "encode-decode", "finetuned"],
        ),
        prompt: str = Input(
            description="A description of the music you want to generate.", default=None
        ),
        input_audio: Path = Input(
            description="An audio file that will influence the generated music. If `continuation` is `True`, the generated music will be a continuation of the audio file. Otherwise, the generated music will mimic the audio file's melody.",
            default=None,
        ),
        duration: int = Input(
            description="Duration of the generated audio in seconds.", default=8, le=30
        ),
        continuation: bool = Input(
            description="If `True`, generated music will continue `melody`. Otherwise, generated music will mimic `audio_input`'s melody.",
            default=False,
        ),
        continuation_start: int = Input(
            description="Start time of the audio file to use for continuation.",
            default=0,
            ge=0,
        ),
        continuation_end: int = Input(
            description="End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip.",
            default=None,
            ge=0,
        ),
        normalization_strategy: str = Input(
            description="Strategy for normalizing audio.",
            default="loudness",
            choices=["loudness", "clip", "peak", "rms"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If None or -1, a random seed will be used.",
            default=None,
        ),
    ) -> Path:

        if prompt is None and input_audio is None:
            raise ValueError("Must provide either prompt or input_audio")
        if continuation and not input_audio:
            raise ValueError("Must provide `input_audio` if continuation is `True`.")
        if model_version == "large" and input_audio and not continuation:
            raise ValueError(
                "Large model does not support melody input. Set `model_version='melody'` to condition on audio input."
            )
        elif model_version == "medium" and input_audio and not continuation:
            raise ValueError(
                "Medium model does not support melody input. Set `model_version='melody'` to condition on audio input."
            )
        elif model_version == "small" and input_audio and not continuation:
            raise ValueError(
                "Small model does not support melody input. Set `model_version='melody'` to condition on audio input."
            )
        elif model_version == "finetuned":
            try:
                self.my_model
            except:
                raise NameError(
                    "There is no fine-tuned 'weight' file found."
                )

        if model_version == "melody":
            model = self.melody_model
        elif model_version == "large":
            model = self.large_model
        elif model_version == "medium":
            model = self.medium_model
        elif model_version == "small":
            model = self.small_model
        elif model_version == "finetuned":
            model = self.my_model

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        if not input_audio:
            set_generation_params(duration)
            wav = model.generate([prompt], progress=True)

        elif model_version == "encode-decode":
            encoded_audio = self._preprocess_audio(input_audio, model)
            set_generation_params(duration)
            wav = model.compression_model.decode(encoded_audio).squeeze(0)

        else:
            input_audio, sr = torchaudio.load(input_audio)
            input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

            continuation_start = 0 if not continuation_start else continuation_start
            if continuation_end is None or continuation_end == -1:
                continuation_end = input_audio.shape[2] / sr

            if continuation_start > continuation_end:
                raise ValueError(
                    "`continuation_start` must be less than or equal to `continuation_end`"
                )

            input_audio_wavform = input_audio[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            input_audio_duration = input_audio_wavform.shape[-1] / sr

            if continuation:
                if (
                    duration + input_audio_duration
                    > model.lm.cfg.dataset.segment_duration
                ):
                    raise ValueError(
                        "duration + continuation duration must be <= 30 seconds"
                    )

                set_generation_params(duration + input_audio_duration)
                wav = model.generate_continuation(
                    prompt=input_audio_wavform,
                    prompt_sample_rate=sr,
                    descriptions=[prompt],
                    progress=True,
                )

            else:
                set_generation_params(duration)
                wav = model.generate_with_chroma(
                    [prompt], input_audio_wavform, sr, progress=True
                )

        audio_write(
            "out",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )
        wav_path = "out.wav"

        if output_format == "mp3":
            mp3_path = "out.mp3"
            subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)

    def _preprocess_audio(
        audio_path, model: MusicGen, duration: tp.Optional[int] = None
    ):

        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)

        # Calculate duration in seconds if not provided
        if duration is None:
            duration = wav.shape[1] / model.sample_rate

        # Check if duration is more than 30 seconds
        if duration > 30:
            raise ValueError("Duration cannot be more than 30 seconds")

        end_sample = int(model.sample_rate * duration)
        wav = wav[:, :end_sample]

        assert wav.shape[0] == 1
        assert wav.shape[1] == model.sample_rate * duration

        wav = wav.cuda()
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            gen_audio = model.compression_model.encode(wav)

        codes, scale = gen_audio

        assert scale is None

        return codes


# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
