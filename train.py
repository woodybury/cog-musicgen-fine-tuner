# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Entry point for dora to launch solvers for running training loops.
See more info on how to use dora: https://github.com/facebookresearch/dora
"""

import logging
import multiprocessing
import os
import os.path
import sys
import typing as tp

import subprocess
import datetime 
from cog import BaseModel, Input, Path
from zipfile import ZipFile
import shutil
import subprocess as sp

import dora
from dora import git_save
from dora.distrib import init
import flashy
import hydra
import omegaconf

import tarfile

import torch

from audiocraft.environment import AudioCraftEnvironment
from audiocraft.utils.cluster import get_slurm_parameters

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

class TrainingOutput(BaseModel):
    weights: Path

def prepare_data(
        dataset_path: Path,
        target_path: str = 'src/train_data',
        one_same_description: str = None,
        meta_path: str = 'src/meta',
        auto_labeling: bool = True):
    # decompress file at dataset_path
    if str(dataset_path).rsplit('.', 1)[1] == 'zip':
        subprocess.run(['unzip', str(dataset_path), '-d', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tar':
        subprocess.run(['tar', '-xvf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'gz':
        subprocess.run(['tar', '-xvzf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tgz':
        subprocess.run(['tar', '-xzvf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] in ['wav', 'mp3', 'flac']:
        os.move(str(dataset_path), target_path + '/' + str(dataset_path.name))
    else:
        raise Exception("Not supported compression file type. The file type should be one of 'zip', 'tar', 'tar.gz', 'tgz' types of compression file, or a single 'wav', 'mp3', 'flac' types of audio file.")
    
    from pydub import AudioSegment
    
    for filename in os.listdir(target_path):
        if filename.endswith(('.mp3', '.wav', '.flac')):
            # move original file out of the way
            audio = AudioSegment.from_file(target_path + '/' + filename)

            # resample
            audio = audio.set_frame_rate(44100)

            # split into 30-second chunks
            for i in range(0, len(audio), 30000):
                chunk = audio[i:i+30000]
                chunk.export(f"{target_path + '/' + filename[:-4]}_chunk{i//1000}.wav", format="wav")
            os.remove(target_path + '/' + filename)

    import json
    import audiocraft.data.audio_dataset

    meta = audiocraft.data.audio_dataset.find_audio_files(target_path, audiocraft.data.audio_dataset.DEFAULT_EXTS, progress=True, resolve=False, minimal=True, workers=10)
    max_sample_rate = 0
    for m in meta:
        if m.sample_rate > max_sample_rate:
            max_sample_rate = m.sample_rate
        fdict = {
            "key": "",
            "artist": "",
            "sample_rate": m.sample_rate,
            "file_extension": m.path.rsplit('.', 1)[1],
            "description": "",
            "keywords": "",
            "duration": m.duration,
            "bpm": "",
            "genre": "",
            "title": "",
            "name": Path(m.path).name.rsplit('.', 1)[0],
            "instrument": "",
            "moods": []
        }
        with open(m.path.rsplit('.', 1)[0] + '.json', "w") as file:
            json.dump(fdict, file)
    audiocraft.data.audio_dataset.save_audio_meta(meta_path + '/data.jsonl', meta)
    
    d_path = Path(target_path)
    d_path.mkdir(exist_ok=True, parents=True)
    audios = list(d_path.rglob('*.mp3')) + list(d_path.rglob('*.wav'))

    for audio in list(audios):
        jsonf = open(str(audio).rsplit('.', 1)[0] + '.json', 'r')
        fdict = json.load(jsonf)
        jsonf.close()
        
        assert Path(str(audio).rsplit('.', 1)[0] + '.txt').exists() or one_same_description is not None

        if one_same_description is not None:
            fdict["description"] = one_same_description
        else:
            f = open(str(audio).rsplit('.', 1)[0] + '.txt', 'r')
            line = f.readline()
            f.close()
            fdict["description"] = line

        with open(str(audio).rsplit('.', 1)[0] + '.json', "w") as file:
            json.dump(fdict, file)

    return max_sample_rate, len(meta)

def train(
        dataset_path: Path = Input("Path to dataset directory. Input audio files will be chunked into multiple 30 second audio files. Must be one of 'tar', 'tar.gz', 'gz', 'zip' types of compressed file, or a single 'wav', 'mp3', 'flac' file.",),
        auto_labeling: bool = Input(description="Creating label data like genre, mood, theme, instrumentation, key, bpm for each track. Using `essentia-tensorflow` for music information retrieval.", default=True),
        one_same_description: str = Input(description="A description for all of audio data", default=None),
        model_version: str = Input(description="Model version to train.", default="small", choices=["melody", "small", "medium"]),
        epochs: int = Input(description="Number of epochs to train for", default=3),
        updates_per_epoch: int = Input(description="Number of iterations for one epoch", default=100),
        batch_size: int = Input(description="Batch size. Must be multiple of 8(number of gpus), for 8-gpu training.", default=20),
        optimizer: str = Input(description="Type of optimizer.", default='dadam', choices=["dadam", "adamw"]),
        lr: float = Input(description="Learning rate", default=1),
        lr_scheduler: str = Input(description="Type of lr_scheduler", default="cosine", choices=["exponential", "cosine", "polynomial_decay", "inverse_sqrt", "linear_warmup"]),
        warmup: int = Input(description="Warmup of lr_scheduler", default=0),
        cfg_p: float = Input(description="CFG dropout ratio", default=0.3),
) -> TrainingOutput:
    
    meta_path = 'src/meta'
    target_path = 'src/train_data'
    
    max_sample_rate, len_dataset = prepare_data(dataset_path, target_path, one_same_description, meta_path, auto_labeling)

    # cfg = omegaconf.OmegaConf.load("flatconfig_" + model_version + ".yaml")
        
    # cfg.datasource.max_sample_rate = max_sample_rate
    # cfg.datasource.train = meta_path
    # cfg.dataset.train.num_samples = len_dataset
    # cfg.optim.epochs = epochs
    # cfg.optim.lr = lr
    # cfg.schedule.lr_scheduler = lr_scheduler
    # cfg.schedule.cosine.warmup = warmup
    # cfg.schedule.polynomial_decay.warmup = warmup
    # cfg.schedule.inverse_sqrt.warmup = warmup
    # cfg.schedule.linear_warmup.warmup = warmup
    # cfg.classifier_free_guidance.training_dropout = cfg_p
    # cfg.logging.log_updates = updates_per_epoch//10
    # cfg.dataset.batch_size = batch_size
    # if updates_per_epoch is None:
    #     cfg.dataset.train.permutation_on_files = False
    #     cfg.optim.updates_per_epoch = 1
    # else:
    #     cfg.dataset.train.permutation_on_files = True
    #     cfg.optim.updates_per_epoch = updates_per_epoch

    if model_version == "medium":
        batch_size = 8
        print(f"Batch size is reset to {batch_size}, since `medium` model can only be trained with 8 with current GPU settings.")

    if batch_size % 8 != 0:
        batch_size = batch_size - (batch_size%8)
        print(f"Batch size is reset to {batch_size}, the multiple of 8(number of gpus).")

    # Setting up dora args
    if model_version != "melody":
        solver = "musicgen/musicgen_base_32khz"
        model_scale = model_version
        conditioner = "text2music"
    else:
        solver = "musicgen/musicgen_melody_32khz"
        model_scale = "medium"
        conditioner = "chroma2music"
    continue_from = f"//pretrained/facebook/musicgen-{model_version}"

    args = ["run", "-d", "--", f"solver={solver}", f"model/lm/model_scale={model_scale}", f"continue_from={continue_from}", f"conditioner={conditioner}"]
    args.append(f"datasource.max_sample_rate={max_sample_rate}")
    args.append(f"datasource.train={meta_path}")
    args.append(f"dataset.train.num_samples={len_dataset}")
    args.append(f"optim.epochs={epochs}")
    args.append(f"optim.lr={lr}")
    args.append(f"schedule.lr_scheduler={lr_scheduler}")
    args.append(f"schedule.cosine.warmup={warmup}")
    args.append(f"schedule.polynomial_decay.warmup={warmup}")
    args.append(f"schedule.inverse_sqrt.warmup={warmup}")
    args.append(f"schedule.linear_warmup.warmup={warmup}")
    args.append(f"classifier_free_guidance.training_dropout={cfg_p}")
    if updates_per_epoch is not None:
        args.append(f"logging.log_updates={updates_per_epoch//10 if updates_per_epoch//10 >=1 else 1}")
    else:
        args.append(f"logging.log_updates=0")
    args.append(f"dataset.batch_size={batch_size}")
    args.append(f"optim.optimizer={optimizer}")

    if updates_per_epoch is None:
        args.append("dataset.train.permutation_on_files=False")
        args.append("optim.updates_per_epoch=1")
    else:
        args.append("dataset.train.permutation_on_files=True")
        args.append(f"optim.updates_per_epoch={updates_per_epoch}")

    # dora_main.main(args)
    sp.call(["dora"]+args)
    # directory = Path(output_dir)
    # directory = Path(str(solver.checkpoint_path()))

    for dirpath, dirnames, filenames in os.walk("tmp"):
        for filename in [f for f in filenames if f == "checkpoint.th"]:
            checkpoint_dir = os.path.join(dirpath, filename)
    
    loaded = torch.load(checkpoint_dir)
    out_path = "trained_model.tar"

    if os.path.isfile(out_path):
        os.remove(out_path)

    torch.save({'xp.cfg': loaded["xp.cfg"], "model": loaded["model"]}, out_path)
    # print(directory.parent)
    # print(directory.name)
    
    # serializer = TensorSerializer(MODEL_OUT)
    # serializer.write_module(solver.model)
    # serializer.close()

    # with tarfile.open(out_path, "w") as tar:
    #     tar.add(directory, arcname=directory.name)

    # out_path = "training_output.zip"
    # with ZipFile(out_path, "w") as zip:
    #     for file_path in directory.rglob("*"):
    #         print(file_path)
    #         zip.write(file_path, arcname=file_path.relative_to(directory))

    import shutil
    shutil.rmtree('src/meta')
    shutil.rmtree('src/train_data')
    shutil.rmtree('tmp')

    return TrainingOutput(weights=Path(out_path))

# main.dora.dir = AudioCraftEnvironment.get_dora_dir()
# main._base_cfg.slurm = get_slurm_parameters(main._base_cfg.slurm)

# if main.dora.shared is not None and not os.access(main.dora.shared, os.R_OK):
#     print("No read permission on dora.shared folder, ignoring it.", file=sys.stderr)
#     main.dora.shared = None

# if __name__ == '__main__':
    #pp()