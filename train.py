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
import sys
import typing as tp

import subprocess
import datetime 
from cog import BaseModel, Input, Path
from zipfile import ZipFile
import shutil

from dora import git_save, hydra_main, XP
import flashy
import hydra
import omegaconf

from tensorizer import TensorSerializer

from audiocraft.environment import AudioCraftEnvironment
from audiocraft.utils.cluster import get_slurm_parameters

logger = logging.getLogger(__name__)


class TrainingOutput(BaseModel):
    weights: Path
    model_version: str

MODEL_OUT = "src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
DIST_OUT_DIR = "tmp/model"

def resolve_config_dset_paths(cfg):
    """Enable Dora to load manifest from git clone repository."""
    # manifest files for the different splits
    for key, value in cfg.datasource.items():
        if isinstance(value, str):
            cfg.datasource[key] = git_save.to_absolute_path(value)

def get_solver(cfg):
    from audiocraft import solvers
    # Convert batch size to batch size for each GPU
    assert cfg.dataset.batch_size % flashy.distrib.world_size() == 0
    cfg.dataset.batch_size //= flashy.distrib.world_size()
    for split in ['train', 'valid', 'evaluate', 'generate']:
        if hasattr(cfg.dataset, split) and hasattr(cfg.dataset[split], 'batch_size'):
            assert cfg.dataset[split].batch_size % flashy.distrib.world_size() == 0
            cfg.dataset[split].batch_size //= flashy.distrib.world_size()
    resolve_config_dset_paths(cfg)
    solver = solvers.get_solver(cfg)
    return solver

def init_seed_and_system(cfg):
    import numpy as np
    import torch
    import random
    from audiocraft.modules.transformer import set_efficient_attention_backend

    # multiprocessing.set_start_method(cfg.mp_start_method)
    logger.debug('Setting mp start method to %s', cfg.mp_start_method)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # torch also initialize cuda seed if available
    torch.manual_seed(cfg.seed)
    # torch.set_num_threads(cfg.num_threads)
    # os.environ['MKL_NUM_THREADS'] = str(cfg.num_threads)
    # os.environ['OMP_NUM_THREADS'] = str(cfg.num_threads)
    # logger.debug('Setting num threads to %d', cfg.num_threads)
    set_efficient_attention_backend(cfg.efficient_attention_backend)
    logger.debug('Setting efficient attention backend to %s', cfg.efficient_attention_backend)

def prepare_data(
        dataset_path: Path,
        target_path: str = 'src/train_data',
        one_same_description: str = None,
        meta_path: str = 'src/meta'):
    # decompress file at dataset_path
    if str(dataset_path).rsplit('.', 1)[1] == 'zip':
        subprocess.run(['unzip', str(dataset_path), '-d', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tar':
        subprocess.run(['tar', '-xvf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'gz':
        subprocess.run(['tar', '-xvzf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tgz':
        subprocess.run(['tar', '-xzvf', str(dataset_path), '-C', target_path + '/'])
    else:
        raise Exception("Not supported compression file type. The file type should be one of 'zip', 'tar', 'tar.gz' or 'tgz'.")
    
    import json
    from pathlib import Path as Ppath
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
            "name": Ppath(m.path).name.rsplit('.', 1)[0],
            "instrument": "",
            "moods": []
        }
        with open(m.path.rsplit('.', 1)[0] + '.json', "w") as file:
            json.dump(fdict, file)
    audiocraft.data.audio_dataset.save_audio_meta(meta_path + '/data.jsonl', meta)
    
    d_path = Ppath(target_path)
    d_path.mkdir(exist_ok=True, parents=True)
    audios = list(d_path.rglob('*.mp3')) + list(d_path.rglob('*.wav'))

    for audio in list(audios):
        jsonf = open(str(audio).rsplit('.', 1)[0] + '.json', 'r')
        fdict = json.load(jsonf)
        jsonf.close()
        
        assert Ppath(str(audio).rsplit('.', 1)[0] + '.txt').exists() or one_same_description is not None

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
        dataset_path: Path = Input("Path to dataset directory",),
        one_same_description: str = Input(description="A description for all of audio data", default=None),
        model_version: str = Input(description="Model version to train.", default="small", choices=["melody", "small", "medium", "large"]),
        lr: float = Input(description="Learning rate", default=1),
        epochs: int = Input(description="Number of epochs to train for", default=2),
        updates_per_epoch: int = Input(description="Number of iterations for one epoch", default=10),
        save_step: int = Input(description="Save model every n steps", default=None),
        batch_size: int = Input(description="Batch size", default=9),
        lr_scheduler: str = Input(description="Type of lr_scheduler", default="cosine", choices=["exponential", "cosine", "polynomial_decay", "inverse_sqrt", "linear_warmup"]),
) -> TrainingOutput:
    
    meta_path = 'src/meta'
    target_path = 'src/train_data'

    output_dir = DIST_OUT_DIR
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    cfg = omegaconf.OmegaConf.load("flatconfig_" + model_version + ".yaml")
    
    max_sample_rate, len_dataset = prepare_data(dataset_path, target_path, one_same_description, meta_path)

    cfg.datasource.max_sample_rate = max_sample_rate
    cfg.datasource.train = meta_path
    cfg.dataset.train.num_samples = len_dataset
    cfg.optim.epochs = epochs
    cfg.optim.updates_per_epoch = updates_per_epoch
    cfg.optim.lr = lr
    cfg.logging.log_updates = updates_per_epoch//10
    cfg.dataset.batch_size = batch_size

    init_seed_and_system(cfg)

    # Setup logging both to XP specific folder, and to stderr.
    # log_name = '%s.log.{rank}' % cfg.execute_only if cfg.execute_only else 'solver.log.{rank}'
    # flashy.setup_logging(level=str(cfg.logging.level).upper(), log_name=log_name)

    # Initialize distributed training, no need to specify anything when using Dora.
    # flashy.distrib.init()

    solver = get_solver(cfg)

    if cfg.show:
        solver.show()
        return

    if cfg.execute_only:
        assert cfg.execute_inplace or cfg.continue_from is not None, \
            "Please explicitly specify the checkpoint to continue from with continue_from=<sig_or_path> " + \
            "when running with execute_only or set execute_inplace to True."
        solver.restore(replay_metrics=False)  # load checkpoint
        solver.run_one_stage(cfg.execute_only)
        return

    solver.run()

    # directory = Path(output_dir)
    directory = solver.checkpoint_path()
    serializer = TensorSerializer(MODEL_OUT)
    serializer.write_module(solver.model)
    serializer.close()

    # out_path = "training_output.zip"
    # with ZipFile(out_path, "w") as zip:
    #     for file_path in directory.rglob("*"):
    #         print(file_path)
    #         zip.write(file_path, arcname=file_path.relative_to(directory))

    return TrainingOutput(weights=Path(MODEL_OUT), model_version=model_version)

# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# main.dora.dir = AudioCraftEnvironment.get_dora_dir()
# main._base_cfg.slurm = get_slurm_parameters(main._base_cfg.slurm)

# if main.dora.shared is not None and not os.access(main.dora.shared, os.R_OK):
#     print("No read permission on dora.shared folder, ignoring it.", file=sys.stderr)
#     main.dora.shared = None

# if __name__ == '__main__':
    #pp()