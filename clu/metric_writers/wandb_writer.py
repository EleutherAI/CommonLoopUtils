# Copyright 2022 The CLU Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetricWriter for writing to TF summary files.

Only works in eager mode. Does not work for Pytorch code, please use
TorchTensorboardWriter instead.
"""

from typing import Any, Mapping, Optional
from absl import logging


from clu.internal import utils
from clu.metric_writers import interface
import tensorflow as tf

import wandb

import socket

Array = interface.Array
Scalar = interface.Scalar


class WandbWriter(interface.MetricWriter):
  """MetricWriter that writes TF summary files."""

  def __init__(self, logdir: str):
    super().__init__()

    # pass wandb key and other parameters via env variables 
    # TODO: change interface so these can be passed via gin?
    api_key = os.environ.get('WANDB_API_KEY', None)
    wandb_team = os.environ.get('WANDB_TEAM', None)
    wandb_group = os.environ.get('WANDB_GROUP', None)
    # add unique id to wandb runs (avoid duplicate names)
    if not wandb_group:
        wandb_group = shortuuid.uuid()
    else:
        wandb_group += "_" + wandb.util.generate_id()
    
    if api_key is not None:
        name = name = f"{socket.gethostname()}-{local_rank()}" if group_name else None
        wandb.init(
                project=neox_args.wandb_project,
                group=group_name,
                name=name,
                save_code=False,
                force=False,
                entity=wandb_team,
            )

  def write_summaries(
      self, step: int,
      values: Mapping[str, Array],
      metadata: Optional[Mapping[str, Any]] = None):
    
    for key, value in values.items():
        # TODO: handle metadata somehow? (compare with summary_writer file)
        wandb.log({key: value}, step=step)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    for key, value in scalars.items():
        wandb.log({key: value}, step=step)

  def write_images(self, step: int, images: Mapping[str, Array]):
    raise NotImplementedError(
        "Writing images to WandB not supported by this class!"
    )

  def write_videos(self, step: int, videos: Mapping[str, Array]):
    logging.log_first_n(
        logging.WARNING,
        "WandbWriter does not support writing videos.", 1)

  def write_audios(
      self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
    raise NotImplementedError(
        "Writing audio to WandB not supported by this class!"
    )

  def write_texts(self, step: int, texts: Mapping[str, str]):
    raise NotImplementedError(
        "Writing audio to WandB not supported by this class!"
    )

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    raise NotImplementedError(
        "Writing histogram to WandB not supported by this class!"
    )

  def write_hparams(self, hparams: Mapping[str, Any]):
    for k, v in hparams.items():
        wandb.config.update({k: v})

  def flush(self):
    pass

  def close(self):
    pass
