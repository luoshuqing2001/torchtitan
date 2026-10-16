# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torchtitan.models.common.decoder_sharding import set_decoder_sharding_config
from torchtitan.models.muse_glimmer.sharding import _set_muse_glimmer_layer_sharding

if TYPE_CHECKING:
    from .model import GlimmerDenseModel


def set_glimmer_dense_sharding_config(
    config: "GlimmerDenseModel.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill sharding configs for Muse Glimmer attention and a dense SwiGLU."""
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_muse_glimmer_layer_sharding(layer_cfg, enable_sp=enable_sp)
