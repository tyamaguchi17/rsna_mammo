from logging import getLogger

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.nn.backbone import load_backbone
from src.nn.backbones.base import BackboneBase
from src.nn.heads.arc_face import ChannelWiseGeM, GeM
from src.utils.checkpoint import get_weights_to_load

logger = getLogger(__name__)


def init_model_from_config(cfg: DictConfig, pretrained: bool):
    model = nn.Sequential()
    backbone = init_backbone(cfg, pretrained=pretrained)
    forward_features = nn.Sequential()

    forward_features.add_module("backbone", backbone)
    if cfg.pool.type == "adaptive":
        forward_features.add_module("pool", nn.AdaptiveAvgPool2d((1, 1)))
        forward_features.add_module("flatten", nn.Flatten())
    elif cfg.pool.type == "gem":
        forward_features.add_module(
            "pool", GeM(p=cfg.pool.p, p_trainable=cfg.pool.p_trainable)
        )
        forward_features.add_module("flatten", nn.Flatten())
    elif cfg.pool.type == "gem_ch":
        forward_features.add_module(
            "pool",
            ChannelWiseGeM(
                dim=backbone.out_features,
                p=cfg.pool.p,
                requires_grad=cfg.pool.p_trainable,
            ),
        )
        forward_features.add_module("flatten", nn.Flatten())

    if cfg.use_bn:
        forward_features.add_module("normalize", nn.BatchNorm1d(backbone.out_features))
        forward_features.add_module("relu", torch.nn.PReLU())

    model.add_module("forward_features", forward_features)
    if cfg.head.type == "linear":
        head = nn.Linear(backbone.out_features, cfg.output_dim, bias=True)
    else:
        raise ValueError(f"{cfg.head.type} is not implemented")
    model.add_module("head", head)

    if cfg.restore_path is not None:
        logger.info(f'Loading weights from "{cfg.restore_path}"...')
        ckpt = torch.load(cfg.restore_path, map_location="cpu")
        model_dict = get_weights_to_load(model, ckpt)
        model.load_state_dict(model_dict, strict=True)

    return model


def init_backbone(cfg: DictConfig, pretrained: bool) -> BackboneBase:
    in_chans = cfg.in_chans
    backbone = load_backbone(
        base_model=cfg.base_model,
        pretrained=pretrained,
        in_chans=in_chans,
    )
    if cfg.grad_checkpointing:
        backbone.set_grad_checkpointing()
    if cfg.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone
