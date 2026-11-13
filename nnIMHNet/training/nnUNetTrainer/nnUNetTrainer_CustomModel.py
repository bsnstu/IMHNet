# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.training.nnUNetTrainer.custom_models.vit_seg_modeling import (
    VisionTransformer as ViT_seg,
    CONFIGS as CONFIGS_ViT_seg,
)


class SegEdgeWrapper(nn.Module):
    """
    原始自定义模型输出:
        seg_output, edge_logits

    但是为了兼容 nnUNet 原始 train_step / validation_step / inference:
        forward 永远只返回 seg_output

    edge_logits 不作为 forward 返回值，而是保存为属性:
        self.last_edge_logits
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.last_edge_logits = None

    def forward(self, x):
        """
        nnUNet 看到的输出只有 seg_output。
        edge_logits 被缓存到 self.last_edge_logits。
        """

        self.last_edge_logits = None

        output = self.model(x)

        if isinstance(output, (tuple, list)):
            if len(output) < 2:
                raise RuntimeError(
                    "Your model returned tuple/list, but length < 2. "
                    "Expected output: (seg_output, edge_logits)."
                )

            seg_output = output[0]
            edge_logits = output[1]

            self.last_edge_logits = edge_logits

            return seg_output

        return output

    def get_last_edge_logits(self):
        return self.last_edge_logits


class SegWithEdgeLoss(nn.Module):
    """
    包装 nnUNet 原始 loss。

    训练阶段:
        total_loss = seg_loss + lambda_edge * edge_loss

    验证阶段:
        total_loss = seg_loss

    这样可以避免 validation 阶段由于 edge_logits 异常导致 val_loss = nan。
    """

    def __init__(
        self,
        base_loss: nn.Module,
        get_edge_logits_fn,
        is_training_fn,
        lambda_edge: float = 0.75,
        require_edge_in_training: bool = True,
        check_nan: bool = True,
    ):
        super().__init__()

        self.base_loss = base_loss
        self.get_edge_logits_fn = get_edge_logits_fn
        self.is_training_fn = is_training_fn

        self.lambda_edge = lambda_edge
        self.require_edge_in_training = require_edge_in_training
        self.check_nan = check_nan

        self.edge_criterion = nn.BCEWithLogitsLoss()

        self.last_seg_loss = None
        self.last_edge_loss = None
        self.last_total_loss = None

    @staticmethod
    def mask_to_edge_2d(mask: torch.Tensor) -> torch.Tensor:
        """
        从二值 mask 中提取边缘。

        Parameters
        ----------
        mask:
            shape [B, 1, H, W]

        Returns
        -------
        edge:
            shape [B, 1, H, W]
        """

        mask = mask.float()

        dilated = F.max_pool2d(
            mask,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        eroded = -F.max_pool2d(
            -mask,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        edge = dilated - eroded
        edge = (edge > 0).float()

        return edge

    def mask_to_edge(self, target) -> torch.Tensor:
        """
        从 nnUNet target 生成 edge_gt。

        target 可能是:
            1. Tensor: [B, 1, H, W]，单通道标签
            2. Tensor: [B, 3, H, W]，RGB保存但三个通道内容一致的标签
            3. list / tuple: deep supervision target

        虽然当前关闭 deep supervision，但这里保留兼容处理。
        """

        if isinstance(target, (list, tuple)):
            target = target[0]

        if target.ndim != 4:
            raise RuntimeError(
                "This edge loss is written for 2D input. "
                "Expected target shape [B, C, H, W], "
                f"but got {target.shape}."
            )

        # 部分PNG标签以RGB格式保存，nnUNet读取后可能得到3个相同通道。
        # 仅当R/G/B完全一致时，才能安全转换成单通道标签。
        if target.shape[1] == 3:
            channel_0 = target[:, 0:1]
            channels_are_equal = (
                torch.equal(channel_0, target[:, 1:2])
                and torch.equal(channel_0, target[:, 2:3])
            )

            if not channels_are_equal:
                raise RuntimeError(
                    "The target has 3 channels, but its RGB channels are not "
                    "identical. It cannot be safely converted to a single-channel "
                    "segmentation mask. Please convert labelsTr to single-channel "
                    "label maps and rerun nnUNet preprocessing."
                )

            target = channel_0

        elif target.shape[1] != 1:
            raise RuntimeError(
                "Expected a single-channel label or an RGB label with three "
                f"identical channels, but got target shape {target.shape}."
            )

        # 同时兼容标签值{0, 1}和{0, 255}。
        mask = (target > 0).float()

        edge_gt = self.mask_to_edge_2d(mask)

        return edge_gt

    @staticmethod
    def _get_target_for_check(target):
        if isinstance(target, (list, tuple)):
            return target[0]
        return target

    def forward(self, seg_output, target):
        """
        nnUNet 原始 train_step / validation_step 会调用:

            loss = self.loss(seg_output, target)

        这里的 seg_output 已经是 wrapper 返回的单一分割输出。
        """

        if self.check_nan:
            if isinstance(seg_output, (list, tuple)):
                for i, out in enumerate(seg_output):
                    if not torch.isfinite(out).all():
                        raise RuntimeError(
                            f"seg_output[{i}] contains NaN or Inf."
                        )
            else:
                if not torch.isfinite(seg_output).all():
                    raise RuntimeError(
                        "seg_output contains NaN or Inf."
                    )

            target_check = self._get_target_for_check(target)
            if not torch.isfinite(target_check.float()).all():
                raise RuntimeError(
                    "target contains NaN or Inf."
                )

        # --------------------------------------------------
        # 1. nnUNet 原始分割 loss
        # --------------------------------------------------
        seg_loss = self.base_loss(seg_output, target)

        if self.check_nan:
            if not torch.isfinite(seg_loss):
                raise RuntimeError(
                    f"seg_loss is not finite: {seg_loss.item()}. "
                    "The NaN comes from nnUNet base segmentation loss, "
                    "not from edge loss."
                )

        self.last_seg_loss = seg_loss.detach()

        # --------------------------------------------------
        # 2. validation 阶段只返回 seg_loss
        # --------------------------------------------------
        if not self.is_training_fn():
            self.last_edge_loss = None
            self.last_total_loss = seg_loss.detach()
            return seg_loss

        # --------------------------------------------------
        # 3. training 阶段才计算 edge loss
        # --------------------------------------------------
        edge_logits = self.get_edge_logits_fn()

        if edge_logits is None:
            if self.require_edge_in_training:
                raise RuntimeError(
                    "edge_logits is None during training. "
                    "Your original model should return "
                    "(seg_output, edge_logits), and SegEdgeWrapper should "
                    "store edge_logits as last_edge_logits."
                )

            self.last_edge_loss = None
            self.last_total_loss = seg_loss.detach()
            return seg_loss

        if edge_logits.ndim != 4:
            raise RuntimeError(
                "Expected edge_logits shape [B, 1, H, W], "
                f"but got {edge_logits.shape}."
            )

        if edge_logits.shape[1] != 1:
            raise RuntimeError(
                "Expected edge_logits to have 1 channel, "
                f"but got shape {edge_logits.shape}."
            )

        if self.check_nan:
            if not torch.isfinite(edge_logits).all():
                raise RuntimeError(
                    "edge_logits contains NaN or Inf during training. "
                    "Please check the edge branch of your model."
                )

        edge_gt = self.mask_to_edge(target)

        if edge_logits.shape[2:] != edge_gt.shape[2:]:
            edge_logits = F.interpolate(
                edge_logits,
                size=edge_gt.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        edge_gt = edge_gt.to(
            device=edge_logits.device,
            dtype=edge_logits.dtype,
        )

        edge_loss = self.edge_criterion(edge_logits, edge_gt)

        if self.check_nan:
            if not torch.isfinite(edge_loss):
                raise RuntimeError(
                    f"edge_loss is not finite: {edge_loss.item()}. "
                    "The NaN comes from edge loss."
                )

        total_loss = seg_loss + self.lambda_edge * edge_loss

        if self.check_nan:
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "total_loss is not finite. "
                    f"seg_loss={seg_loss.item()}, "
                    f"edge_loss={edge_loss.item()}, "
                    f"lambda_edge={self.lambda_edge}."
                )

        self.last_edge_loss = edge_loss.detach()
        self.last_total_loss = total_loss.detach()

        return total_loss


class nnUNetTrainer_CustomModel(nnUNetTrainer):
    crop_center = (112, 128)  # (y, x)
    crop_size = (224, 224)    # (height, width)

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device=device,
        )

        # 自定义 ViT 模型不使用 nnUNet deep supervision
        self.enable_deep_supervision = False

        # 训练 epoch
        self.num_epochs = 150

        # 每隔多少 epoch 保存一次
        self.save_every = 10

        # 建议先用 0.01，稳定后可以尝试 0.05 或 0.1
        self.lambda_edge = 0.01

        # 如果你的自定义 ViT 对默认 nnUNet 学习率不稳定，可以打开这一行
        self.initial_lr = 1e-2

    @classmethod
    def _center_crop_tensor(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Crop a 2D nnUNet tensor around ``crop_center`` to ``crop_size``."""
        if tensor.ndim < 2:
            raise RuntimeError(
                f"Expected a tensor with spatial dimensions, got {tensor.shape}."
            )

        center_y, center_x = cls.crop_center
        crop_h, crop_w = cls.crop_size
        height, width = tensor.shape[-2:]

        # nnUNet preprocessing/plans may already provide the requested patch.
        # In that case no further crop is necessary.
        if (height, width) == (crop_h, crop_w):
            return tensor

        top = center_y - crop_h // 2
        left = center_x - crop_w // 2
        bottom = top + crop_h
        right = left + crop_w

        if top < 0 or left < 0 or bottom > height or right > width:
            raise RuntimeError(
                "The requested center crop is outside the input image: "
                f"input={(height, width)}, center={cls.crop_center}, "
                f"crop_size={cls.crop_size}, bounds={(top, bottom, left, right)}."
            )

        return tensor[..., top:bottom, left:right]

    @classmethod
    def _crop_target(cls, target):
        if isinstance(target, list):
            return [cls._center_crop_tensor(item) for item in target]
        if isinstance(target, tuple):
            return tuple(cls._center_crop_tensor(item) for item in target)
        return cls._center_crop_tensor(target)

    @classmethod
    def _crop_batch(cls, batch: dict) -> dict:
        """Return a shallow batch copy with data and labels cropped identically."""
        cropped_batch = dict(batch)
        cropped_batch["data"] = cls._center_crop_tensor(batch["data"])
        cropped_batch["target"] = cls._crop_target(batch["target"])
        return cropped_batch

    def train_step(self, batch: dict) -> dict:
        return super().train_step(self._crop_batch(batch))

    def validation_step(self, batch: dict) -> dict:
        return super().validation_step(self._crop_batch(batch))

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        nnUNet 默认网络通常有:
            network.decoder.deep_supervision

        但自定义 ViT 模型没有 decoder.deep_supervision。
        因此这里直接禁用 deep supervision，避免访问不存在的 decoder。
        """

        self.enable_deep_supervision = False
        return

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        print("=" * 80)
        print("Using custom ViT segmentation model with edge attribute loss")
        print(f"num_input_channels: {num_input_channels}")
        print(f"num_output_channels: {num_output_channels}")
        print(f"enable_deep_supervision from nnUNet: {enable_deep_supervision}")
        print("Force disable deep supervision for custom model")
        print("=" * 80)

        # 你的数据当前设计为 3 通道输入
        if num_input_channels != 3:
            raise RuntimeError(
                f"Expected 3 input channels, but got {num_input_channels}. "
                "Please check dataset.json, imagesTr/imagesTs and preprocessing."
            )

        # nnUNet 二分类 CE + Dice 通常使用 2 通道输出:
        # channel 0: background
        # channel 1: foreground
        if num_output_channels != 2:
            raise RuntimeError(
                f"Expected 2 output channels, but got {num_output_channels}. "
                "For binary segmentation with nnUNet CE + Dice loss, "
                "the network should usually output 2 channels."
            )

        config_vit = CONFIGS_ViT_seg["ViT-B_16"]
        config_vit.n_classes = num_output_channels
        config_vit.n_skip = 3

        base_model = ViT_seg(
            config_vit,
            img_size=224,
            num_classes=num_output_channels,
        )

        pretrained_path = Path(config_vit.pretrained_path)
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"ViT pretrained weights not found: {pretrained_path}"
            )
        print(f"Loading ViT pretrained weights from: {pretrained_path}")
        base_model.load_from(weights=np.load(pretrained_path))

        model = SegEdgeWrapper(base_model)

        return model

    def _unwrap_network(self):
        """
        获取真正的 SegEdgeWrapper。

        兼容几种情况:
            1. 普通 self.network
            2. DDP: self.network.module
            3. torch.compile: self.network._orig_mod
            4. DDP + torch.compile 的组合
        """

        net = self.network

        changed = True
        while changed:
            changed = False

            if hasattr(net, "module"):
                net = net.module
                changed = True

            if hasattr(net, "_orig_mod"):
                net = net._orig_mod
                changed = True

        return net

    def _get_edge_logits_from_network(self):
        """
        从 SegEdgeWrapper 中读取最近一次 forward 缓存的 edge_logits。
        """

        net = self._unwrap_network()

        if hasattr(net, "get_last_edge_logits"):
            return net.get_last_edge_logits()

        if hasattr(net, "last_edge_logits"):
            return net.last_edge_logits

        return None

    def _is_network_training(self):
        """
        判断当前是否处于 training 阶段。

        nnUNet 在 validation 时会调用:
            self.network.eval()

        因此这里可以用 self.network.training 区分 train / val。
        """

        return self.network.training

    def _build_loss(self):
        """
        不重写 train_step。

        先构建 nnUNet 原始 loss:
            base_loss = super()._build_loss()

        再包装为:
            train: seg_loss + lambda_edge * edge_loss
            val:   seg_loss
        """

        base_loss = super()._build_loss()

        loss = SegWithEdgeLoss(
            base_loss=base_loss,
            get_edge_logits_fn=self._get_edge_logits_from_network,
            is_training_fn=self._is_network_training,
            lambda_edge=self.lambda_edge,
            require_edge_in_training=True,
            check_nan=True,
        )

        return loss
