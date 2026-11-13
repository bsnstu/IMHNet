# -*- coding: utf-8 -*-
"""
nnUNetv2 trainer variant with:

1. A maximum of 150 training epochs.
2. Standard nnU-Net online validation every 10 epochs.
3. No changes to the network, loss, optimizer, data augmentation,
   train/validation split, or number of iterations per epoch.

Validation is performed after epochs 10, 20, ..., 150. The resulting
foreground Dice is logged by nnU-Net as ``mean_fg_dice`` (Pseudo Dice).
"""

from time import time
from typing import List

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_val10(nnUNetTrainer):
    """Train for 150 epochs and validate once every 10 epochs."""

    validation_interval = 10

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        # Store this experiment in the user-requested temporary result folder.
        self.output_folder_base = (
            r"D:\BSN\nnunetv2\nnUNet_results\Dataset001_IMH\temp"
        )
        self.output_folder = join(self.output_folder_base, f"fold_{fold}")

        # This is set before initialize(), so nnU-Net's PolyLR scheduler
        # is also configured for exactly 150 epochs.
        self.num_epochs = 150

        # Used only inside this trainer to tell on_epoch_end whether the
        # standard nnU-Net validation loop ran in the current epoch.
        self._validated_this_epoch = False

    def _should_validate(self) -> bool:
        """Return True after epochs 10, 20, ..., 150."""
        return (self.current_epoch + 1) % self.validation_interval == 0

    def run_training(self):
        """
        Use nnU-Net's original training and validation steps, but execute
        the validation block only once every 10 epochs.
        """
        self.on_train_start()

        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()

            train_outputs: List[dict] = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(
                    self.train_step(next(self.dataloader_train))
                )
            self.on_train_epoch_end(train_outputs)

            self._validated_this_epoch = self._should_validate()

            if self._validated_this_epoch:
                self.print_to_log_file(
                    f"Running validation after epoch "
                    f"{self.current_epoch + 1}/{self.num_epochs}."
                )

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs: List[dict] = []

                    for _ in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(
                            self.validation_step(next(self.dataloader_val))
                        )

                    self.on_validation_epoch_end(val_outputs)
            else:
                self.print_to_log_file(
                    f"Skipping validation after epoch "
                    f"{self.current_epoch + 1}/{self.num_epochs}; "
                    f"validation interval is {self.validation_interval}."
                )

            self.on_epoch_end()

        self.on_train_end()

    def on_epoch_end(self):
        """
        Call nnU-Net's original epoch-end logic after validation epochs.

        During non-validation epochs, log training information and advance
        the epoch without reading missing validation/Dice entries.
        """
        if self._validated_this_epoch:
            super().on_epoch_end()
            return

        self.logger.log(
            "epoch_end_timestamps",
            time(),
            self.current_epoch,
        )

        train_loss = self.logger.get_value(
            "train_losses",
            step=-1,
        )

        epoch_time = (
            self.logger.get_value(
                "epoch_end_timestamps",
                step=-1,
            )
            - self.logger.get_value(
                "epoch_start_timestamps",
                step=-1,
            )
        )

        self.print_to_log_file(
            "train_loss",
            np.round(train_loss, decimals=4),
        )
        self.print_to_log_file(
            "validation",
            "skipped",
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(epoch_time, decimals=2)} s"
        )

        # Preserve nnU-Net's periodic latest-checkpoint behavior. With the
        # default save_every=50, epochs 50 and 100 are validation epochs and
        # are handled by super().on_epoch_end(). This branch also remains
        # correct if save_every is changed later.
        if (
            (self.current_epoch + 1) % self.save_every == 0
            and self.current_epoch != self.num_epochs - 1
        ):
            self.save_checkpoint(
                join(
                    self.output_folder,
                    "checkpoint_latest.pth",
                )
            )

        if self.local_rank == 0:
            # Some nnU-Net versions can plot sparse validation points; if an
            # older logger cannot, training must not fail because of plotting.
            try:
                self.logger.plot_progress_png(self.output_folder)
            except Exception as error:
                self.print_to_log_file(
                    "Progress plot skipped on a non-validation epoch:",
                    repr(error),
                )

        self.current_epoch += 1
