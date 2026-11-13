import argparse
import random
import os

import torch
from torch import optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from Dataset_slice_2d import SegDataset
from torch.utils.data import DataLoader


# =========================================================
# device
# =========================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================

# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--root_path",
    type=str,
    default="./data/re_kfold",
    help="root dir for data",
)

parser.add_argument(
    "--fold",
    type=str,
    default=2,
)

parser.add_argument(
    "--list_dir",
    type=str,
    default="./lists/lists_Synapse",
    help="list dir",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=1,
    help="output channel of network",
)

parser.add_argument(
    "--max_iterations",
    type=int,
    default=30000,
    help="maximum iteration number to train",
)

parser.add_argument(
    "--max_epochs",
    type=int,
    default=150,
    help="maximum epoch number to train",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="batch_size per gpu",
)

parser.add_argument(
    "--n_gpu",
    type=int,
    default=1,
    help="total gpu",
)

parser.add_argument(
    "--deterministic",
    type=int,
    default=0,
    help="whether use deterministic training",
)

parser.add_argument(
    "--base_lr",
    type=float,
    default=0.0001,
    help="segmentation network learning rate",
)

parser.add_argument(
    "--img_size",
    type=int,
    default=224,
    help="input patch size of network input",
)

parser.add_argument(
    "--seed",
    type=int,
    default=1234,
    help="random seed",
)

parser.add_argument(
    "--n_skip",
    type=int,
    default=3,
    help="using number of skip-connect",
)

parser.add_argument(
    "--vit_name",
    type=str,
    default="ViT-B_16",
    help="select one vit model",
)

parser.add_argument(
    "--vit_patches_size",
    type=int,
    default=16,
    help="vit_patches_size",
)

parser.add_argument(
    "--out_path",
    type=str,
    default="./output/2d_all",
)

args = parser.parse_args()


# =========================================================
# 2D Dice Loss


# =========================================================
class SafeDiceLoss2D(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        """
        probs:   [B, 1, H, W], probabilities after sigmoid
        targets: [B, 1, H, W], binary 0/1 labels

        Compute Dice loss only for slices with foreground targets.
        Return 0 when all target slices in the batch are empty.
        """
        B = probs.size(0)

        probs_flat = probs.contiguous().view(B, -1)
        targets_flat = targets.contiguous().view(B, -1)

        target_sum = targets_flat.sum(dim=1)

        positive_mask = target_sum > 0

        if positive_mask.sum() == 0:
            return probs.sum() * 0.0

        probs_pos = probs_flat[positive_mask]
        targets_pos = targets_flat[positive_mask]

        intersection = (probs_pos * targets_pos).sum(dim=1)
        union = probs_pos.sum(dim=1) + targets_pos.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1.0 - dice.mean()

        return loss


# =========================================================


# =========================================================
def dice_2d_per_slice_from_pred(pred, target, smooth=1e-6, ignore_empty=True):
    """
    pred:   [B, 1, H, W], binary 0/1 predictions
    target: [B, 1, H, W], binary 0/1 labels

    Returns:
        dice_list: list[float]

    Notes:
        - Compute Dice separately for each 2D slice.
        - Do not combine slices into a 3D volume.
        - Exclude empty ground-truth slices when ignore_empty=True.
    """
    B = pred.size(0)

    pred_flat = pred.contiguous().view(B, -1)
    target_flat = target.contiguous().view(B, -1)

    dice_list = []

    for i in range(B):
        pred_i = pred_flat[i]
        target_i = target_flat[i]

        target_sum = target_i.sum()
        pred_sum = pred_i.sum()

        if ignore_empty and target_sum == 0:
            continue

        intersection = (pred_i * target_i).sum()
        union = pred_sum + target_sum

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_list.append(float(dice.item()))

    return dice_list


# =========================================================

# =========================================================
def train_model(
    model,
    optimizer,
    dataloader,
    dataloader_val,
    device,
    args,
    num_epochs=150,
):
    """
    Training:
        batch["image"]: [B, 3, H, W]
        batch["mask"]:  [B, 1, H, W]
        batch["edge"]:  [B, 1, H, W]

    Validation:
        Input formats:

        1. Slice-level validation:
            batch["image"]: [B, 3, H, W]
            batch["mask"]:  [B, 1, H, W]

        2. Case-level validation:
            batch["image"]: [1, D, 3, H, W]
            batch["mask"]:  [1, D, 1, H, W]

        Dice is defined per 2D slice rather than per 3D volume.
    """

    os.makedirs(args.out_path, exist_ok=True)

    dice_total = []
    max_2d_dice = 0.0

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss_2d = SafeDiceLoss2D()

    for epoch in range(num_epochs):
        # =================================================
        
        # =================================================
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader, 1):
            inputs = batch["image"].to(device)          # [B, 3, H, W]
            label = batch["mask"].float().to(device)    # [B, 1, H, W]
            edge = batch["edge"].float().to(device)     # [B, 1, H, W]

            optimizer.zero_grad()

            outputs = model(inputs)

            
            # 1. outputs, edge_outputs = model(inputs)
            # 2. outputs = model(inputs)
            if isinstance(outputs, (list, tuple)):
                pred_mask_logits = outputs[0]
                pred_edge_logits = outputs[1] if len(outputs) > 1 else None
            else:
                pred_mask_logits = outputs
                pred_edge_logits = None

            
            loss_bce = bce_loss(pred_mask_logits, label)

            pred_mask_prob = torch.sigmoid(pred_mask_logits)
            loss_dice = dice_loss_2d(pred_mask_prob, label)

            loss_area = 0.5 * loss_bce + 0.5 * loss_dice

            
            if pred_edge_logits is not None:
                loss_edge = bce_loss(pred_edge_logits, edge)
                loss = loss_area + 0.75 * loss_edge
            else:
                loss = loss_area

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mean_train_loss = epoch_loss / step
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {mean_train_loss:.4f}")

        # =================================================
        
        # =================================================
        decay = 0.5

        if (epoch % 25 == 0) and (epoch != 0):
            args.base_lr = args.base_lr * decay

            if args.base_lr <= 1e-8:
                args.base_lr = 1e-8

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.base_lr

            print(f"Learning rate decayed to: {args.base_lr}")

        # =================================================
        
        # =================================================
        if (epoch % 10 == 0) or (epoch == num_epochs - 1):
            model.eval()

            slice_dice_list = []
            slice_dice_loss_list = []

            with torch.no_grad():
                for val_step, batch in enumerate(dataloader_val, 1):
                    images = batch["image"].to(device)
                    masks = batch["mask"].float().to(device)



                    # -------------------------------------
                    
                    # images: [B, C, H, W]
                    # masks:  [B, 1, H, W]
                    # -------------------------------------

                    images_2d = images
                    masks_2d = masks
                    outputs = model(images_2d)

                    if isinstance(outputs, (list, tuple)):
                        pred_logits_2d = outputs[0]
                    else:
                        pred_logits_2d = outputs

                    pred_prob_2d = torch.sigmoid(pred_logits_2d)
                    pred_bin_2d = (pred_prob_2d >= 0.5).float()

                    
                    batch_slice_dice = dice_2d_per_slice_from_pred(
                        pred=pred_bin_2d,
                        target=masks_2d,
                        smooth=1e-6,
                        ignore_empty=True,
                    )

                    if len(batch_slice_dice) > 0:
                        slice_dice_list.extend(batch_slice_dice)
                        slice_dice_loss_list.extend(
                            [1.0 - d for d in batch_slice_dice]
                        )

            if len(slice_dice_list) == 0:
                mean_2d_dice = 0.0
                mean_2d_dice_loss = 0.0
            else:
                mean_2d_dice = float(np.mean(slice_dice_list))
                mean_2d_dice_loss = float(np.mean(slice_dice_loss_list))

            dice_total.append(round(mean_2d_dice, 4))

            print("=" * 70)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Validation")
            print(f"Mean 2D Dice: {mean_2d_dice:.4f}")
            print(f"Mean 2D Dice Loss: {mean_2d_dice_loss:.4f}")
            print(f"Valid positive 2D slices: {len(slice_dice_list)}")
            print(f"Best 2D Dice: {max_2d_dice:.4f}")
            print("=" * 70)

            
            if mean_2d_dice >= max_2d_dice:
                max_2d_dice = mean_2d_dice

                save_path = os.path.join(
                    args.out_path,
                    f"best_model_2d_all_dice_k{args.fold}.pth",
                )

                torch.save(model.state_dict(), save_path)

                print(f"Saved best model: {save_path}")
                print(f"Best Mean 2D Dice: {max_2d_dice:.4f}")

    # =====================================================
    
    # =====================================================
    save_txt_path = os.path.join(args.out_path, "val_2d_dice.txt")

    with open(save_txt_path, "a", encoding="utf-8") as file:
        file.write("val_2d_dice: " + str(dice_total) + "\n")
        file.write("best_2d_dice: " + str(round(max_2d_dice, 4)) + "\n")

    return model


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.out_path, exist_ok=True)

    # =====================================================
    
    # =====================================================
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    print(config_vit.pretrained_path)

    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )

    net = ViT_seg(
        config_vit,
        img_size=args.img_size,
        num_classes=config_vit.n_classes,
    ).to(device)

    net.load_from(weights=np.load(config_vit.pretrained_path))

    optimizer = optim.Adam(net.parameters(), lr=args.base_lr)

    # =====================================================
    
    # =====================================================
    liver_dataset = SegDataset(
        root=args.root_path,
        k_fold=args.fold,
        split="train",
        crop_center=(112, 128),
        #label_p_path=r"./data/re_kfold/split_data/split_train.png",
        augment=True,
        normalize=True,
        image_mean=0.5,
        image_std=0.5,
        augment_label_p=True,
    )

    dataloader = DataLoader(
        liver_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # =====================================================
    
    
    
    
    
    # =====================================================
    liver_dataset_val = SegDataset(
        root=args.root_path,
        k_fold=args.fold,
        split="val",
        image_size=(224, 224),
        crop_center=(112, 128),
        normalize=True,
        image_mean=0.5,
        image_std=0.5,
    )

    dataloader_val = DataLoader(
        liver_dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    train_model(
        model=net,
        optimizer=optimizer,
        dataloader=dataloader,
        dataloader_val=dataloader_val,
        device=device,
        args=args,
        num_epochs=args.max_epochs,
    )
