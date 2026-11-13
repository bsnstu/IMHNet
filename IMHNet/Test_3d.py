# -*- coding: utf-8 -*-
"""
Test script: output final mean metrics only.

Purpose:
    Use the trained best_model_3d_dice.pth to test on the test set.
    Calculate case-level 3D Dice and HD.
    Save only final mean/std values in chart-friendly CSV/JSON format.

Output:
    1. test_summary_for_plot.csv
       metric,mean,std,num_cases,num_inf_hd_cases
       Dice,0.8123,0.1021,20,0
       HD,12.3456,3.2109,20,0

    2. test_summary_for_plot.json
       {
           "num_cases": 20,
           "num_inf_hd_cases": 0,
           "metrics": [
               {"metric": "Dice", "mean": 0.8123, "std": 0.1021},
               {"metric": "HD", "mean": 12.3456, "std": 3.2109}
           ]
       }

    3. test_summary.txt
       Human-readable result.
"""

import os
import csv
import json
import argparse
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from Dataset_split_normalize import SegDataset_val


try:
    from scipy import ndimage
except ImportError:
    ndimage = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()

parser.add_argument("--root_path", type=str,
                    default=r"E:\PyhonProject_gpu\data\k_data\new_data\data\re_kfold",
                    help="root dir for data")
parser.add_argument('--fold', type=str,
                    default=1)

parser.add_argument("--checkpoint", type=str,
                    default=None,
                    help="trained best model checkpoint")

parser.add_argument("--out_path", type=str,
                    default=None,
                    help="save dir")

parser.add_argument("--num_classes", type=int, default=1)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--vit_name", type=str, default="ViT-B_16")
parser.add_argument("--n_skip", type=int, default=3)
parser.add_argument("--vit_patches_size", type=int, default=16)

parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--deterministic", type=int, default=0)

parser.add_argument("--crop_center_x", type=int, default=112)
parser.add_argument("--crop_center_y", type=int, default=128)

parser.add_argument("--label_p_path", type=str,
                    default=None,
                    help="path to split_train.png. If None, use root_path/split_data/split_train.png")

parser.add_argument("--image_mean", type=float, default=0.5)
parser.add_argument("--image_std", type=float, default=0.5)

# HD spacing. If you do not know the real physical spacing, keep all as 1.0.
parser.add_argument("--spacing_z", type=float, default=1.0)
parser.add_argument("--spacing_y", type=float, default=1.0)
parser.add_argument("--spacing_x", type=float, default=1.0)

args = parser.parse_args()
if args.checkpoint is None:
    args.checkpoint = f"./output/2d_all/best_model_2d_all_dice_k{args.fold}.pth"
if args.out_path is None:
    args.out_path = f"./output/3d_dice/2d_check_3d_result/fold_{args.fold}_re/"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_3d_from_pred(pred, target, smooth=1e-6):
    """
    pred:   [D, 1, H, W], binary 0/1 tensor
    target: [D, 1, H, W], binary 0/1 tensor
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    if union.item() == 0:
        return torch.tensor(1.0, device=pred.device)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def sen_3d_from_pred(pred, target, smooth=1e-6):
    """
    pred:   [D, 1, H, W], binary 0/1 tensor
    target: [D, 1, H, W], binary 0/1 tensor

    3D SEN (Sensitivity / Recall):
        TP / (TP + FN)

    Empty-case handling:
        1) target empty, pred empty:
           -> all-negative 3D case correctly predicted
           -> SEN = 1.0

        2) target empty, pred non-empty:
           -> all-negative 3D case with false-positive prediction
           -> SEN = 0.0

        3) target non-empty:
           -> SEN = TP / (TP + FN)
    """

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    target_sum = target.sum()
    pred_sum = pred.sum()

    # =========================
    # All-negative 3D GT case
    # =========================
    if target_sum.item() == 0:

        # GT all negative, prediction also all negative
        if pred_sum.item() == 0:
            return torch.tensor(
                1.0,
                dtype=torch.float32,
                device=pred.device
            )

        # GT all negative, but prediction contains positives
        else:
            return torch.tensor(
                0.0,
                dtype=torch.float32,
                device=pred.device
            )

    # =========================
    # Positive 3D GT case
    # =========================
    tp = (pred * target).sum()

    sen = (tp + smooth) / (target_sum + smooth)

    return sen


def hd_3d_from_numpy(pred_np, target_np, spacing=(1.0, 1.0, 1.0)):
    """
    pred_np:   [D, H, W], binary 0/1
    target_np: [D, H, W], binary 0/1

    Return:
        3D Hausdorff Distance.

    Empty handling:
        pred empty and target empty -> 0
        only one empty -> inf
    """
    if ndimage is None:
        raise ImportError("scipy is required for HD calculation. Please install scipy.")

    pred_np = pred_np.astype(bool)
    target_np = target_np.astype(bool)

    pred_sum = pred_np.sum()
    target_sum = target_np.sum()

    if pred_sum == 0 and target_sum == 0:
        return 0.0

    if pred_sum == 0 or target_sum == 0:
        return 100

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

    pred_eroded = ndimage.binary_erosion(pred_np, structure=structure, border_value=0)
    target_eroded = ndimage.binary_erosion(target_np, structure=structure, border_value=0)

    pred_surface = np.logical_xor(pred_np, pred_eroded)
    target_surface = np.logical_xor(target_np, target_eroded)

    if pred_surface.sum() == 0:
        pred_surface = pred_np

    if target_surface.sum() == 0:
        target_surface = target_np

    dt_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    dist_pred_to_target = dt_target[pred_surface]
    dist_target_to_pred = dt_pred[target_surface]

    hd = max(dist_pred_to_target.max(), dist_target_to_pred.max())
    return float(hd)


def assd_3d_from_numpy(pred_np, target_np, spacing=(1.0, 1.0, 1.0)):
    """
    pred_np:   [D, H, W], binary 0/1
    target_np: [D, H, W], binary 0/1

    Return:
        3D Average Symmetric Surface Distance (ASSD).

    Empty handling:
        pred empty and target empty -> 0
        only one empty -> 100
    """
    if ndimage is None:
        raise ImportError("scipy is required for ASSD calculation. Please install scipy.")

    pred_np = pred_np.astype(bool)
    target_np = target_np.astype(bool)

    pred_sum = pred_np.sum()
    target_sum = target_np.sum()

    if pred_sum == 0 and target_sum == 0:
        return 0.0

    if pred_sum == 0 or target_sum == 0:
        return 100.0

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

    pred_eroded = ndimage.binary_erosion(pred_np, structure=structure, border_value=0)
    target_eroded = ndimage.binary_erosion(target_np, structure=structure, border_value=0)

    pred_surface = np.logical_xor(pred_np, pred_eroded)
    target_surface = np.logical_xor(target_np, target_eroded)

    if pred_surface.sum() == 0:
        pred_surface = pred_np

    if target_surface.sum() == 0:
        target_surface = target_np

    dt_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    dist_pred_to_target = dt_target[pred_surface]
    dist_target_to_pred = dt_pred[target_surface]

    assd = (dist_pred_to_target.mean() + dist_target_to_pred.mean()) / 2.0
    return float(assd)


def build_model(args):
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )

    model = ViT_seg(
        config_vit,
        img_size=args.img_size,
        num_classes=config_vit.n_classes,
    ).to(device)

    return model


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint:", checkpoint_path)

    return model


def finite_mean_std(values):
    finite_values = [float(v) for v in values if np.isfinite(v)]

    if len(finite_values) == 0:
        return None, None

    arr = np.array(finite_values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def save_summary_for_plot(dice_list, hd_list, sen_list, assd_list, args):
    os.makedirs(args.out_path, exist_ok=True)

    csv_path = os.path.join(args.out_path, "test_summary_for_plot.csv")
    json_path = os.path.join(args.out_path, "test_summary_for_plot.json")
    txt_path = os.path.join(args.out_path, "test_summary.txt")

    mean_dice, std_dice = finite_mean_std(dice_list)
    mean_hd, std_hd = finite_mean_std(hd_list)
    mean_sen, std_sen = finite_mean_std(sen_list)
    mean_assd, std_assd = finite_mean_std(assd_list)

    num_cases = len(dice_list)
    num_inf_hd_cases = int(np.sum([not np.isfinite(x) for x in hd_list]))
    num_inf_assd_cases = int(np.sum([not np.isfinite(x) for x in assd_list]))

    # =========================
    # 1. CSV for direct plotting
    # =========================
    rows = [
        {
            "metric": "Dice",
            "mean": mean_dice,
            "std": std_dice,
            "num_cases": num_cases,
            "num_inf_hd_cases": num_inf_hd_cases,
        },
        {
            "metric": "HD",
            "mean": mean_hd,
            "std": std_hd,
            "num_cases": num_cases,
            "num_inf_hd_cases": num_inf_hd_cases,
        },
        {
            "metric": "SEN",
            "mean": mean_sen,
            "std": std_sen,
            "num_cases": num_cases,
            "num_inf_hd_cases": num_inf_hd_cases,
        },
        {
            "metric": "ASSD",
            "mean": mean_assd,
            "std": std_assd,
            "num_cases": num_cases,
            "num_inf_hd_cases": num_inf_hd_cases,
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "mean", "std", "num_cases", "num_inf_hd_cases"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # =========================
    # 2. JSON for LLM / plotting tools
    # =========================
    summary = {
        "num_cases": num_cases,
        "num_inf_hd_cases": num_inf_hd_cases,
        "num_inf_assd_cases": num_inf_assd_cases,
        "metrics": [
            {
                "metric": "Dice",
                "mean": mean_dice,
                "std": std_dice,
            },
            {
                "metric": "HD",
                "mean": mean_hd,
                "std": std_hd,
            },
            {
                "metric": "SEN",
                "mean": mean_sen,
                "std": std_sen,
            },
            {
                "metric": "ASSD",
                "mean": mean_assd,
                "std": std_assd,
            },
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # =========================
    # 3. Readable txt
    # =========================
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Test Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Number of cases: {num_cases}\n")
        f.write(f"Mean Dice: {mean_dice:.6f}\n")
        f.write(f"Std  Dice: {std_dice:.6f}\n")
        f.write(f"Mean SEN : {mean_sen:.6f}\n")
        f.write(f"Std  SEN : {std_sen:.6f}\n")

        if mean_hd is None:
            f.write("Mean HD: None\n")
            f.write("Std  HD: None\n")
        else:
            f.write(f"Mean HD: {mean_hd:.6f}\n")
            f.write(f"Std  HD: {std_hd:.6f}\n")

        if mean_assd is None:
            f.write("Mean ASSD: None\n")
            f.write("Std  ASSD: None\n")
        else:
            f.write(f"Mean ASSD: {mean_assd:.6f}\n")
            f.write(f"Std  ASSD: {std_assd:.6f}\n")

        f.write(f"Number of inf HD cases: {num_inf_hd_cases}\n")
        f.write(f"Number of inf ASSD cases: {num_inf_assd_cases}\n")
        f.write("=" * 60 + "\n")

    print("=" * 70)
    print("Test finished.")
    print(f"Number of cases: {num_cases}")
    print(f"Mean Dice: {mean_dice:.6f}")
    print(f"Std  Dice: {std_dice:.6f}")
    print(f"Mean SEN : {mean_sen:.6f}")
    print(f"Std  SEN : {std_sen:.6f}")

    if mean_hd is None:
        print("Mean HD: None")
        print("Std  HD: None")
    else:
        print(f"Mean HD: {mean_hd:.6f}")
        print(f"Std  HD: {std_hd:.6f}")

    if mean_assd is None:
        print("Mean ASSD: None")
        print("Std  ASSD: None")
    else:
        print(f"Mean ASSD: {mean_assd:.6f}")
        print(f"Std  ASSD: {std_assd:.6f}")

    print(f"Number of inf HD cases: {num_inf_hd_cases}")
    print(f"Number of inf ASSD cases: {num_inf_assd_cases}")
    print("Saved CSV :", csv_path)
    #print("Saved JSON:", json_path)
    #print("Saved TXT :", txt_path)
    print("=" * 70)


@torch.no_grad()
def test_model(model, dataloader_test, args):
    model.eval()

    dice_list = []
    hd_list = []
    sen_list = []
    assd_list = []

    spacing = (args.spacing_z, args.spacing_y, args.spacing_x)

    for test_step, batch in enumerate(dataloader_test, 1):
        images = batch["image"].to(device)          # [1, D, 3, H, W]
        masks = batch["mask"].float().to(device)    # [1, D, 1, H, W]

        case_id = batch["case_id"][0]

        B, D, C, H, W = images.shape

        images_2d = images.view(B * D, C, H, W)     # [D, 3, H, W]

        outputs = model(images_2d)

        if isinstance(outputs, (list, tuple)):
            pred_logits_2d = outputs[0]
        else:
            pred_logits_2d = outputs

        pred_prob_2d = torch.sigmoid(pred_logits_2d)
        pred_bin_2d = (pred_prob_2d >= 0.5).float()

        pred_bin_3d = pred_bin_2d.view(B, D, 1, H, W)

        case_pred = pred_bin_3d[0]     # [D, 1, H, W]
        case_mask = masks[0]           # [D, 1, H, W]

        case_dice = dice_3d_from_pred(case_pred, case_mask)
        case_sen = sen_3d_from_pred(case_pred, case_mask)

        pred_np = case_pred[:, 0].detach().cpu().numpy().astype(np.uint8)  # [D, H, W]
        mask_np = case_mask[:, 0].detach().cpu().numpy().astype(np.uint8)  # [D, H, W]

        case_hd = hd_3d_from_numpy(pred_np, mask_np, spacing=spacing)
        case_assd = assd_3d_from_numpy(pred_np, mask_np, spacing=spacing)

        dice_value = float(case_dice.item())
        hd_value = float(case_hd)
        sen_value = float(case_sen.item())
        assd_value = float(case_assd)

        dice_list.append(dice_value)
        hd_list.append(hd_value)
        sen_list.append(sen_value)
        assd_list.append(assd_value)

        print(
            f"Case [{test_step}/{len(dataloader_test)}] "
            f"{case_id} | "
            f"Dice: {dice_value:.4f} | "
            f"HD: {hd_value:.4f} | "
            f"SEN: {sen_value:.4f} | "
            f"ASSD: {assd_value:.4f}"
        )

    save_summary_for_plot(dice_list, hd_list, sen_list, assd_list, args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    set_seed(args.seed)

    os.makedirs(args.out_path, exist_ok=True)

    if args.label_p_path is None:
        args.label_p_path = os.path.join(args.root_path, "split_data", "split_train.png")

    print("Using device:", device)
    print("Root path:", args.root_path)
    print("Checkpoint:", args.checkpoint)
    print("Output path:", args.out_path)
    print("Label_p path:", args.label_p_path)

    test_dataset = SegDataset_val(
        root=args.root_path,
        k_fold=args.fold,
        split="test",
        image_size=(args.img_size, args.img_size),
        crop_center=(args.crop_center_x, args.crop_center_y),
        label_p_path=args.label_p_path,
        normalize=True,
        image_mean=args.image_mean,
        image_std=args.image_std,
    )

    dataloader_test = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    net = build_model(args)
    net = load_checkpoint(net, args.checkpoint)
    net.eval()

    test_model(net, dataloader_test, args)
