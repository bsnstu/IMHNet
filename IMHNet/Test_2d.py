# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
import random
import re
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from PIL import Image

from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.ndimage import binary_erosion

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from Dataset_slice_2d import SegDataset


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===========================
# Args
# ===========================

parser=argparse.ArgumentParser()

parser.add_argument("--root_path",type=str,default=r"./data/re_kfold")
parser.add_argument("--fold",type=int,default=2)
parser.add_argument("--checkpoint",type=str,default=None)
parser.add_argument("--out_path",type=str,default=None)

parser.add_argument("--num_classes",type=int,default=1)
parser.add_argument("--img_size",type=int,default=224)
parser.add_argument("--vit_name",type=str,default="ViT-B_16")
parser.add_argument("--n_skip",type=int,default=3)
parser.add_argument("--vit_patches_size",type=int,default=16)

parser.add_argument("--seed",type=int,default=1234)
parser.add_argument("--deterministic",type=int,default=0)

parser.add_argument("--threshold",type=float,default=0.5)
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--repeat_channels",type=int,default=3)
parser.add_argument("--hd_empty_value",type=float,default=100)
parser.add_argument(
    "--save_predictions",
    type=int,
    choices=[0,1],
    default=1,
    help="save test segmentation PNG files under the corresponding fold",
)

args=parser.parse_args()


if args.checkpoint is None:
    args.checkpoint=f"./output/2d_all/best_model_2d_all_dice_k{args.fold}.pth"

if args.out_path is None:
    args.out_path=f"./output/2d_all/fold_{args.fold}/"



# ===========================
# Seed
# ===========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



# ===========================
# Input
# ===========================

def prepare_model_input(images,repeat_channels=3):

    if images.dim()!=4:
        raise RuntimeError(f"Input shape error:{images.shape}")

    if repeat_channels==3 and images.size(1)==1:
        images=images.repeat(1,3,1,1)

    return images



# ===========================
# Dice
# ===========================

def dice_2d_single_slice(pred_i,target_i,smooth=1e-6):

    pred_sum=pred_i.sum()
    target_sum=target_i.sum()

    intersection=(pred_i*target_i).sum()
    union=pred_sum+target_sum

    if union.item()==0:
        return 1.0

    return float(
        ((2*intersection+smooth)/(union+smooth)).item()
    )



# ===========================
# SEN
# ===========================

def sen_2d_single_slice(pred_i, target_i):
    """
    Calculate sensitivity (SEN) for a single 2D slice.

    Rules:
    1. GT positive:
       SEN = TP / (TP + FN)

    2. GT empty and prediction empty:
       SEN = 1.0

    3. GT empty but prediction non-empty:
       SEN = 0.0
    """

    pred_sum = pred_i.sum()
    target_sum = target_i.sum()

    # Ground truth is empty
    if target_sum.item() == 0:
        # Prediction is also empty
        if pred_sum.item() == 0:
            return 1.0
        # False-positive prediction
        else:
            return 0.0

    # Ground truth contains positive pixels
    tp = (pred_i * target_i).sum()
    fn = ((1 - pred_i) * target_i).sum()

    return float((tp / (tp + fn)).item())



# ===========================
# ASSD
# ===========================

def assd_2d_single_slice(pred_i,target_i,H,W,empty_value=100):

    pred=pred_i.cpu().numpy().reshape(H,W).astype(bool)
    target=target_i.cpu().numpy().reshape(H,W).astype(bool)

    if pred.sum()==0 and target.sum()==0:
        return 0.0

    if pred.sum()==0 or target.sum()==0:
        return empty_value


    pred_edge=pred ^ binary_erosion(pred)
    target_edge=target ^ binary_erosion(target)

    p=np.argwhere(pred_edge)
    t=np.argwhere(target_edge)

    if len(p)==0 or len(t)==0:
        return empty_value


    d1=np.min(cdist(p,t),axis=1)
    d2=np.min(cdist(t,p),axis=1)

    return float((d1.mean()+d2.mean())/2)



# ===========================
# HD
# ===========================

def hd_2d_single_slice(pred_np,target_np,hd_empty_value=100):

    p=np.argwhere(pred_np>0)
    t=np.argwhere(target_np>0)

    if len(p)==0 and len(t)==0:
        return 0.0

    if len(p)==0 or len(t)==0:
        return float(hd_empty_value)

    return float(
        max(
            directed_hausdorff(p,t)[0],
            directed_hausdorff(t,p)[0]
        )
    )



# ===========================
# Compute metrics
# ===========================

def compute_2d_dice_hd_include_empty(
        pred,
        target,
        start_slice_index=0,
        smooth=1e-6,
        hd_empty_value=100):


    B=pred.size(0)
    H,W=pred.shape[-2:]


    pred_flat=pred.view(B,-1)
    target_flat=target.view(B,-1)


    pred_np=pred.cpu().numpy()
    target_np=target.cpu().numpy()


    dice_list=[]
    hd_list=[]
    sen_list=[]
    assd_list=[]

    slice_info_list=[]



    for i in range(B):

        pred_i=pred_flat[i]
        target_i=target_flat[i]


        dice=dice_2d_single_slice(
            pred_i,
            target_i,
            smooth
        )


        sen=sen_2d_single_slice(
            pred_i,
            target_i
        )


        assd=assd_2d_single_slice(
            pred_i,
            target_i,
            H,
            W
        )


        hd=hd_2d_single_slice(
            pred_np[i,0],
            target_np[i,0],
            hd_empty_value
        )


        dice_list.append(dice)
        hd_list.append(hd)
        sen_list.append(sen)
        assd_list.append(assd)


        slice_info_list.append(
            {
                "slice_index":int(start_slice_index+i),
                "dice":dice,
                "hd":hd,
                "sen":sen,
                "assd":assd
            }
        )


    return (
        dice_list,
        hd_list,
        sen_list,
        assd_list,
        slice_info_list
    )

# ================= Model =================

def build_model(args):
    config_vit=CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes=args.num_classes
    config_vit.n_skip=args.n_skip

    if "R50" in args.vit_name:
        config_vit.patches.grid=(
            args.img_size//args.vit_patches_size,
            args.img_size//args.vit_patches_size
        )

    return ViT_seg(
        config_vit,
        img_size=args.img_size,
        num_classes=config_vit.n_classes
    ).to(device)



def get_main_output(x):
    return x[0] if isinstance(x,(list,tuple)) else x



def load_checkpoint(model,path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ckpt=torch.load(
        path,
        map_location=device
    )

    state=(
        ckpt["state_dict"]
        if "state_dict" in ckpt
        else ckpt["model_state_dict"]
        if "model_state_dict" in ckpt
        else ckpt
    )

    model.load_state_dict(
        state,
        strict=False
    )

    print("Load:",path)

    return model



def finite_mean_std(x):

    x=[float(i) for i in x if np.isfinite(i)]

    if not x:
        return None,None

    x=np.array(x)

    return float(x.mean()),float(x.std())


def make_windows_safe_name(value,default_name):

    name=str(value).strip()
    name=re.sub(r'[<>:"/\\|?*]+',"_",name)
    name=name.rstrip(" .")

    return name if name else default_name


def get_batch_value(batch,key,index,default_value):

    if key not in batch:
        return default_value

    value=batch[key]

    if torch.is_tensor(value):
        if value.dim()==0:
            return value.item()
        value=value[index]
        return value.item() if value.numel()==1 else value

    if isinstance(value,(list,tuple)):
        return value[index]

    return value


def save_prediction_batch(pred,batch,start_index,args):
    """Save binary prediction slices into this fold's test_predictions folder."""

    prediction_root=os.path.join(
        args.out_path,
        "test_predictions"
    )
    os.makedirs(prediction_root,exist_ok=True)

    saved_paths=[]

    for i in range(pred.size(0)):

        global_index=start_index+i

        case_id=get_batch_value(
            batch,
            "case_id",
            i,
            ""
        )
        image_filename=get_batch_value(
            batch,
            "image_filename",
            i,
            ""
        )

        safe_case_id=make_windows_safe_name(
            case_id,
            "unknown_case"
        )

        original_name=os.path.basename(
            str(image_filename).strip()
        )

        if original_name:
            stem,_=os.path.splitext(original_name)
            safe_stem=make_windows_safe_name(
                stem,
                f"slice_{global_index:05d}"
            )
            if str(case_id).strip():
                output_name=f"{safe_case_id}_{safe_stem}.png"
            else:
                output_name=f"{safe_stem}.png"
        else:
            output_name=f"slice_{global_index:05d}.png"

        output_path=os.path.join(
            prediction_root,
            output_name
        )

        pred_np=pred[i,0].detach().cpu().numpy()
        pred_np=(pred_np>0).astype(np.uint8)*255

        Image.fromarray(pred_np,mode="L").save(output_path)
        saved_paths.append(output_path)

    return saved_paths



def save_summary_for_plot(
        dice,hd,sen,assd,detail,total,args):

    os.makedirs(args.out_path,exist_ok=True)

    mean_dice,std_dice=finite_mean_std(dice)
    mean_hd,std_hd=finite_mean_std(hd)
    mean_sen,std_sen=finite_mean_std(sen)
    mean_assd,std_assd=finite_mean_std(assd)


    rows=[
        ["Dice",mean_dice,std_dice],
        ["HD",mean_hd,std_hd],
        ["SEN",mean_sen,std_sen],
        ["ASSD",mean_assd,std_assd]
    ]


    with open(
        os.path.join(args.out_path,"test_summary.csv"),
        "w",
        newline=""
    ) as f:

        w=csv.writer(f)

        w.writerow(
            ["metric","mean","std"]
        )

        w.writerows(rows)



    with open(
        os.path.join(args.out_path,"test_slice_details.csv"),
        "w",
        newline=""
    ) as f:

        w=csv.DictWriter(
            f,
            fieldnames=[
                "slice_index",
                "dice",
                "hd",
                "sen",
                "assd",
                "prediction_path",
            ]
        )

        w.writeheader()
        w.writerows(detail)



    result={
        "Dice":mean_dice,
        "HD":mean_hd,
        "SEN":mean_sen,
        "ASSD":mean_assd,
        "num_slice":total,
        "prediction_root":(
            os.path.join(args.out_path,"test_predictions")
            if args.save_predictions
            else None
        )
    }


    with open(
        os.path.join(args.out_path,"test_summary.json"),
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )


    with open(
        os.path.join(args.out_path,"test_summary.txt"),
        "w"
    ) as f:

        for k,v in result.items():
            f.write(f"{k}:{v}\n")


    print(result)

# ================= Test =================

@torch.no_grad()
def test_model(model,loader,args):

    model.eval()

    dice_all=[]
    hd_all=[]
    sen_all=[]
    assd_all=[]

    detail=[]

    total=0


    for step,batch in enumerate(loader,1):

        img=batch["image"].to(device)
        mask=batch["mask"].float().to(device)


        img=prepare_model_input(
            img,
            args.repeat_channels
        )


        out=model(img)

        out=get_main_output(out)


        if out.shape[-2:]!=mask.shape[-2:]:

            out=torch.nn.functional.interpolate(
                out,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            )


        prob=torch.sigmoid(out)

        pred=(prob>=args.threshold).float()

        prediction_paths=[]
        if args.save_predictions:
            prediction_paths=save_prediction_batch(
                pred,
                batch,
                total,
                args
            )



        dice,hd,sen,assd,info=compute_2d_dice_hd_include_empty(
            pred,
            mask,
            total,
            hd_empty_value=args.hd_empty_value
        )

        for i,item in enumerate(info):
            item["prediction_path"]=(
                prediction_paths[i]
                if args.save_predictions
                else ""
            )


        dice_all+=dice
        hd_all+=hd
        sen_all+=sen
        assd_all+=assd

        detail+=info


        total+=img.size(0)



        print(
            f"{step}/{len(loader)} "
            f"Dice:{np.mean(dice):.4f} "
            f"HD:{np.mean(hd):.3f} "
            f"SEN:{np.mean(sen):.4f} "
            f"ASSD:{np.mean(assd):.3f}"
        )


    save_summary_for_plot(
        dice_all,
        hd_all,
        sen_all,
        assd_all,
        detail,
        total,
        args
    )





# ================= Main =================


if __name__=="__main__":


    if args.deterministic:

        cudnn.benchmark=False
        cudnn.deterministic=True

    else:

        cudnn.benchmark=True
        cudnn.deterministic=False



    set_seed(args.seed)



    os.makedirs(
        args.out_path,
        exist_ok=True
    )


    print("Device:",device)

    print("Fold:",args.fold)

    print("Checkpoint:",args.checkpoint)

    print("Output:",args.out_path)

    print(
        "Prediction output:",
        os.path.join(args.out_path,"test_predictions")
        if args.save_predictions
        else "disabled"
    )



    # ================= Dataset =================

    test_dataset=SegDataset(
        root=args.root_path,
        k_fold=args.fold,
        split="test",
        image_size=(
            args.img_size,
            args.img_size
        )
    )


    test_loader=DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )



    sample=test_dataset[0]

    print(
        "image:",
        sample["image"].shape
    )

    print(
        "mask:",
        sample["mask"].shape
    )



    # ================= Model =================


    model=build_model(args)


    model=load_checkpoint(
        model,
        args.checkpoint
    )


    model.eval()



    test_model(
        model,
        test_loader,
        args
    )
