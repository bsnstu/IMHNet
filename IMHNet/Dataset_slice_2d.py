# -*- coding: utf-8 -*-

"""
Slice-level Dataset

Train / Val / Test all return slice-level samples.

Input:
    channel 1:
        original image

    channel 2:
        myocardium mask

    channel 3:
        label_p prior map


Return:

image:
    [3,224,224]

mask:
    [1,224,224]

edge:
    [1,224,224]


Normalization:

concat(image,myo,label_p)
then

tensor / 255.0

"""

import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image

import numpy as np



class SegDataset(Dataset):

    def __init__(
        self,
        root,
        k_fold=1,
        split="train",

        image_size=(224,224),
        crop_center=(112,128),

        image_suffix=".png",
        mask_suffix=".png",
        myo_suffix=".png",
        edge_suffix=".png",

        label_p_path=None,

        augment=False,

        normalize=True,

        image_mean=0.5,
        image_std=0.5,

        augment_label_p=True,

    ):


        self.root = Path(root)

        self.k_fold = k_fold

        self.split = split
        fold_name = f"k_{k_fold}"

        # =========================
        # data path
        # =========================

        self.image_root = (
            self.root /
            "split_data" /
            fold_name /
            split
        )


        self.mask_root = (
            self.root /
            "split_label" /
            fold_name /
            split
        )


        self.myo_root = (
            self.root /
            "split_myo" /
            fold_name /
            split
        )


        self.edge_root = (
            self.root /"split_edge" /fold_name /split
        )



        if label_p_path is None:
            self.label_p_path = (self.root /"split_data" /"split_train.png"
            )




        self.image_size = image_size
        self.crop_center = crop_center
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.myo_suffix = myo_suffix
        self.edge_suffix = edge_suffix
        self.augment = augment
        self.normalize = normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.augment_label_p = augment_label_p
        self.samples = []





        # =========================
        # collect cases
        # =========================


        image_cases = sorted(
            [
                p.name
                for p in self.image_root.iterdir()
                if p.is_dir()
            ]
        )


        mask_cases = sorted(
            [
                p.name
                for p in self.mask_root.iterdir()
                if p.is_dir()
            ]
        )


        myo_cases = sorted(
            [
                p.name
                for p in self.myo_root.iterdir()
                if p.is_dir()
            ]
        )


        edge_cases = sorted(
            [
                p.name
                for p in self.edge_root.iterdir()
                if p.is_dir()
            ]
        )



        # =========================
        # case consistency
        # =========================


        if not(
            set(image_cases)
            ==
            set(mask_cases)
            ==
            set(myo_cases)
            ==
            set(edge_cases)
        ):

            raise RuntimeError(
                "Case folders are inconsistent"
            )



        # =========================
        # build slice list
        # =========================


        for case_id in image_cases:


            image_dir = (
                self.image_root /case_id
            )


            mask_dir = (
                self.mask_root /case_id
            )


            myo_dir = (
                self.myo_root /case_id
            )


            edge_dir = (
                self.edge_root /case_id
            )



            image_files = sorted(
                image_dir.glob(
                    f"*{self.image_suffix}"
                )
            )


            mask_files = sorted(
                mask_dir.glob(
                    f"*{self.mask_suffix}"
                )
            )


            myo_files = sorted(
                myo_dir.glob(
                    f"*{self.myo_suffix}"
                )
            )


            edge_files = sorted(
                edge_dir.glob(
                    f"*{self.edge_suffix}"
                )
            )


            for slice_index,( image_path, mask_path,myo_path,edge_path) in enumerate(
                zip(
                    image_files,
                    mask_files,
                    myo_files,
                    edge_files
                )
            ):

                self.samples.append(

                    {

                    "image_path":
                        image_path,

                    "mask_path":
                        mask_path,

                    "myo_path":
                        myo_path,

                    "edge_path":
                        edge_path,


                    "case_id":
                        case_id,


                    "image_filename":
                        image_path.name,


                    "mask_filename":
                        mask_path.name,


                    "slice_index":
                        slice_index,

                    }

                )


    def __len__(self):

        return len(self.samples)





    # =========================
    # crop
    # =========================

    def _crop_by_center(
        self,
        img,
        center,
        crop_size
    ):
        center_x,center_y = center
        crop_w,crop_h = crop_size
        w,h = img.size
        left = int(
            center_x-crop_w//2
        )

        top = int(
            center_y-crop_h//2
        )
        right = left+crop_w
        bottom = top+crop_h

        if (
            left>=0
            and top>=0
            and right<=w
            and bottom<=h
        ):

            return img.crop(
                (
                    left,
                    top,
                    right,
                    bottom
                )
            )



        img_np=np.array(img)

        pad_left=max(
            0,
            -left
        )
        pad_top=max(
            0,
            -top
        )

        pad_right=max(
            0,
            right-w
        )

        pad_bottom=max(
            0,
            bottom-h
        )



        img_np=np.pad(

            img_np,

            (
                (
                    pad_top,
                    pad_bottom
                ),

                (
                    pad_left,
                    pad_right
                )
            ),

            mode="constant",

            constant_values=0
        )
        left += pad_left
        top += pad_top
        img_np = img_np[
            top:top+crop_h,
            left:left+crop_w
        ]

        return Image.fromarray(img_np.astype(np.uint8))



    # =========================
    # load image
    # =========================


    def _load_image(self,path):

        img = Image.open(path).convert("L")


        img = self._crop_by_center(

            img,

            self.crop_center,

            self.image_size
        )



        img=np.array(
            img
        ).astype(
            np.float32
        )


        img=np.expand_dims(
            img,
            axis=0
        )


        return torch.from_numpy(img).float()

        # =========================
        # load binary mask
        # =========================

    def _load_binary_mask(self, path):

        mask = Image.open(path).convert("L")
        mask = self._crop_by_center(mask, self.crop_center, self.image_size)
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, 0)

        return torch.from_numpy(mask).float()

    # =========================
    # load input myo mask
    # =========================

    def _load_input_binary_mask(self, path):

        mask = Image.open(path).convert("L")
        mask = self._crop_by_center(mask, self.crop_center, self.image_size)

        mask = np.array(mask).astype(np.float32)
        mask = (mask > 0).astype(np.float32) * 255.0

        mask = np.expand_dims(mask, 0)

        return torch.from_numpy(mask).float()

    # =========================
    # load label prior map
    # =========================

    def _load_input_image(self, path):

        img = Image.open(path).convert("L")

        img = self._crop_by_center(
            img,
            self.crop_center,
            self.image_size
        )

        img = np.array(img).astype(np.float32)

        img = np.expand_dims(img, 0)

        return torch.from_numpy(img).float()

    # =========================
    # augmentation
    # =========================

    def _augment_together(
            self,
            image,
            mask,
            myo,
            edge,
            label_p
    ):

        if torch.rand(1).item() < 0.5:

            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])
            myo = torch.flip(myo, [2])
            edge = torch.flip(edge, [2])

            if self.augment_label_p:
                label_p = torch.flip(label_p, [2])

        if torch.rand(1).item() < 0.5:

            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])
            myo = torch.flip(myo, [1])
            edge = torch.flip(edge, [1])

            if self.augment_label_p:
                label_p = torch.flip(label_p, [1])

        return image, mask, myo, edge, label_p

    # =========================
    # get item
    # =========================

    def __getitem__(self, index):

            sample = self.samples[index]

            image = self._load_image(
                sample["image_path"]
            )

            mask = self._load_binary_mask(
                sample["mask_path"]
            )

            myo = self._load_input_binary_mask(
                sample["myo_path"]
            )

            edge = self._load_binary_mask(
                sample["edge_path"]
            )

            label_p = self._load_input_image(
                self.label_p_path
            )

            if self.augment and self.split == "train":
                image, mask, myo, edge, label_p = self._augment_together(
                    image,
                    mask,
                    myo,
                    edge,
                    label_p
                )

            # concat first, normalize once

            input_tensor = torch.cat(
                [
                    image,
                    myo,
                    label_p
                ],
                dim=0
            ).float() / 255.0

            return {

                "image": input_tensor,

                "mask": mask,

                "edge": edge,

                "case_id": sample["case_id"],

                "image_filename": sample["image_filename"],

                "mask_filename": sample["mask_filename"],

                "slice_index": sample["slice_index"]

            }

    # =================================================
    # test
    # =================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root = r"./data/re_kfold"

    dataset = SegDataset(

        root=root,

        k_fold=1,

        split="train",

        image_size=(224, 224),

        crop_center=(112, 128),

        augment=True

    )

    loader = DataLoader(

        dataset,

        batch_size=4,

        shuffle=True,

        num_workers=0

    )

    batch = next(iter(loader))

    print("image:", batch["image"].shape)

    print("mask:", batch["mask"].shape)

    print("edge:", batch["edge"].shape)

    print("case:", batch["case_id"])
