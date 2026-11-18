import csv
import os
from os import path
from pathlib import Path

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import imageio.v3 as iio
from scipy.ndimage import binary_dilation, generate_binary_structure

from Helper import Helper
from Image8 import Image
from Patching import Patching


class Slicing:
    def __init__(self, config, image_generator_config_json):
        self.directory = image_generator_config_json['Directories']['BaseDirectory']
        if self.directory is None:
            self.test_dir = Path(image_generator_config_json['Directories']['TestDirectory'])
        else:
            base_path = Path(self.directory)
            self.test_dir = base_path / Path(image_generator_config_json['Directories']['TestDirectory'])

        self.seed = image_generator_config_json['Settings']['Seed']
        self.patch_size = image_generator_config_json['Settings']['PatchSize']
        self.resize_to = image_generator_config_json['Settings']['ResizeTo']
        self.downsampling = image_generator_config_json['Settings']['Downsampling']

        self.model_name = config['Settings']['ModelName']
        self.label_classes = config['Settings']['SegmentationClasses']
        self.slicing_worst_image = config['Settings']['SlicingWorstImage']
        self.select_worst_slices = config['Settings']['SelectWorstSlices']
        self.slicing_amount = config['Settings']['SlicingAmount']
        self.show_wrong_pixels = config['Settings']['ShowWrongPixels']
        self.axis = config['Settings']['Axis']
        self.no_empty = config['Settings']['NoEmpty']

        if self.directory is None:
            self.summary_log_dir = Path(config['Settings']['SummaryLogDir']) / Path(self.model_name)
            self.network_saving_path = Path(config['Settings']['Network']['NetworkSavingPath']) / Path(self.model_name)
            self.slicing_dir = Path(config['Settings']['SlicingDir']) / Path(self.model_name)
        else:
            base_path = Path(self.directory)
            self.summary_log_dir = base_path / Path(config['Settings']['SummaryLogDir']) / Path(self.model_name)
            self.network_saving_path = base_path / Path(config['Settings']['Network']['NetworkSavingPath']) / Path(
                self.model_name)
            self.slicing_dir = base_path / Path(config['Settings']['SlicingDir']) / Path(self.model_name)

        self.existing_model = config['Settings']['Network']['ExistingModel']

        self.test_patching = Patching(self.patch_size, "not_necessary_here")

        self.network = None

    def generate_2d_slices(self):
        self.slicing_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.summary_log_dir / "sample_test_log.csv"
        df = pd.read_csv(csv_path)
        if self.slicing_worst_image:
            df_columns = ["accuracy_bg"]
            for seg in self.label_classes:
                if seg != 0:
                    df_columns.append("accuracy_class_" + str(seg))
            df["mean_accuracy"] = df[df_columns].mean(axis=1)
            worst_csv_row = df.loc[df["mean_accuracy"].idxmin()]
            image_path = worst_csv_row["sample"]
        else:
            random_csv_row = df.sample(n=1, random_state=self.seed)
            s = str(random_csv_row["sample"].iloc[0])
            image_path = re.sub(r'^\d+\s+', '', s)

        model_path = self.network_saving_path / self.existing_model
        self.network = Helper.load_VNet_without_compiling(model_path)

        pred_image, gt_image, pred_adapted_image = self.predict_image(image_path)
        axis_to_cut = self.axis
        for axis in axis_to_cut:
            if self.select_worst_slices:
                selected_slices = self.select_worst_slices_function(pred_image, gt_image, image_path, self.summary_log_dir, axis)
            else:
                selected_slices = self.select_random_slices(pred_image, gt_image, image_path, self.summary_log_dir, axis)
            self.save_worst_slices(pred_image, gt_image, pred_adapted_image, selected_slices, axis)

    def predict_image(self, image_original_path_string):
        image_original_path = Path(image_original_path_string)
        image_label_path = image_original_path.with_name(image_original_path.name.replace("original", "label"))
        match = re.search(r"(\d+)", image_original_path_string)
        number = int(match.group(1))
        pred_image = Image()

        pred_image.empty_image_constructor(self.resize_to, self.downsampling, number)
        gt_image = Image()
        gt_image.constructor_image(image_label_path, number)

        current_patch_position = [0, 0, 0]
        last_patch_position, total_patches = Helper.count_patches(self.test_dir, self.patch_size)

        for current_patch_number in range(total_patches):

            dataset, current_patch_position = self.test_patching.create_patch_image(
                image_original_path,
                image_label_path,
                current_patch_number,
                current_patch_position,
                last_patch_position,
                number
            )
            dataset = self.test_patching.shuffle(dataset, self.seed, 1)

            for image_patch, gt_patch in dataset:
                pred_logits = self.network(image_patch, training=False)
                pred = tf.argmax(pred_logits, axis=-1)

                pred_image.set_roi(pred, current_patch_position, self.patch_size)

        if self.show_wrong_pixels:
            result = self.change_pred_image_including_wrong_classes(pred_image, gt_image)
        else:
            result = None
        return pred_image, gt_image, result

    def change_pred_image_including_wrong_classes(self, pred_image, gt_image):
        result = pred_image.voxel_data.copy()

        mask = pred_image.voxel_data != gt_image.voxel_data  # where prediction is wrong
        result[mask] = pred_image.voxel_data[mask] + 10
        return result

    def mean_iou_per_class(self, pred, gt, mask=None):
        if mask is None:
            mask = np.ones_like(gt, dtype=bool)

        ious = []
        for c in range(len(self.label_classes)):
            pred_c = (pred == c) & mask
            gt_c = (gt == c) & mask

            intersection = np.sum(pred_c & gt_c)
            union = np.sum(pred_c | gt_c)

            if union == 0:
                continue  # skip empty class

            ious.append(intersection / union)

        mean_iou = np.mean(ious) if ious else 1.0
        return mean_iou, ious  # (overall mean, per-class list)


    def accuracy_score(self, pred, gt, mask=None):
        if mask is None:
            mask = np.ones_like(gt, dtype=bool)

        total = mask.sum()
        if total == 0:
            return np.nan

        correct = (pred == gt) & mask
        return correct.sum() / total

    def get_border_mask(self, gt_slice, label_a, label_b, width=5):
        """
        Band of ~`width` pixels straddling the border between two classes.
        Dilates each class mask by width//2 and takes the overlap.
        """
        mask_a = gt_slice == label_a
        mask_b = gt_slice == label_b

        struct = generate_binary_structure(2, 1)
        half_width = max(width // 2, 1)

        dilated_a = binary_dilation(mask_a, structure=struct, iterations=half_width)
        dilated_b = binary_dilation(mask_b, structure=struct, iterations=half_width)

        return dilated_a & dilated_b

    def get_all_transition_mask(self, gt_slice, outer=0, middle=1, inner=2, width=5):
        """Union of inner/middle border and middle/outer border."""
        inner_middle = self.get_border_mask(gt_slice, inner, middle, width)
        middle_outer = self.get_border_mask(gt_slice, middle, outer, width)
        return inner_middle | middle_outer

    def save_slice_metrics(self, pred_image_voxel, gt_image_voxel, indices, image_name, csv_path, errors, axis):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.isfile(csv_path)

        fieldnames = ["image_name", "axis", "slice_number", "accuracy", "meanIoU",
                      "accuracy_transition_region", "meanIoU_transition_region", "dice_score"]

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for k in range(len(indices)):
                idx = indices[k]
                pred_slice = np.take(pred_image_voxel, idx, axis=axis)
                gt_slice = np.take(gt_image_voxel, idx, axis=axis)

                accuracy = self.accuracy_score(pred_slice, gt_slice)
                mean_iou, _ = self.mean_iou_per_class(pred_slice, gt_slice)

                transition_mask = self.get_all_transition_mask(gt_slice)
                accuracy_tr = self.accuracy_score(pred_slice, gt_slice, mask=transition_mask)
                mean_iou_tr, _ = self.mean_iou_per_class(pred_slice, gt_slice, mask=transition_mask)

                writer.writerow({
                    "image_name": image_name,
                    "axis": axis,
                    "slice_number": int(idx),
                    "accuracy": accuracy,
                    "meanIoU": mean_iou,
                    "accuracy_transition_region": accuracy_tr,
                    "meanIoU_transition_region": mean_iou_tr,
                    "dice_score": 1 - float(errors[k]),
                })

    def select_worst_slices_function(self, pred_image, gt_image, image_name, csv_dir, axis=1):
        pred_image_voxel = pred_image.voxel_data
        gt_image_voxel = gt_image.voxel_data

        candidate_indices = []
        errors = []
        for i in range(pred_image_voxel.shape[axis]):
            pred_slice = np.take(pred_image_voxel, i, axis=axis)
            gt_slice = np.take(gt_image_voxel, i, axis=axis)
            if self.no_empty and not np.any(gt_slice != 0):
                continue

            candidate_indices.append(i)
            errors.append(1 - self.dice_per_class(pred_slice, gt_slice))

        print("pred shape worst slice extraction:", pred_image_voxel.shape)
        print("gt shape worst slice extraction:  ", gt_image_voxel.shape)

        candidate_indices = np.array(candidate_indices, dtype=int)
        errors = np.array(errors)

        order = np.argsort(errors)[::-1][:self.slicing_amount]  # positions of the worst, worst first
        worst_indices = candidate_indices[order]  # real slice numbers
        worst_errors = errors[order]  # their errors, same order

        self.save_slice_metrics(
            pred_image_voxel, gt_image_voxel, worst_indices,
            image_name=image_name,
            csv_path=os.path.join(csv_dir, "worst_slices.csv"),
            errors=worst_errors,
            axis=axis
        )

        return worst_indices

    def select_random_slices(self, pred_image, gt_image, image_name, csv_dir, axis=1):
        pred_image_voxel = pred_image.voxel_data
        gt_image_voxel = gt_image.voxel_data

        n = pred_image_voxel.shape[axis]
        rng = np.random.default_rng(seed=self.seed)
        random_indices = rng.choice(n, size=min(self.slicing_amount, n), replace=False)

        errors = np.array([
            1 - self.dice_per_class(
                np.take(pred_image_voxel, i, axis=axis),
                np.take(gt_image_voxel, i, axis=axis),
            )
            for i in random_indices
        ])

        self.save_slice_metrics(
            pred_image_voxel, gt_image_voxel, random_indices,
            image_name=image_name,
            csv_path=os.path.join(csv_dir, "random_slices.csv"),
            errors=errors,
            axis=axis
        )

        return random_indices

    def dice_per_class(self, pred, gt):
        dices = []
        for c in range(len(self.label_classes)):
            pred_c = (pred == c)
            gt_c = (gt == c)

            intersection = np.sum(pred_c & gt_c)
            denom = np.sum(pred_c) + np.sum(gt_c)

            if denom == 0:
                continue  # skip empty class

            dice = 2 * intersection / denom
            dices.append(dice)

        return np.mean(dices) if dices else 1.0

    def save_worst_slices(self, pred_image, gt_image, pred_adapted_image, worst_slices, axis = 1):
        pred_image_voxel = pred_image.voxel_data
        gt_image_voxel = gt_image.voxel_data
        image_number = pred_image.image_number

        for rank, i in enumerate(worst_slices, start=1):
            pred_adapted_slice = None

            if axis == 0:
                pred_slice = pred_image_voxel[i, :, :]
                gt_slice = gt_image_voxel[i, :, :]
                if pred_adapted_image is not None:
                    pred_adapted_slice = pred_adapted_image[i, :, :]
            elif axis == 1:
                pred_slice = pred_image_voxel[:, i, :]
                gt_slice = gt_image_voxel[:, i, :]
                if pred_adapted_image is not None:
                    pred_adapted_slice = pred_adapted_image[:, i, :]
            elif axis == 2:
                pred_slice = pred_image_voxel[:, :, i]
                gt_slice = gt_image_voxel[:, :, i]
                if pred_adapted_image is not None:
                    pred_adapted_slice = pred_adapted_image[:, :, i]

            print("pred shape image saving:", pred_image_voxel.shape)
            print("gt shape image saving:  ", gt_image_voxel.shape)

            iio.imwrite(
                self.slicing_dir / f"{rank:02d}_slice_{i}_image_{image_number}_pred_{axis}_axis.png",
                self.label_to_rgb(pred_slice))

            iio.imwrite(
                self.slicing_dir / f"{rank:02d}_slice_{i}_image_{image_number}_gt_{axis}_axis.png",
                self.label_to_rgb(gt_slice))

            iio.imwrite(
                self.slicing_dir / f"{rank:02d}_slice_{i}_pred_image_{image_number}_adapted_{axis}_axis.png",
                self.label_to_rgb(pred_adapted_slice))

    def label_to_rgb(self, slice_2d):
        h, w = slice_2d.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        rgb[slice_2d == 0] = [0, 0, 0]  # black
        rgb[slice_2d == 1] = [96, 108, 56]  # #606C38
        rgb[slice_2d == 2] = [188, 108, 37]  # #BC6C25
        rgb[slice_2d == 10] = [254, 250, 240]  # black
        rgb[slice_2d == 11] = [40, 54, 24]  # #606C38
        rgb[slice_2d == 12] = [221, 161, 94]  # #BC6C25

        return rgb













