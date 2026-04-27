from os import path
from pathlib import Path

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import imageio.v3 as iio

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
        self.slicing_amount = config['Settings']['SlicingAmount']
        self.show_wrong_pixels = config['Settings']['ShowWrongPixels']

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
            image_path = random_csv_row["sample"]

        model_path = self.network_saving_path / self.existing_model
        self.network = Helper.load_VNet_without_compiling(model_path)

        pred_image, gt_image, pred_adapted_image = self.predict_image(image_path)
        worst_slices = self.slicing_comparison(pred_image, gt_image)
        self.save_worst_slices(pred_image, gt_image, pred_adapted_image, worst_slices)

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

    def slicing_comparison(self, pred_image, gt_image):
        pred_image_voxel = pred_image.voxel_data
        gt_image_voxel = gt_image.voxel_data

        errors = []
        for i in range(pred_image_voxel.shape[1]):
            pred_slice = pred_image_voxel[:, i, :]
            gt_slice = gt_image_voxel[:, i, :]

            error = 1 - self.dice_per_class(pred_slice, gt_slice)
            errors.append(error)

        errors = np.array(errors)
        worst_indices = np.argsort(errors)[-self.slicing_amount:][::-1]

        return worst_indices

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

    def save_worst_slices(self, pred_image, gt_image, pred_adatped_image, worst_slices):
        pred_image_voxel = pred_image.voxel_data
        gt_image_voxel = gt_image.voxel_data

        for rank, i in enumerate(worst_slices, start=1):
            pred_slice = pred_image_voxel[:, i, :]
            gt_slice = gt_image_voxel[:, i, :]
            pred_adatped_slice = pred_adatped_image[:, i, :]

            iio.imwrite(
                self.slicing_dir / f"{rank:02d}_slice_{i}_pred.png",
                self.label_to_rgb(pred_slice))

            iio.imwrite(
                self.slicing_dir / f"{rank:02d}_slice_{i}_gt.png",
                self.label_to_rgb(gt_slice))

            iio.imwrite(
                self.slicing_dir / f"{rank:02d}_slice_{i}_pred_adapted.png",
                self.label_to_rgb(pred_adatped_slice))

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











