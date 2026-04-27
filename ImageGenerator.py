import datetime
import glob
import os
import shutil
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm

from Image8 import Image

class ImageGenerator:
    def __init__(self, config):
        self.directory = config['Directories']['BaseDirectory']
        if self.directory is None:
            self.input_dir = Path(config['Directories']['InputDirectory'])
            self.train_dir = Path(config['Directories']['TrainDirectory'])
            self.test_dir = Path(config['Directories']['TestDirectory'])
        else:
            base_path = Path(self.directory)
            self.input_dir = base_path / Path(config['Directories']['InputDirectory'])
            self.train_dir = base_path / Path(config['Directories']['TrainDirectory'])
            self.test_dir = base_path / Path(config['Directories']['TestDirectory'])

        self.method = config['Settings']['Method']
        self.downsampling = config['Settings']['Downsampling']
        self.order_name = config['Settings']['OrderName']
        self.resize_to = config['Settings']['ResizeTo']
        self.patch_size = config['Settings']['PatchSize']
        self.train_data_size = config['Settings']['TrainDataSize']
        self.seed = config['Settings']['Seed']

        self.zero_max_percentage = config['RandomGeneratorSpecifics']['ZeroMaxPercentage']
        self.retry_amount = config['RandomGeneratorSpecifics']['RetryAmount']
        self.fg_patches_percentage = config['RandomGeneratorSpecifics']['FgPatchesPercentage']
        self.train_samples = config['RandomGeneratorSpecifics']['TrainSamples']


    def generate_images(self):
        print("{}: Generate images...".format(datetime.datetime.now()))
        self.train_dir.mkdir(exist_ok=True)
        for index, path in enumerate(self.input_dir.glob("*.mhd")):
            image = Image()
            image_label = Image()
            print("{}: Images processed: {}".format(datetime.datetime.now(), (index)))
            image_label.constructor_image(path, index)
            image.constructor_original_image(image_label)

            image_label.resize_image(self.resize_to, self.downsampling, self.order_name)
            image.resize_image(self.resize_to, self.downsampling, self.order_name)

            image_label.generate_image_label()
            image.generate_image_original()

            image_label.save_image(self.train_dir, False)
            image.save_image(self.train_dir, True)

    def train_test_split(self):
        self.test_dir.mkdir(exist_ok=True)

        original_prefix = "original_img"
        label_prefix = "label_img"

        pairs = []
        for original_mhd in self.train_dir.glob(f"{original_prefix}*.mhd"):
            case_id = original_mhd.stem[len(original_prefix):]

            files = [
                self.train_dir / f"{original_prefix}{case_id}.mhd",
                self.train_dir / f"{original_prefix}{case_id}.raw",
                self.train_dir / f"{label_prefix}{case_id}.mhd",
                self.train_dir / f"{label_prefix}{case_id}.raw",
            ]

            if all(f.exists() for f in files):
                pairs.append(files)
            else:
                missing = [f.name for f in files if not f.exists()]
                print(f"Warning: incomplete case {case_id}, missing: {missing}")
        if not pairs:
            print("No complete cases found.")
            return

        random.seed(self.seed)
        random.shuffle(pairs)

        split_index = int(len(pairs) * self.train_data_size)
        train_pairs = pairs[:split_index]
        test_pairs = pairs[split_index:]

        for original_mhd, original_raw, label_mhd, label_raw in test_pairs:
            shutil.move(str(original_mhd), str(self.test_dir / original_mhd.name))
            shutil.move(str(original_raw), str(self.test_dir  / original_raw.name))
            shutil.move(str(label_mhd), str(self.test_dir  / label_mhd.name))
            shutil.move(str(label_raw), str(self.test_dir  / label_raw.name))

        print(f"Total pairs: {len(pairs)}")
        print(f"Train pairs kept in place: {len(train_pairs)}")
        print(f"Test pairs moved: {len(test_pairs)}")
        print(f"Test folder: {self.test_dir}")

    def generate_random_images(self):
        print("{}: Generate random patched images...".format(datetime.datetime.now()))
        self.train_dir.mkdir(exist_ok=True)
        random.seed(self.seed)
        original_files = sorted(self.input_dir.glob("*original*.mhd"))
        rng = random.Random(self.seed)
        no_image_created = 0

        pairs = []
        for _ in range(self.train_samples):
            orig = rng.choice(original_files)
            label = orig.with_name(orig.name.replace("original", "label"))
            pairs.append((orig, label))

        fg_patches_reached = False
        for index, (original_file, label_file) in enumerate(tqdm(pairs)):
            image = Image()
            image_label = Image()
            image.constructor_image(original_file, index)
            image_label.constructor_image(label_file, index)

            not_accepted = True
            i = 0
            while not_accepted and i < self.retry_amount:
                d_start = rng.randint(0, image.depth  - self.patch_size[0])
                h_start = rng.randint(0, image.height - self.patch_size[1])
                w_start = rng.randint(0, image.width  - self.patch_size[2])

                start_and_end = [
                    d_start, d_start + self.patch_size[0],
                    h_start, h_start + self.patch_size[1],
                    w_start, w_start + self.patch_size[2],
                ]

                image_roi = image.get_roi(start_and_end)
                label_roi = image_label.get_roi(start_and_end)

                not_accepted = self.contains_more_zeros_than_allowed(self.zero_max_percentage, label_roi)
                if fg_patches_reached:
                    break

                i += 1

            if i == self.retry_amount:
                no_image_created += 1
            else:
                image.voxel_data = image_roi
                image.depth = image_roi.shape[0]
                image.height = image_roi.shape[1]
                image.width = image_roi.shape[2]
                image_label.voxel_data = label_roi
                image_label.depth = label_roi.shape[0]
                image_label.height = label_roi.shape[1]
                image_label.width = label_roi.shape[2]

                image.save_image(self.train_dir, True)
                image_label.save_image(self.train_dir, False)

            if (index / len(pairs)) >= self.fg_patches_percentage:
                fg_patches_reached = True

        count = len(glob.glob(os.path.join(self.train_dir, "*original*.mhd")))
        print(f"Total train images created: {count}")
        print(f"Total not created_images: {no_image_created}")

    def contains_more_zeros_than_allowed(self, zero_max_percentage, label_roi):
        zero_fraction = np.mean(label_roi == 0)
        if zero_fraction > zero_max_percentage:
            return True
        else:
            return False

    def show_image_sizes(self):
        max_depth = 0
        max_height = 0
        max_width = 0

        min_depth = float('inf')
        min_height = float('inf')
        min_width = float('inf')

        min_bb_depth_lower = float('inf')
        min_bb_height_lower = float('inf')
        min_bb_width_lower = float('inf')

        max_bb_depth_upper = 0
        max_bb_height_upper = 0
        max_bb_width_upper = 0

        max_bb_depth_lower = 0
        max_bb_height_lower = 0
        max_bb_width_lower = 0

        min_bb_depth_upper = float('inf')
        min_bb_height_upper = float('inf')
        min_bb_width_upper = float('inf')

        for index, path in enumerate(self.input_dir.glob("*.mhd")):
            image = Image()
            image.constructor_image(path, index)
            print(f"{path.name}: {image.depth}, {image.height}, {image.width}")

            bbox = image.get_bounding_box()
            if bbox is not None:
                (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
                print(f"Bounding box:")
                print(f"  Depth: {z_min} → {z_max}")
                print(f"  Height: {y_min} → {y_max}")
                print(f"  Width: {x_min} → {x_max}")
            else:
                print("  No foreground found!")

            max_depth = max(max_depth, image.depth)
            max_height = max(max_height, image.height)
            max_width = max(max_width, image.width)

            min_depth = min(min_depth, image.depth)
            min_height = min(min_height, image.height)
            min_width = min(min_width, image.width)

            min_bb_depth_lower = min(min_bb_depth_lower, z_min)
            min_bb_height_lower = min(min_bb_height_lower, y_min)
            min_bb_width_lower = min(min_bb_width_lower, x_min)

            max_bb_depth_upper = max(max_bb_depth_upper, z_max)
            max_bb_height_upper = max(max_bb_height_upper, y_max)
            max_bb_width_upper = max(max_bb_width_upper, x_max)

            max_bb_depth_lower = max(max_bb_depth_lower, z_min)
            max_bb_height_lower = max(max_bb_height_lower, y_min)
            max_bb_width_lower = max(max_bb_width_lower, x_min)

            min_bb_depth_upper = min(min_bb_depth_upper, z_max)
            min_bb_height_upper = min(min_bb_height_upper, y_max)
            min_bb_width_upper = min(min_bb_width_upper, x_max)

        print("\nBiggest dimensions:")
        print("Depth:", max_depth)
        print("Height:", max_height)
        print("Width:", max_width)

        print("\nSmallest dimensions:")
        print("Depth:", min_depth)
        print("Height:", min_height)
        print("Width:", min_width)

        print("\nGlobal outer boundaries:")
        print(f"Depth: {min_bb_depth_lower} → {max_bb_depth_upper}")
        print(f"Height: {min_bb_height_lower} → {max_bb_height_upper}")
        print(f"Width: {min_bb_width_lower} → {max_bb_width_upper}")

        print("\nGlobal inner boundaries:")
        print(f"Depth: {max_bb_depth_lower} → {min_bb_depth_upper}")
        print(f"Height: {max_bb_height_lower} → {min_bb_height_upper}")
        print(f"Width: {max_bb_width_lower} → {min_bb_width_upper}")


