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

        self.random = config['Settings']['Random']
        self.downsampling = config['Settings']['Downsampling']
        self.order_name = config['Settings']['OrderName']
        self.zero_padding_to = config['Settings']['ZeroPaddingTo']
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

            image_label.pad_image_new(self.zero_padding_to, self.downsampling, self.order_name)
            image.pad_image_new(self.zero_padding_to, self.downsampling, self.order_name)

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
                print("Random Roi not found {}".format(index))
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

    def contains_more_zeros_than_allowed(self, zero_max_percentage, label_roi):
        zero_fraction = np.mean(label_roi == 0)
        if zero_fraction > zero_max_percentage:
            return True
        else:
            return False


    def check_if_zero(self):
        print("{}: Zero checker...".format(datetime.datetime.now()))
        self.train_dir.mkdir(exist_ok=True)

        for original_mhd in sorted(self.test_dir.glob("original_img*.mhd")):
            case_id = original_mhd.stem[len("original_img"):]
            label_mhd = self.test_dir / f"label_img{case_id}.mhd"

            image_original = Image()
            image_label = Image()

            image_original.constructor_image(original_mhd, case_id)
            image_label.constructor_image(label_mhd, case_id)

            print(str(original_mhd))
            image_original.check_if_all_zero()
            print(str(label_mhd))
            image_label.check_if_all_zero()



