import datetime
import shutil
from pathlib import Path
import random

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

        self.downsampling = config['Settings']['Downsampling']
        self.order_name = config['Settings']['OrderName']
        self.zero_padding_to = config['Settings']['ZeroPaddingTo']
        self.train_data_size = config['Settings']['TrainDataSize']
        self.seed = config['Settings']['Seed']


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



