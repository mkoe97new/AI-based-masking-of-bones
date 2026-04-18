import datetime
import math
import os
import glob
import random
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import time

from tabulate import tabulate
from tqdm import tqdm

from Image8 import Image
from Callbacks import SaveTrainDataResults, SaveNetwork, SaveTestDataResults
from VNet10 import VNet
from Patching import Patching


class ImageModel:
    def __init__(self, config, image_generator_config_json):
        self.config = config
        self.image_generator_config_json = image_generator_config_json
        print("{}: Reading configuration file...".format(datetime.datetime.now()))

        # image_generator_config_json
        self.directory = image_generator_config_json['Directories']['BaseDirectory']
        if self.directory is None:
            self.input_dir = Path(image_generator_config_json['Directories']['InputDirectory'])
            self.train_dir = Path(image_generator_config_json['Directories']['TrainDirectory'])
            self.test_dir = Path(image_generator_config_json['Directories']['TestDirectory'])
            self.pred_dir = Path(image_generator_config_json['Directories']['PredDirectory'])
            self.log_dir = Path("logs")
        else:
            base_path = Path(self.directory)
            self.input_dir = base_path / Path(image_generator_config_json['Directories']['InputDirectory'])
            self.train_dir = base_path / Path(image_generator_config_json['Directories']['TrainDirectory'])
            self.test_dir = base_path / Path(image_generator_config_json['Directories']['TestDirectory'])
            self.pred_dir = base_path / Path(image_generator_config_json['Directories']['PredDirectory'])
            self.log_dir = base_path / Path("logs")

        self.seed = self.image_generator_config_json['Settings']['Seed']
        self.downsampling = self.image_generator_config_json['Settings']['Downsampling']

        # config
        self.server = self.config['Settings']['Server']

        # training configuration
        self.model_name = self.config['Settings']['ModelName']
        self.input_channel_num = self.config['Settings']['InputChannels']  # is 1 because it is greyscale
        self.output_channel_num = len(self.config['Settings']['SegmentationClasses'])
        self.label_classes = self.config['Settings']['SegmentationClasses']

        self.batch_size = self.config['Settings']['BatchSize']
        self.patch_size = self.config['Settings']['PatchSize']
        self.testing = self.config['Settings']['Testing']
        if self.directory is None:
            self.summary_log_dir = Path(config['Settings']['SummaryLogDir']) / Path(self.model_name)
            self.network_saving_path = Path(config['Settings']['Network']['NetworkSavingPath']) / Path(self.model_name)
            existing_model_value = config['Settings']['Network']['ExistingModel']
            self.existing_model = Path(existing_model_value) if existing_model_value is not None else None
        else:
            base_path = Path(self.directory)
            self.summary_log_dir = base_path / Path(config['Settings']['SummaryLogDir']) / Path(self.model_name)
            self.network_saving_path = base_path / Path(config['Settings']['Network']['NetworkSavingPath']) / Path(self.model_name)
            existing_model_value = config['Settings']['Network']['ExistingModel']
            self.existing_model = base_path / Path(existing_model_value) if existing_model_value is not None else None

        self.epochs = self.config['Settings']['Network']['Epochs']
        self.dropout = self.config['Settings']['Network']['Dropout']
        self.dropoutrate = self.config['Settings']['Network']['DropoutRate']
        self.initializer_name = self.config['Settings']['Network']['Initializer']
        self.loss_function_name = self.config['Settings']['Network']['Loss']
        self.activation_function_name = self.config['Settings']['Network']['ActivationFunction']
        self.network_saving_range = self.config['Settings']['Network']['NetworkSavingRange']
        self.optimizer_name = self.config['Settings']['Network']['Optimizer']['Name']
        self.learning_rate = self.config['Settings']['Network']['Optimizer']['LearningRate']

        self.train_dataset = None
        self.network = None
        self.train_patching = Patching(self.patch_size, self.train_dir)

        os.makedirs(self.log_dir, exist_ok=True)
        log_path = self.log_dir / Path(self.model_name + "_output.log")
        log_file = log_path.open("w", encoding="utf-8")
        sys.stdout = log_file

    def print_config_parames(self):
        # Define the data
        data = [
            ["Batch size train", self.batch_size],
            ["Patch size", self.patch_size],
            ["Epochs", self.epochs],
            ["Dropout", self.dropout],
            ["Dropout rate", self.dropoutrate],
            ["Loss function", self.loss_function_name],
            ["Activation function", self.activation_function_name],
            ["Optimizer", self.optimizer_name],
            ["Initialization", self.initializer_name],
            ["Learning Rate", self.learning_rate]
        ]
        headers = ["Parameter", "Value"]
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

    def set_seed(self):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def build_VNet(self, optimizer, loss_function):
        print("Build VNet...")
        if self.dropout:
            self.network = VNet(self.initializer_name, self.activation_function_name, self.dropoutrate)
        else:
            self.network = VNet(self.initializer_name, self.activation_function_name, 0.0)
        self.network.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

    def load_VNet(self, optimizer, loss_function):
        print("Loading VNet...")
        self.network = tf.keras.models.load_model(self.existing_model, custom_objects={'VNet': VNet})
        self.network.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

    def get_optimizer(self):
        if self.optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        elif self.optimizer_name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        return optimizer

    def get_loss_function(self):
        if self.loss_function_name == 'CategoricalCrossentropyLoss':
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        elif self.loss_function_name == 'DiceLoss':
            loss_function = tf.keras.losses.Dice()
        else:
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        return loss_function

    def get_dimsize_from_mhd(self, file_path: Path):
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("DimSize"):
                    _, value = line.split("=")
                    dims = tuple(map(int, value.strip().split()))
                    return dims
        raise ValueError(f"No DimSize found in {file_path}")

    def dim_sizes_to_dhw_config(self, whd_tuple):
        whd_list = list(whd_tuple)
        dhw_list = [whd_list[2], whd_list[1], whd_list[0]]
        dhw_tuple = tuple(dhw_list)

        return dhw_tuple

    def read_dimension(self, read_in_dir):
        dim_sizes = []

        for mhd_file in read_in_dir.glob("*.mhd"):
            dims = self.get_dimsize_from_mhd(mhd_file)
            dim_sizes.append(dims)

        if not dim_sizes:
            raise ValueError("No .mhd files found")

        first = dim_sizes[0]

        for file_dims in dim_sizes:
            if file_dims != first:
                raise ValueError(
                    f"Inconsistent DimSize found: {first} vs {file_dims}"
                )

        dhw = self.dim_sizes_to_dhw_config(first)
        return dhw

    def count_patches(self, read_in_dir):
        dimensions = self.read_dimension(read_in_dir)
        patch_size_tuple = tuple(self.patch_size)
        num_patches = tuple(d // p for d, p in zip(dimensions, patch_size_tuple))
        total_patches = math.prod(num_patches)
        return list(num_patches), total_patches

    def init_callbacks(self, current_patch_number, total_patches, last_patch_position, loss_function, confusion_matrix):
        train_callback = SaveTrainDataResults(current_patch_number, self.summary_log_dir)
        model_callback = SaveNetwork(self.network_saving_range, current_patch_number, self.network_saving_path)
        test_callback = SaveTestDataResults(current_patch_number, self.summary_log_dir, self.patch_size, self.test_dir, self.seed, len(self.label_classes), total_patches, last_patch_position, loss_function, self.testing, confusion_matrix)
        return train_callback, model_callback, test_callback

    def train(self):
        print("{}: VNet Tensorflow training start...".format(datetime.datetime.now()))
        self.print_config_parames()
        self.set_seed()
        start_time = time.time()


        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        last_patch_position, total_patches = self.count_patches(self.train_dir)
        current_patch_position = [0,0,0]

        if not self.existing_model or not self.existing_model.exists():
            self.build_VNet(optimizer, loss_function)
        else:
            self.load_VNet(optimizer, loss_function)

        os.makedirs(self.network_saving_path, exist_ok=True)
        os.makedirs(self.summary_log_dir, exist_ok=True)


        print("Amount of training patches: {}".format(total_patches))
        for current_patch_number in tqdm(range(total_patches)):
            print("Patch Position: {}, Current Patch: {}".format(
                current_patch_position, current_patch_number
            ))
            last_patch = current_patch_number == total_patches - 1

            dataset, current_patch_position = self.train_patching.create_patch_dataset(
                current_patch_number,
                current_patch_position,
                last_patch_position
            )

            self.train_dataset = self.train_patching.shuffle(dataset, self.seed, self.batch_size)

            confusion_matrix = False
            if last_patch:
                confusion_matrix = True
            train_callback, model_callback, test_callback = self.init_callbacks(current_patch_number, total_patches, last_patch_position, loss_function, confusion_matrix)
            print("{}: Epochs starting ...".format(datetime.datetime.now()))
            if self.testing == "end" and last_patch:
                self.network.fit(self.train_dataset, epochs=self.epochs, verbose=0,
                                 callbacks=[train_callback, model_callback, test_callback])
            elif self.testing == "end" and not last_patch:
                self.network.fit(self.train_dataset, epochs=self.epochs, verbose=0,
                                 callbacks=[train_callback, model_callback])
            else:
                self.network.fit(self.train_dataset, epochs=self.epochs, verbose=0,
                                 callbacks=[train_callback, model_callback, test_callback])


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        return "in train"

    """
    def predict(self):
        print("{}: VNet Tensorflow Predict start...".format(datetime.datetime.now()))
        self.set_seed()

        optimizer = self.get_optimizer()
        if not self.network:
            self.load_VNet(optimizer)

        amount_of_patches = int((self.zero_padding_to[0] / self.patch_size[0]) * (
                self.zero_padding_to[1] / self.patch_size[1]) * (self.zero_padding_to[2] / self.patch_size[2]))

        column_number = -1  # weil ich jeden durchgang + 1 hab und sonst würde ich nicht bei 0 beginnen sondern schon bei 1
        row_number = 0
        depth_number = 0

        file_pattern = os.path.join(self.predict_data_dir, f"*raw*")
        files = glob.glob(file_pattern)

        print_volume = None
        volumes = np.zeros((self.zero_padding_to[0], self.zero_padding_to[1], self.zero_padding_to[2]))
        volumes = np.expand_dims(volumes, axis=0)
        for j in range(len(files) - 1):  # -1 weil ich ja schon eine dimension am anfang eröffne
            volume = np.zeros((self.zero_padding_to[0], self.zero_padding_to[1], self.zero_padding_to[2]))
            volume_expanded = np.expand_dims(volume, axis=0)
            volumes = np.vstack((volumes, volume_expanded))

        for i in range(1, amount_of_patches):
            # contains only one element
            column_number, row_number, depth_number, all_labels_all_imgs_one_class = self.create_patch_dataset(
                i, column_number,
                row_number,
                depth_number,
                self.train_data_dir,
                "predict")

            start_width = column_number * self.patch_size[0]
            start_height = row_number * self.patch_size[1]
            start_depth = depth_number * self.patch_size[2]

            self.predict_dataset_tf = self.predict_dataset_tf.batch(self.batch_size_train)
            prediction = self.network.predict(self.predict_dataset_tf)

            for k in range(len(prediction)):
                patch_volume = np.argmax(prediction[k], axis=-1)
                print_volume = patch_volume

                volumes[k][start_width:start_width + self.patch_size[0],
                start_height:start_height + self.patch_size[1],
                start_depth:start_depth + self.patch_size[2]] += patch_volume

            break
        print("----------------------------------------------------------------------------------")
        print(print_volume)
        """
"""os.makedirs(self.prediction_save_dir + "/" + self.model_name, exist_ok=True)
for k in range(len(volumes)):
    file_path = self.prediction_save_dir + "/" + self.model_name + "/" + 'prediciton' + str(k)
    np.save(file_path, volumes[k])"""


