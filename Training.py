import datetime
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from pathlib import Path
import time

from tabulate import tabulate
from tqdm import tqdm

from Helper import Helper
from Callbacks import SaveTrainDataResults, SaveNetwork, SaveTestDataResults
from RandomPatchCallbacks import SaveTrainRandomDataResults, SaveRandomNetwork, SaveTestRandomDataResults
from VNet10 import VNet
from Patching import Patching


class Training:
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
        self.method = self.image_generator_config_json['Settings']['Method']

        # training configuration
        self.model_name = self.config['Settings']['ModelName']
        self.input_channel_num = self.config['Settings']['InputChannels']  # is 1 because it is greyscale
        self.label_classes = self.config['Settings']['SegmentationClasses']

        self.batch_size = self.config['Settings']['BatchSize']
        self.patch_size = self.image_generator_config_json['Settings']['PatchSize']
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

    def build_VNet(self, optimizer, loss_function):
        print("Build VNet...")
        if self.dropout:
            network = VNet(self.initializer_name, self.activation_function_name, self.dropoutrate)
        else:
            network = VNet(self.initializer_name, self.activation_function_name, 0.0)
        network.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
        return network

    def init_callbacks(self, current_patch_number, total_patches, last_patch_position, loss_function, confusion_matrix):
        train_callback = SaveTrainDataResults(current_patch_number, self.summary_log_dir)
        model_callback = SaveNetwork(self.network_saving_range, current_patch_number, self.network_saving_path)
        test_callback = SaveTestDataResults(current_patch_number, self.summary_log_dir, self.patch_size, self.test_dir, self.seed, len(self.label_classes), total_patches, last_patch_position, loss_function, self.testing, confusion_matrix)
        return train_callback, model_callback, test_callback

    def init_random_callbacks(self, loss_function):
        train_callback = SaveTrainRandomDataResults(self.summary_log_dir)
        model_callback = SaveRandomNetwork(self.network_saving_range, self.network_saving_path)
        test_callback = SaveTestRandomDataResults(self.summary_log_dir, self.patch_size, self.test_dir, self.seed, len(self.label_classes), loss_function, self.testing)
        return train_callback, model_callback, test_callback

    def train(self):
        print("{}: VNet Tensorflow training start...".format(datetime.datetime.now()))
        self.print_config_parames()
        Helper.set_seed(self.seed)
        start_time = time.time()


        optimizer = Helper.get_optimizer(self.optimizer_name, self.learning_rate)
        loss_function = Helper.get_loss_function(self.loss_function_name)

        last_patch_position, total_patches = Helper.count_patches(self.train_dir, self.patch_size)
        current_patch_position = [0,0,0]

        if not self.existing_model or not self.existing_model.exists():
            self.network = self.build_VNet(optimizer, loss_function)
        else:
            self.network = Helper.load_VNet(self.existing_model, optimizer, loss_function)

        os.makedirs(self.network_saving_path, exist_ok=True)
        os.makedirs(self.summary_log_dir, exist_ok=True)


        print("Amount of training patches: {}".format(total_patches))
        for current_patch_number in tqdm(range(total_patches)):
            print("Patch Position: {}, Current Patch: {}".format(
                current_patch_position, current_patch_number
            ))
            last_patch = current_patch_number == total_patches - 1

            train_dataset_with_paths, current_patch_position = self.train_patching.create_patch_dataset(
                current_patch_number,
                current_patch_position,
                last_patch_position
            )
            dataset = train_dataset_with_paths.map(
                lambda image, label, original_path, label_path: (image, label)
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

    def random_patch_train(self):
        print("{}: VNet Tensorflow training start...".format(datetime.datetime.now()))
        self.print_config_parames()
        Helper.set_seed(self.seed)
        start_time = time.time()

        optimizer = Helper.get_optimizer(self.optimizer_name, self.learning_rate)
        loss_function = Helper.get_loss_function(self.loss_function_name)

        if not self.existing_model or not self.existing_model.exists():
            self.network = self.build_VNet(optimizer, loss_function)
        else:
            self.network = Helper.load_VNet(self.existing_model, optimizer, loss_function)

        os.makedirs(self.network_saving_path, exist_ok=True)
        os.makedirs(self.summary_log_dir, exist_ok=True)

        dataset = self.train_patching.create_dataset()
        self.train_dataset = self.train_patching.shuffle(dataset, self.seed, self.batch_size)

        file_count = sum(len(files) for _, _, files in os.walk('.'))
        steps_per_epoch = math.ceil((file_count/4) / self.batch_size)
        train_callback, model_callback, test_callback = self.init_random_callbacks(loss_function)
        print("{}: Epochs starting ...".format(datetime.datetime.now()))
        self.network.fit(self.train_dataset, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
            verbose=2, callbacks=[train_callback, model_callback, test_callback])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

