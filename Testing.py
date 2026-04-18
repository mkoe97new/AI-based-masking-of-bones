import os
import sys
import datetime
import time
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm

from Callbacks import SaveTestDataResults
from Helper import Helper
from Patching import Patching


class Testing:
    def __init__(self, config, image_generator_config_json):
        self.config = config
        self.image_generator_config_json = image_generator_config_json
        print("{}: Reading configuration file...".format(datetime.datetime.now()))

        # image_generator_config_json
        self.directory = image_generator_config_json['Directories']['BaseDirectory']
        if self.directory is None:
            self.input_dir = Path(image_generator_config_json['Directories']['InputDirectory'])
            self.test_dir = Path(image_generator_config_json['Directories']['TestDirectory'])
            self.log_dir = Path("logs")
        else:
            base_path = Path(self.directory)
            self.input_dir = base_path / Path(image_generator_config_json['Directories']['InputDirectory'])
            self.test_dir = base_path / Path(image_generator_config_json['Directories']['TestDirectory'])
            self.log_dir = base_path / Path("logs")

        self.seed = self.image_generator_config_json['Settings']['Seed']

        # training configuration
        self.model_name = self.config['Settings']['ModelName']
        self.output_channel_num = len(self.config['Settings']['SegmentationClasses'])
        self.label_classes = self.config['Settings']['SegmentationClasses']
        self.test_frequency = self.config['Settings']['TestFrequency']

        self.patch_size = self.config['Settings']['PatchSize']
        if self.directory is None:
            self.summary_log_dir = Path(config['Settings']['SummaryLogDir']) / Path(self.model_name)
            self.network_saving_path = Path(config['Settings']['Network']['NetworkSavingPath']) / Path(self.model_name)
        else:
            base_path = Path(self.directory)
            self.summary_log_dir = base_path / Path(config['Settings']['SummaryLogDir']) / Path(self.model_name)
            self.network_saving_path = base_path / Path(config['Settings']['Network']['NetworkSavingPath']) / Path(
                self.model_name)

        self.activation_function_name = self.config['Settings']['Network']['ActivationFunction']
        self.existing_model = config['Settings']['Network']['ExistingModel']
        self.optimizer_name = self.config['Settings']['Network']['Optimizer']['Name']


        self.network = None
        self.train_patching = Patching(self.patch_size, self.test_dir)

        os.makedirs(self.log_dir, exist_ok=True)
        log_path = self.log_dir / Path(self.model_name + "test_output.log")
        log_file = log_path.open("w", encoding="utf-8")
        sys.stdout = log_file

    def print_config_parames(self):
        # Define the data
        data = [
            ["Model name", self.model_name],
            ["Patch size", self.patch_size],
            ["Activation function", self.activation_function_name],
            ["Optimizer", self.optimizer_name],
            ["ExistingModel", self.existing_model],
            ["Test Frequency", self.test_frequency]
        ]
        headers = ["Parameter", "Value"]
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

    def test(self):
        print("{}: VNet Tensorflow test start...".format(datetime.datetime.now()))
        self.print_config_parames()
        Helper.set_seed(self.seed)
        start_time = time.time()

        optimizer = Helper.get_optimizer(self.optimizer_name, 0.01)
        loss_function = Helper.get_loss_function(self.activation_function_name)
        num_dirs = sum(1 for p in self.network_saving_path.iterdir() if p.is_dir())
        last_patch_position, total_patches = Helper.count_patches(self.test_dir, self.patch_size)

        for current_patch_number in tqdm(range(1, num_dirs, self.test_frequency)):
            model_path = self.network_saving_path / Path("patch" + str(current_patch_number - 1)) / self.existing_model

            self.network = Helper.load_VNet(model_path, optimizer, loss_function)
            test_callback = SaveTestDataResults(current_patch_number, self.summary_log_dir, self.patch_size, self.test_dir, self.seed, len(self.label_classes), total_patches, last_patch_position, loss_function, "full", True)
            test_callback.set_model(self.network)
            test_callback.on_train_begin()
            test_callback.on_train_end()



        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")