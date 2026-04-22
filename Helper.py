import math
import os
import random

import numpy as np
import tensorflow as tf

from VNet10 import VNet


class Helper:

    @staticmethod
    def get_optimizer(optimizer_name, learning_rate):
        if optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        elif optimizer_name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        return optimizer

    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1e-5):
        y_true = tf.squeeze(y_true, axis=-1)  # -> [B, D, H, W]
        y_true = tf.cast(y_true, tf.int32)

        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)  # -> [B, D, H, W, 3]
        y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)

        y_pred = tf.cast(y_pred, tf.float32)

        # optionally ignore background (channel 0)
        y_true_one_hot = y_true_one_hot[..., 1:]
        y_pred = y_pred[..., 1:]

        intersection = tf.reduce_sum(y_true_one_hot * y_pred, axis=[1, 2, 3])
        denominator = tf.reduce_sum(y_true_one_hot + y_pred, axis=[1, 2, 3])

        dice = (2. * intersection + smooth) / (denominator + smooth)
        return 1 - tf.reduce_mean(dice)


    @staticmethod
    def get_loss_function(loss_function_name):
        if loss_function_name == 'CategoricalCrossentropyLoss':
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        elif loss_function_name == 'DiceLoss':
            loss_function = tf.keras.losses.Dice()
        elif loss_function_name == 'DiceLossCategoricalCrossentropyLoss':
            ce = tf.keras.losses.SparseCategoricalCrossentropy()
            def combined_loss(y_true, y_pred):
                return 0.5 * ce(y_true, y_pred) + 0.5 * Helper.dice_loss(y_true, y_pred)
            return combined_loss
        else:
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        return loss_function

    @staticmethod
    def load_VNet(model_path, optimizer, loss_function):
        print("Loading VNet...")
        network = tf.keras.models.load_model(model_path, custom_objects={'VNet': VNet})
        network.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

        return network

    @staticmethod
    def get_dimsize_from_mhd(file_path):
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("DimSize"):
                    _, value = line.split("=")
                    dims = tuple(map(int, value.strip().split()))
                    return dims
        raise ValueError(f"No DimSize found in {file_path}")

    @staticmethod
    def dim_sizes_to_dhw_config(whd_tuple):
        whd_list = list(whd_tuple)
        dhw_list = [whd_list[2], whd_list[1], whd_list[0]]
        dhw_tuple = tuple(dhw_list)

        return dhw_tuple

    @staticmethod
    def read_dimension(read_in_dir):
        dim_sizes = []

        for mhd_file in read_in_dir.glob("*.mhd"):
            dims = Helper.get_dimsize_from_mhd(mhd_file)
            dim_sizes.append(dims)

        if not dim_sizes:
            raise ValueError("No .mhd files found")

        first = dim_sizes[0]

        for file_dims in dim_sizes:
            if file_dims != first:
                raise ValueError(
                    f"Inconsistent DimSize found: {first} vs {file_dims}"
                )

        dhw = Helper.dim_sizes_to_dhw_config(first)
        return dhw

    @staticmethod
    def count_patches(read_in_dir, patch_size):
        dimensions = Helper.read_dimension(read_in_dir)
        patch_size_tuple = tuple(patch_size)
        num_patches = tuple(d // p for d, p in zip(dimensions, patch_size_tuple))
        total_patches = math.prod(num_patches)
        return list(num_patches), total_patches

    @staticmethod
    def set_seed(seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)