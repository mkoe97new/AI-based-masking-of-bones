import numpy as np
import tensorflow as tf

from Image8 import Image


class Patching():
    def __init__(self, patch_size, read_in_dir):
        self.patch_size = patch_size
        self.read_in_dir = read_in_dir

    def calculate_roi(self, current_patch_position):
        depth_start = current_patch_position[0] * self.patch_size[0]
        height_start = current_patch_position[1] * self.patch_size[1]
        width_start = current_patch_position[2] * self.patch_size[2]
        depth_end = depth_start + self.patch_size[0]
        height_end = height_start + self.patch_size[1]
        width_end = width_start + self.patch_size[2]

        return [depth_start, depth_end, height_start, height_end, width_start, width_end]

    def create_patch_dataset(self, current_patch_number, current_patch_position, last_patch_position):
        if current_patch_number != 0:
            current_patch_position[0] += 1

            if current_patch_number % last_patch_position[0] == 0:
                current_patch_position[1] += 1
                current_patch_position[0] = 0
            if current_patch_number % last_patch_position[0] == 0 and current_patch_number % (last_patch_position[0] * last_patch_position[1]) == 0:
                current_patch_position[2] += 1
                current_patch_position[1] = 0
                current_patch_position[0] = 0

        start_and_end = self.calculate_roi(current_patch_position)
        original_images_list = []
        label_images_list = []
        for original_mhd in sorted(self.read_in_dir.glob("original_img*.mhd")):
            case_id = original_mhd.stem[len("original_img"):]
            label_mhd = self.read_in_dir / f"label_img{case_id}.mhd"

            image_original = Image()
            image_label = Image()

            image_original.constructor_image(original_mhd, case_id)
            image_label.constructor_image(label_mhd, case_id)
            image_original_roi = image_original.get_roi(start_and_end)
            image_label_roi = image_label.get_roi(start_and_end)

            image_original_roi = np.expand_dims(image_original_roi, axis=-1)
            image_label_roi = np.expand_dims(image_label_roi, axis=-1)

            original_images_list.append(image_original_roi)
            label_images_list.append(image_label_roi)

        original_images_list_np = np.array(original_images_list)
        label_images_list_np = np.array(label_images_list)

        original_images_list_np = original_images_list_np.astype(np.float32)
        label_images_list_np = label_images_list_np.astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((original_images_list_np, label_images_list_np))

        return dataset, current_patch_position

    def shuffle(self, dataset, seed, batch_size):
        dataset = (
            dataset
            .shuffle(buffer_size=1024, seed=seed)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return dataset