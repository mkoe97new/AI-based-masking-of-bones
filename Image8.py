import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


class Image:
    width = 0
    height = 0
    depth = 0
    spacing = None
    voxel_data = None
    image_number = None
    sitk_image = None

    def constructor_image(self, metadata_path, index):

        image = sitk.ReadImage(metadata_path)
        self.sitk_image = image
        self.width = image.GetWidth()
        self.height = image.GetHeight()
        self.depth = image.GetDepth()
        self.spacing = image.GetSpacing()

        self.voxel_data = sitk.GetArrayFromImage(image)
        self.image_number = index

    def constructor_original_image(self, image_label):
        self.width = image_label.width
        self.height = image_label.height
        self.depth = image_label.depth
        self.spacing = image_label.spacing
        self.image_number = image_label.image_number
        self.sitk_image = image_label.sitk_image
        self.voxel_data = image_label.voxel_data

    def empty_image_constructor(self, image_size, downsampling_factor, index):
        self.voxel_data = np.zeros((image_size[0], image_size[1], image_size[2]))
        self.width = image_size[2] / downsampling_factor
        self.height = image_size[1] / downsampling_factor
        self.depth = image_size[0] / downsampling_factor
        self.image_number = index

        if downsampling_factor > 1:
            self.voxel_data = self.downsapling_new(downsampling_factor, 0)

    def generate_image_label(self):
        modified_image = np.select(
            [
                self.voxel_data == 0,  # Condition 1
                np.isin(self.voxel_data, [1, 2]),  # Condition 2
                np.isin(self.voxel_data, [3, 4])  # Condition 3
            ],
            [
                0,  # Value for Condition 1
                1,  # Value for Condition 2
                2  # Value for Condition 3
            ]
        )
        self.voxel_data = modified_image

    def generate_image_original(self):
        modified_image = np.select(
            [
                np.isin(self.voxel_data, [0, 1, 3]),  # Condition 2
                np.isin(self.voxel_data, [2, 4])  # Condition 3
            ],
            [
                0,  # Value for Condition 1
                1,  # Value for Condition 2
            ]
        )
        #modified_image = np.where(np.isin(image.get_voxel_data(), [0, 1, 3]), 0, 1)
        self.voxel_data = modified_image

    def save_image(self, output_path, original):
        if original:
            raw_file_path = output_path / ('original_img_' + str(self.image_number) + '.raw')
            mhd_file_path = output_path / ('original_img_' + str(self.image_number) + '.mhd')

            self.voxel_data = self.voxel_data.astype(np.int8)
            self.voxel_data.astype(np.int8).tofile(raw_file_path)
            image = sitk.GetImageFromArray(self.voxel_data)

            sitk.WriteImage(image, mhd_file_path)
        else:
            raw_file_path = output_path / ('label_img_' + str(self.image_number) + '.raw')
            mhd_file_path = output_path / ('label_img_' + str(self.image_number) + '.mhd')

            self.voxel_data = self.voxel_data.astype(np.int8)
            self.voxel_data.astype(np.int8).tofile(raw_file_path)
            image = sitk.GetImageFromArray(self.voxel_data)

            sitk.WriteImage(image, mhd_file_path)

    def get_roi(self, start_and_end):
        image_roi = self.voxel_data[start_and_end[0]:start_and_end[1], start_and_end[2]:start_and_end[3], start_and_end[4]:start_and_end[5]]
        return image_roi

    def set_roi(self, patch, patch_position, patch_size):
        z0 = int(patch_position[0]) * patch_size[0]
        y0 = int(patch_position[1]) * patch_size[1]
        x0 = int(patch_position[2]) * patch_size[2]

        self.voxel_data[
            z0:z0 + patch_size[0],
            y0:y0 + patch_size[1],
            x0:x0 + patch_size[2]
            ] = patch

    def resize_image(self, resize_to, downsampling_factor, order_name):
        self.voxel_data = self.voxel_data[
          :resize_to[0], :resize_to[1], :resize_to[2]
          ]
        self.depth, self.height, self.width = self.voxel_data.shape

        pad_d = resize_to[0] - self.depth
        pad_h = resize_to[1] - self.height
        pad_w = resize_to[2] - self.width

        padded = np.pad(
            self.voxel_data,
            pad_width=(
                (0, pad_d),  # depth
                (0, pad_h),  # height
                (0, pad_w),  # width
            ),
            mode='constant',
            constant_values=0
        )

        self.voxel_data = padded.copy()
        self.depth = resize_to[0]
        self.height = resize_to[1]
        self.width = resize_to[2]
        order = self.order_mapping(order_name)

        if downsampling_factor > 1:
            self.depth = resize_to[0] / downsampling_factor
            self.height = resize_to[1] / downsampling_factor
            self.width = resize_to[2] / downsampling_factor
            self.voxel_data = self.downsapling(downsampling_factor, order)

    def downsapling(self, factor=2, order=1):
            zoom_factors = (1.0 / factor, 1.0 / factor, 1.0 / factor)
            return zoom(self.voxel_data, zoom=zoom_factors, order=order)

    def order_mapping(self, order_name):
        if order_name == "NN":
            order = 0
        elif order_name == "Interpolation":
            order = 1
        else:
            order = 1

        return order

    def get_bounding_box(self):
        # Find all non-background voxels
        coords = np.argwhere(self.voxel_data > 0)

        if coords.size == 0:
            return None  # no object in image

        # coords are in (z, y, x) order
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        return (z_min, z_max), (y_min, y_max), (x_min, x_max)