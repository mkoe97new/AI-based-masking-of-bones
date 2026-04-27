import argparse
import json
import sys
import os

from Slicing import Slicing

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from Testing import Testing
from Training import Training
from ImageGenerator import ImageGenerator

def get_parser():
    parser = argparse.ArgumentParser(description='Implementation for segmentation.')

    parser.add_argument(
        '-p', '--phase',
        dest='phase',
        help='Training phase (default=train)',
        choices=['train', 'test', 'generate', 'generate_2d_slices'],
        default='train')

    parser.add_argument(
        '--config_json',
        dest='config_json',
        help='JSON file for model configuration',
        type=str,
        default='./configs/main_config.json')

    parser.add_argument(
        '--generator_config_json',
        dest='generator_config_json',
        help='JSON file for generator configuration',
        type=str,
        default='./configs/generator.json')

    args = parser.parse_args()

    # Require both configs for train / test / predict
    if args.phase in ['train', 'test']:
        if not args.config_json:
            parser.error('--config_json is required for phase train and test')
        if not args.generator_config_json:
            parser.error('--generator_config_json is required for phase train and test')

    return args


def main(args):
    # read configuration file
    with open(args.generator_config_json) as generator_config_json:
        generator_config_json = json.load(generator_config_json)
    image_generator = ImageGenerator(generator_config_json)

    if args.phase == "train":
        with open(args.config_json) as config_json:
            config = json.load(config_json)
        training = Training(config, generator_config_json)
        if training.method == "generate_random":
            training.random_patch_train()
        else:
            training.train()
    elif args.phase == "test":
        with open(args.config_json) as config_json:
            config = json.load(config_json)
        testing = Testing(config, generator_config_json)
        if testing.random:
            testing.random_patch_test()
        else:
            testing.test()
    elif args.phase == "generate":
        if image_generator.method == "generate":
            image_generator.generate_images()
            image_generator.train_test_split()
        elif image_generator.method == "generate_random":
            image_generator.generate_random_images()
        elif image_generator.method == "sizes":
            image_generator.show_image_sizes()
    elif args.phase == "generate_2d_slices":
        with open(args.config_json) as config_json:
            config = json.load(config_json)
        slicing = Slicing(config, generator_config_json)
        slicing.generate_2d_slices()
    else:
        sys.exit("Invalid training phase")


if __name__ == "__main__":
    args = get_parser()
    main(args)
