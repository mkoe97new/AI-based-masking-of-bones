import argparse
import json
import sys

from Testing import Testing
from Training import Training
from ImageGenerator import ImageGenerator

def get_parser():
    parser = argparse.ArgumentParser(description='Implementation for segmentation.')

    parser.add_argument(
        '-p', '--phase',
        dest='phase',
        help='Training phase (default=train)',
        choices=['train', 'test', 'generate_images'],
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
    if args.phase in ['train']:
        if not args.config_json:
            parser.error('--config_json is required for phase train')
        if not args.generator_config_json:
            parser.error('--generator_config_json is required for phase train')

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
        training.train()
    elif args.phase == "test":
        with open(args.config_json) as config_json:
            config = json.load(config_json)
        testing = Testing(config, generator_config_json)
        testing.test()
    elif args.phase == "generate_images":
        image_generator.generate_images()
        image_generator.train_test_split()
        #image_generator.check_if_zero()
    else:
        sys.exit("Invalid training phase")


if __name__ == "__main__":
    args = get_parser()
    main(args)
