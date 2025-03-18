import cv2
import numpy as np
import apriltag
import math
from rect import InterestingRect
from tracker import Tracker
import yaml
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a YAML configuration file.")
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.yaml")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            print(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' does not exist.")
        exit(1)

    Tracker(config).run()