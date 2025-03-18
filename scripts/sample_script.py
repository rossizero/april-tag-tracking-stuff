import cv2
import numpy as np
from tracker import Tracker
import yaml
import os
import argparse


if __name__ == '__main__':
    # read config file via args, defaulting in "sample.yaml"
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

    # start tracker and show corrected rects until user presses 'q'
    tracker = Tracker(config)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        result = tracker.get_apriltag_detection(frame=frame)

        if result:
            detection, pose = result
            # Draw the bounding box around the detected AprilTag
            pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            for ir, img in tracker.get_corrected_rects(frame=frame):
                cv2.imshow(ir.name, img)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
