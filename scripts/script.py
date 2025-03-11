import cv2
import numpy as np
import apriltag
import math
from rect import InterestingRect
import yaml
import os


class Tracker:
    def __init__(self, config):
        rectangles = config["rectangles"]
        self.rects = [InterestingRect(rect["name"], *np.array(rect["points"], dtype=np.float32())) for rect in rectangles]

        variables = config["variables"]
        self.tag_size = variables["tag_size"]
        self.fx, self.fy = variables["focal_length"]
        self.tag_id = variables["tag_id"]
        self.tag_family = variables["tag_family"]

    def get_camera_matrix(self, fx, fy, cx, cy):
        camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        return camera_matrix

    def calculate_image_points(self, rects, pose, camera_matrix):
        combined = np.vstack([rect.config_points for rect in rects])
        split_indices = np.cumsum([rect.config_points.shape[0] for rect in rects])[:-1]

        h_tag_points = np.hstack((combined, np.ones((combined.shape[0], 1))))  # convert in homogeneous coords
        posed_points = pose @ h_tag_points.T  # apply pose
        camera_points = posed_points.T[:, :-1]  # convert back into 3d points
        h_image_points = camera_matrix @ camera_points.T  # apply camera matrix
        image_points = h_image_points.T[:, :2] / h_image_points.T[:, 2:3]  # normalize
        
        restored_arrays = np.split(image_points, split_indices, axis=0)
        for i, points in enumerate(restored_arrays):
            rects[i].set_camera_points(*points)
        return rects
    
    def run(self):
        cap = cv2.VideoCapture(0)

        # Initialize the AprilTag detector
        options = apriltag.DetectorOptions(families=self.tag_family)
        detector = apriltag.Detector(options)
        
        # do stuff
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # camera stuff
            cx, cy = (gray.shape[1] / 2, gray.shape[0] / 2)
            camera_matrix = self.get_camera_matrix(self.fx, self.fy, cx, cy)

            detections = detector.detect(gray)

            if detections:
                detection = None

                # only do sth if the right tag id is being detected
                # weird shit happens if there are more than one of this tag id visible
                for d in detections:
                    if d.tag_id == self.tag_id:
                        detection = d
                        break
                
                if detection:
                    # Draw the bounding box around the detected AprilTag
                    pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                    # do stuff for each area of interest
                    pose, _, _ = detector.detection_pose(detection, (self.fx, self.fy, cx, cy), self.tag_size)
                    image_rects = self.calculate_image_points(self.rects, pose, camera_matrix)
                    for ir in image_rects:
                        img = ir.get_warped_image(frame, 300)
                        if img is not None:
                            cv2.imshow(ir.name, img)
                
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.yaml")
    with open(file, "r") as file:
        config = yaml.safe_load(file)
        
    Tracker(config).run()