import cv2
import numpy as np
import apriltag
import math


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    tag_points = np.array([
        [0.1, 0.1, 0],
        [-0.1, -0.1, 0],
        [-0.1, 0.1, 0],
        [0.1, -0.1, 0],
    ])
    
    # Initialize the AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    
    # do stuff
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # camera stuff
        fx, fy = (600, 600)
        cx, cy = (gray.shape[1] / 2, gray.shape[0] / 2)
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        tag_size = 0.2

        detections = detector.detect(gray)

        if detections:
            detection = detections[0]  # we only need one marker, so do weird stuff if there are more

            # Draw the bounding box around the detected AprilTag
            pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            pose, _, _ = detector.detection_pose(detection, (fx, fy, cx, cy), tag_size)
            # homography = detection.homography

            
            h_tag_points = np.hstack((tag_points, np.ones((tag_points.shape[0], 1))))  # convert in homogeneous coords
            posed_points = pose @ h_tag_points.T  # apply pose
            camera_points = posed_points.T[:, :-1]  # convert back into 3d points
            h_image_points = camera_matrix @ camera_points.T  # apply camera matrix
            image_points = h_image_points.T[:, :2] / h_image_points.T[:, 2:3]  # normalize
            
            # draw
            for point in image_points:
                cv2.circle(frame, point[:2].astype(np.int32), 5, (0, 0, 255), -1)

        cv2.imshow('Labeled detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()