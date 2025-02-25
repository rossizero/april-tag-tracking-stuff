import cv2
import numpy as np
import apriltag
import math


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    # Initialize the AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    while cap.isOpened():
        ret, frame = cap.read()
        #dst = frame
        cv2.imshow('AprilTag Detection', frame)
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        if detections:
            for detection in detections:
                # Draw the bounding box around the detected AprilTag
                pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                pose, _, _ = detector.detection_pose(detection, (600, 600, gray.shape[1] / 2, gray.shape[0] / 2), 0.2)
                print("pose", pose)

                # Apply the inverse rotation to the frame
                rows, cols = frame.shape[:2]
                if np.linalg.det(detection.homography) == 0:
                    print("Homography matrix is not invertible.")

                b = np.linalg.inv(detection.homography)
                homography_inv = np.eye(3)
                homography_inv[:2, :2] = b[:2, :2]
                dst = cv2.warpPerspective(frame, homography_inv, (cols, rows), flags=cv2.INTER_LINEAR)
                fx, fy = (600, 600)
                cx, cy = (gray.shape[1] / 2, gray.shape[0] / 2)
                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                
                tag_point = np.array([0, 0.1, 0, 1])

                # Apply the homography matrix to the point
                #transformed_point = detection.homography @ tag_point
                # Normalize the transformed point
                #transformed_point /= transformed_point[2]
                # Extract the x and y coordinates
                #image_point = transformed_point[:2].astype(np.int32)
                print(pose.shape)
                h_point = pose @ tag_point
                camera_point = h_point[:3]
                image_point_homogeneous = camera_matrix @ camera_point
                image_point = image_point_homogeneous[:2] / image_point_homogeneous[2]
                print(image_point)
                cv2.circle(frame, image_point[:2].astype(np.int32), 5, (0,0,255), -1)
        
            #cv2.imshow('Warped Frame', dst)
        # Display the original frame
        cv2.imshow('AprilTag Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()