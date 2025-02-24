import cv2
import numpy as np
import apriltag
import math

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize the AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    while cap.isOpened():
        ret, frame = cap.read()
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

                # Pose estimation
                camera_matrix = np.array([[600, 0, gray.shape[1] / 2],
                                          [0, 600, gray.shape[0] / 2],
                                          [0, 0, 1]])
                dist_coeffs = np.zeros((4, 1))
                tag_size = 0.165  # Size of the AprilTag (meters, adjust this to your tag size)

                retval, rvec, tvec = cv2.solvePnP(detection.corners_3d, detection.corners, camera_matrix, dist_coeffs)

                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                eulerAngles = rotation_matrix_to_euler_angles(R)

                # Calculate inverse rotation matrix
                R_inv = np.linalg.inv(R)

                # Apply the inverse rotation to the frame
                rows, cols = frame.shape[:2]
                M_inv = R_inv[:2, :2]
                dst = cv2.warpAffine(frame, M_inv, (cols, rows))

                # Display translation and rotation
                print(f'Translation (tvec): {tvec.flatten()}')
                print(f'Rotation (Euler Angles): {eulerAngles}')

                # Display the resulting frame with rotation correction
                cv2.imshow('AprilTag Detection with Correction', dst)

        # Display the original frame
        cv2.imshow('AprilTag Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()