import cv2
import numpy as np
import apriltag
import math


def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    
    def angle_from_center(point):
        return math.atan2(point[1] - center[1], point[0] - center[0])
    
    sorted_points = sorted(points, key=angle_from_center)
    return np.array(sorted_points)


def cut_and_warp_trapezoid(image, points, pose):
    pts = np.array(points, dtype=np.int32)
    pts = sort_points_clockwise(pts)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    trapezoid = cv2.bitwise_and(image, mask)

    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    
    center = np.array([w/2, h/2])
    cropped_trapezoid = trapezoid[y:y+h, x:x+w]
    
    if False:  # revert the rotation only
        rotation_part = pose[:2, :2]
        translation_part = pose[:2, 3]  # atually irrelevant
        initial_transformation_matrix = np.hstack([rotation_part.T, translation_part.reshape(2, 1)])
        
        # rotate around center = move to the upper left, rotate then move back
        trans_center = np.array([
            [1, 0, -center[1]],
            [0, 1, -center[0]]
        ], dtype=np.float32)

        reverse_trans_center = np.array([
            [1, 0, center[1]],
            [0, 1, center[0]]
        ], dtype=np.float32)

        # Combine the transformations
        transformation_matrix = reverse_trans_center @ np.vstack([initial_transformation_matrix, [0, 0, 1]]) @ np.vstack([trans_center, [0, 0, 1]])
        final_transformation_matrix = transformation_matrix[:2, :]
        if cropped_trapezoid.shape[0] > 0 and cropped_trapezoid.shape[1] > 0:
            cropped_trapezoid = cv2.warpAffine(cropped_trapezoid, transformation_matrix, (w, h))
    else: # revert complete pose
        original_points = points.astype(np.float32)
        w = max(w, h)
        h = w
        transformed_points = np.float32([
            [0, 0],
            [0, h],
            [w, h],
            [w, 0],
        ])

        center = np.array([w/2, h/2])

        M = cv2.getPerspectiveTransform(transformed_points, original_points)
        M_inv = np.linalg.inv(M)

        if cropped_trapezoid.shape[0] > 0 and cropped_trapezoid.shape[1] > 0:
            cropped_trapezoid = cv2.warpPerspective(image, M_inv, (w, h))
    return cropped_trapezoid


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    tag_points = np.array([
        [-0.1, -0.1, 0],
        [-0.1, 0.1, 0],
        [0.1, 0.1, 0],
        [0.1, -0.1, 0],
        #[-0.14, 0.2, -0.23],
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
            
            small = cut_and_warp_trapezoid(frame, image_points, pose)
            if small.shape[0] > 0 and small.shape[1] > 0:
                cv2.imshow('rect only', small)
        cv2.imshow('Labeled detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()