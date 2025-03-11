import math
import numpy as np
import cv2


class InterestingRect:
    def __init__(self, name: str, p1: np.array, p2: np.array, p3: np.array, p4: np.array):
        self.config_points = np.array([
            p1, p2, p3, p4
        ])
        self.points = self.config_points.copy()
        self.name = name
    
    def set_camera_points(self, p1: np.array, p2: np.array, p3: np.array, p4: np.array):
        self.points = np.array([
            p1, p2, p3, p4
        ])

    def __get_sorted_points(self):
        center = np.mean(self.points, axis=0)
        
        def angle_from_center(point):
            return math.atan2(point[1] - center[1], point[0] - center[0])
        
        sorted_points = sorted(self.points, key=angle_from_center)
        return np.array(sorted_points)

    def __get_trapezoid(self, image) -> np.array:
        # need to sort points, so that the polygon does not overlap with itself
        pts = self.__get_sorted_points().astype(np.int32)
        pts = pts[:, :2]
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        trapezoid = cv2.bitwise_and(image, mask)
        x, y, w, h = self.__get_bounding_rect()

        cropped = trapezoid[
            y : y + h, 
            x : x + w
            ]
        return cropped

    def __get_bounding_rect(self):
        rect = cv2.boundingRect(self.points.astype(np.int32))  # Tuple: x, y, w, h
        return rect
    
    def get_center(self, local=False):
        x, y, w, h = self.__get_bounding_rect()
        if local:
            return np.array([w/2, h/2])
        return np.array([x + w/2, y + h/2])

    def get_raw_image(self, frame) -> np.array:
        trapezoid = self.__get_trapezoid(frame)
        if trapezoid.shape[0] > 0 and trapezoid.shape[1] > 0:
            return trapezoid

    def get_warped_image(self, image, size:int = 0) -> np.array:
        """
        returns the perspective-corrected image
        """
        trapezoid = self.__get_trapezoid(image)

        original_points = self.points.astype(np.float32)
        _, _, width, height = self.__get_bounding_rect()
        config_points_mm = self.config_points[:, :2].copy() * 1000
        _, _, w, h = cv2.boundingRect(config_points_mm.astype(np.int32))
        width = size if size > 0 else width

        height = int(width * h/w)

        target_rect = np.float32([
            [0, 0],
            [0, height],
            [width, height],
            [width, 0],
        ])

        M = cv2.getPerspectiveTransform(target_rect, original_points)
        M_inv = np.linalg.inv(M)

        if trapezoid.shape[0] > 0 and trapezoid.shape[1] > 0:
            trapezoid = cv2.warpPerspective(image, M_inv, (width, height))
        
        if trapezoid.shape[0] > 0 and trapezoid.shape[1] > 0:
            return trapezoid
        
    def get_rotated_image(self, frame, pose) -> np.array:
        """
        returns the rotation-corrected image
        """
        trapezoid = self.__get_trapezoid(frame)

        rotation_part = pose[:2, :2]
        translation_part = pose[:2, 3]  # atually irrelevant
        initial_transformation_matrix = np.hstack([rotation_part.T, translation_part.reshape(2, 1)])
        
        center = self.get_center(local=True)

        # rotate around local center = move to the upper left, rotate then move back
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
        if trapezoid.shape[0] > 0 and trapezoid.shape[1] > 0:
            trapezoid = cv2.warpAffine(trapezoid, transformation_matrix, (w, h))
        
        return trapezoid
