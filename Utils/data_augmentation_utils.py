import numpy as np
import pandas as pd
import math
import cv2
def apply_random_rotation(image, bbox, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    rad_angle = math.radians(angle)

    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    x_min, y_min, x_max, y_max = bbox
    box_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    points_ones = np.hstack([box_points, np.ones((4, 1))])
    rotated_points = rotation_matrix.dot(points_ones.T).T

    x_min_rot, y_min_rot = rotated_points[:, 0].min(), rotated_points[:, 1].min()
    x_max_rot, y_max_rot = rotated_points[:, 0].max(), rotated_points[:, 1].max()
    rotated_bbox = [x_min_rot, y_min_rot, x_max_rot, y_max_rot]

    return rotated_image, rotated_bbox


def apply_random_shearing(image, bbox, max_shear=15):
    shear_x = np.random.uniform(-math.radians(max_shear), math.radians(max_shear))
    shear_y = np.random.uniform(-math.radians(max_shear), math.radians(max_shear))
    shear_matrix = np.array([[1, math.tan(shear_x), 0], [math.tan(shear_y), 1, 0]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

    x_min, y_min, x_max, y_max = bbox
    box_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    points_ones = np.hstack([box_points, np.ones((4, 1))])
    sheared_points = shear_matrix.dot(points_ones.T).T

    x_min_shear, y_min_shear = sheared_points[:, 0].min(), sheared_points[:, 1].min()
    x_max_shear, y_max_shear = sheared_points[:, 0].max(), sheared_points[:, 1].max()
    sheared_bbox = [x_min_shear, y_min_shear, x_max_shear, y_max_shear]

    return sheared_image, sheared_bbox

