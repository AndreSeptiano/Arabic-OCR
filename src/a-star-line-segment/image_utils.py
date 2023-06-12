import base64
import numpy as np
import io
import math
from PIL import Image

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

def crop_bbox(bbox, crop_region):
    x1, x2, y1, y2 = crop_region
    x_min, y_min, x_max, y_max = bbox

    # Adjust bounding box coordinates based on crop region
    adjusted_x_min = x_min - x1
    adjusted_y_min = y_min - y1
    adjusted_x_max = x_max - x1
    adjusted_y_max = y_max - y1

    # Add the adjusted bounding box coordinates to the list
    return (adjusted_x_min, adjusted_y_min, adjusted_x_max, adjusted_y_max)

def rotate_bbox(bbox, angle, image_shape):
    x_min, y_min, x_max, y_max = bbox
    old_height, old_width = image_shape[:2]
    angle_radian = math.radians(angle)
    
    # Calculate center coordinates of the image
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Adjust center coordinates for the rotation
    new_center_x = (center_x * np.cos(angle_radian)) + (center_y * np.sin(angle_radian))
    new_center_y = (center_y * np.cos(angle_radian)) - (center_x * np.sin(angle_radian))
    
    # Calculate new bounding box coordinates
    new_x_min = new_center_x - (x_max - x_min) / 2
    new_y_min = new_center_y - (y_max - y_min) / 2
    new_x_max = new_center_x + (x_max - x_min) / 2
    new_y_max = new_center_y + (y_max - y_min) / 2
    
    # Adjust bounding box coordinates to fit within the image boundaries
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(old_width - 1, new_x_max)
    new_y_max = min(old_height - 1, new_y_max)
    
    return (int(new_x_min), int(new_y_max), int(new_x_max), int(new_y_min))
