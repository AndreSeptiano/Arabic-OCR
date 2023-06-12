import json
import numpy as np
import cv2

from image_utils import img_b64_to_arr
from a_star_line_segment import get_background_color
from image_cleaner import split_page, deskew, remove_border, segment_img

def get_bounding_boxes(image_path):
    with open(image_path, 'r') as json_file:
        data = json.load(json_file)
    
    raw_img = np.squeeze(img_b64_to_arr(data['imageData']))
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    background_mean = get_background_color(raw_img)

    raw_top_img, raw_bot_img = split_page(raw_img)
    top_img, top_img_border, top_thicc = deskew(raw_top_img, background_mean)
    bot_img, bot_img_border, bot_thicc= deskew(raw_bot_img, background_mean)

    top_img = remove_border(top_img, top_img_border, top_thicc, background_mean)
    bot_img = remove_border(bot_img, bot_img_border, bot_thicc, background_mean)

    top_bbox = segment_img(top_img)
    bot_bbox = segment_img(bot_img)
    
    return top_bbox, bot_bbox
