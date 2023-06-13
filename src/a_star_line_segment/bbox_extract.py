import json
import numpy as np
import cv2

from .astar import get_background_color
from .image_cleaner import *

def get_bounding_boxes(img):
    raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert raw_img is not None
    background_mean = get_background_color(raw_img)

    raw_top_img, raw_bot_img = split_page(raw_img)
    assert raw_top_img is not None
    assert raw_bot_img is not None
    top_img, top_img_border, top_thicc = deskew(raw_top_img, background_mean)
    
    bot_img, bot_img_border, bot_thicc= deskew(raw_bot_img, background_mean)

    top_img = remove_border(top_img, top_img_border, top_thicc, background_mean)
    bot_img = remove_border(bot_img, bot_img_border, bot_thicc, background_mean)

    top_bbox = segment_img(top_img)
    bot_bbox = segment_img(bot_img)
    
    line_imgs = []
    for bbox in top_bbox:
        line_imgs.append(top_img[bbox[1]:bbox[3],bbox[0]:bbox[2]])
    for bbox in bot_bbox:
        line_imgs.append(bot_img[bbox[1]:bbox[3],bbox[0]:bbox[2]])
    return line_imgs
