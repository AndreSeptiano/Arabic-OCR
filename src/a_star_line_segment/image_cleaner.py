import cv2
import math
import numpy as np

from heapq import *
from .astar import *

from skimage.transform import rotate
from scipy.ndimage import binary_dilation, convolve1d

def split_page(img):
    height, width = img.shape[:2]
    center_x = width // 2
    crop_size = width // 4

    binarized_image = binarize_image(img)[:, center_x - crop_size : center_x + crop_size]
    vpp = vertical_projections(binarized_image)
    cut_point = center_x - crop_size + np.argmin(vpp)

    return img[:, :cut_point], img[:, cut_point:]

def rotate(image: np.ndarray, angle: float, background) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def deskew(img, background_mean):
    binarized_image = binarize_image(img)
    hpp = horizontal_projections(binarized_image)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binarized_image.astype(np.uint8), connectivity=8)
    areas = np.array(sorted([[i+1, stats[i+1][-1]] for i in range(len(stats[1:]))], key=lambda x: x[1]))
    outlier_labels = [elem[0] for elem in get_upper_outliers(areas)]

    mask = np.zeros_like(img)
    for label in outlier_labels:
        mask += np.where(labels == label, 1, 0).astype(np.uint8)

    center_x, center_y = img.shape[1]//2, img.shape[0]//2
    border_thicc = 0
    first_color = None
    for pix in reversed(range(center_x)):
        if mask[center_y, pix] > 0 and not first_color:
            first_color = (center_y, pix)
        if first_color and mask[center_y, pix] == 0:
            border_thicc = first_color[1] - pix; break

    bound = None
    border_label = None
    for label in outlier_labels:
        if labels[first_color[0]][first_color[1]] == label:
            border_label = label
            bound = stats[label]; break
    shf_top, shf_bot, shf_left, shf_right = bound[1], bound[1]+bound[3], bound[0], bound[0]+bound[2]
    focus_img = img[shf_top:shf_bot, shf_left:shf_right]

    border_mask = np.where(labels == border_label, 1, 0).astype(np.uint8)[shf_top:shf_bot, shf_left:shf_right]
    predicted_angle = 0
    highest_hp = 0

    for index,angle in enumerate(range(-20,20,2)):
        angle /= 10

    rotated = rotate(border_mask, angle, 0)
    hp = horizontal_projections(rotated)

    border_sum = np.sum(hp[np.argmax(hp)-int(border_thicc/2):np.argmax(hp)+int(border_thicc/2)])
    if highest_hp < border_sum:
        predicted_angle = angle
        highest_hp = border_sum

    return rotate(focus_img, predicted_angle, background_mean), rotate(border_mask, predicted_angle, 0), border_thicc

def remove_border(img, border_mask, border_thicc, background_mean):
    hpp = horizontal_projections(border_mask)
    h_threshold = np.max(hpp)/2
    vpp = vertical_projections(border_mask)
    v_threshold = np.max(vpp)/2

    v_border_mask = vpp > v_threshold
    v_border_mask = np.reshape(v_border_mask, (1, -1))

    h_border_mask = hpp > h_threshold
    h_border_mask = np.reshape(h_border_mask, (-1, 1))

    border_mask2 = np.logical_or(h_border_mask, v_border_mask)
    structure = np.ones((2 * border_thicc + 1, 2 * border_thicc + 1))
    border_mask2 = binary_dilation(border_mask2, structure=structure)

    img[border_mask2 > 0] = background_mean
    return img

def segment_img(img):
    binarized_image = binarize_image(img)
    hpp = horizontal_projections(binarized_image) 
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image.astype(np.uint8), connectivity=8)
    areas = stats[1:,-1]
    thres = get_upper_outliers_thres(areas)
    filtered_stats = stats[stats[:,-1] > thres][1:]

    for label in range(1, np.max(labels) + 1):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < thres:
            labels[labels == label] = 0
    diac_mask = (labels > 0).astype(np.uint8)  # Convert labels to binary mask

    diac_hpp = horizontal_projections(diac_mask)
    line_thicc = int(np.average(filtered_stats[:, -2]))
    kernel = np.ones(int(line_thicc/2)) / int(line_thicc/2)
    diac_hpp = convolve1d(diac_hpp, kernel, mode='reflect')

    peaks = find_peak_regions(diac_hpp, line_thicc)
    peaks_indexes = np.array(peaks)[:, 0].astype(int)

    diff = np.diff(peaks_indexes)
    indices = np.where(diff != 1)[0] + 1
    peak_groups = np.split(peaks_indexes, indices)
    # remove very small regions, these are basically errors in algorithm because of our threshold value
    peak_groups = [item for item in peak_groups if len(item) > 2 and np.min(item) >= line_thicc/2 and img.shape[0] - np.max(item) >= line_thicc/2]

    binary_image = get_binary(img)
    segment_separating_lines = []
    for i, sub_image_index in enumerate(peak_groups):
        nmap = binary_image[sub_image_index[0]:sub_image_index[-1], :]
        start_y = np.argmin(horizontal_projections(nmap))
        path = np.array(astar(nmap, get_blocks(nmap), (start_y, 0), (start_y, nmap.shape[1]-1)))
        offset_from_top = sub_image_index[0]
        path[:,0] += offset_from_top
        segment_separating_lines.append(path)
    segment_separating_lines.insert(0, np.array([[0, i] for i in range(img.shape[1])]))
    segment_separating_lines.append(np.array([[img.shape[0], i] for i in range(img.shape[1])]))

    bbox = []
    for index, line_segments in enumerate(segment_separating_lines):
        if index < len(segment_separating_lines)-1:
            lower_line = np.min(segment_separating_lines[index][:,0])
            upper_line = np.max(segment_separating_lines[index+1][:,0])

            bbox.append((0, upper_line, img.shape[1], lower_line))

    return tuple(bbox)
