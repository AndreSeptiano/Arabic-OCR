import cv2
import numpy as np

from heapq import *
from skimage.filters import threshold_otsu

def trim_line(img, threshold=10):
    binary = binarize_image(img)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    
    left = np.min(stats[:,0])-10 if np.min(stats[:,0])-10 >= 0 else 0
    right = np.max(stats[:,0])+np.max(stats[:,2])+10 if np.max(stats[:,0])+np.max(stats[:,2])+10 <= binary.shape[1] else binary.shape[1]
    top = np.min(stats[:,1])-10 if np.min(stats[:,1])-10 >= 0 else 0
    bot = np.max(stats[:,1])+np.max(stats[:,3])+10 if np.max(stats[:,1])+np.max(stats[:,3])+10 <= binary.shape[0] else binary.shape[0]

    return img[top:bot, left:right]

def find_peak_regions(hpp, thickness):
    peaks = []
    for i in range(1, len(hpp)):
        hppv = hpp[i]
        if hppv == 0:
            continue

        start = i-int(thickness/2) if i-int(thickness/2) >= 0 else 0
        end = i+int(thickness/2) if i+int(thickness/2) < len(hpp) else len(hpp)

        if hppv >= np.max(hpp[start:i]) and hppv >= np.max(hpp[i:end]):
            peaks.extend(([j, hpp[j]] for j in range(start, end)))
        i = end

    peaks = [[i, hpp[i]] for i in range(hpp.shape[0]) if [i, hpp[i]] not in peaks]
    return peaks

def find_peak_regions2(hpp, thickness):
    inv_peaks = []
    for i in range(thickness, len(hpp)-thickness):
        hppv = hpp[i]

    is_peak = True
    for j in range(i-thickness, i+thickness):
        if hppv < hpp[j]:
            is_peak = False; break 

    if is_peak:
        inv_peaks.extend(([i, hpp[i]] for i in range(i-int(thickness/2), i+int(thickness/2))))

    peaks = [[i, hpp[i]] for i in range(hpp.shape[0]) if [i, hpp[i]] not in inv_peaks]

    return peaks

def find_bottom_regions(hpp, threshold):
    peaks = []
    for i, hppv in enumerate(hpp):
        if hppv > threshold:
            peaks.append([i, hppv])
    return peaks

# now that everything is cleaner, its time to segment all the lines using the A* algorithm
def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary

def get_background_color(raw_img):
    binarized_image = binarize_image(raw_img)
    # Perform connected component labeling
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binarized_image.astype(np.uint8), connectivity=8)
    # Calculate the average pixel value of the background
    background_mean = np.mean(raw_img[labels == 0])

    return background_mean

def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)

def vertical_projections(sobel_image):
    return np.sum(sobel_image, axis=0)

def binarize_image(image):
    threshold = threshold_otsu(image)
    return image < threshold

def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def get_blocks(nmap):
    # Perform connected component labeling
    _, labels, stats, _ = cv2.connectedComponentsWithStats(nmap.astype(np.uint8), connectivity=8)
    blocks = []

    for i in range(1, len(stats)):  # Skip the background component at index 0
        if stats[i][3] == nmap.shape[0]:
            blocks.append((stats[i][0], stats[i][0] + stats[i][2]))

    return sorted(blocks, key=lambda x: x[0])

def get_upper_outliers(data):
    first_col = data[:, 1]
    mean = np.mean(first_col)
    std = np.std(first_col)

  # Define the threshold for outliers (e.g., 2 standard deviations)
    threshold = mean + 2 * std

  # Filter the array based on the outlier condition
    upper_outliers = data[first_col > threshold]
    return upper_outliers

def get_upper_outliers_thres(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    return Q3 + 1.5 * IQR

def astar(array, blocks, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    block_ahead = (array.shape[1]+1, -1)

    if len(blocks) != 0:
        block_ahead = blocks.pop(0)

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]
        if current[1] > block_ahead[1] and len(blocks) != 0:
            block_ahead = blocks.pop(0)
        elif current[1] > block_ahead[1] and len(blocks) == 0:
            block_ahead = (array.shape[1]+1, -1)

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1 and not (block_ahead[0] <= neighbor[1] <= block_ahead[1]):
                        continue
                else:
                    continue # array bound y walls
            else:
                continue # array bound x walls
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return []
