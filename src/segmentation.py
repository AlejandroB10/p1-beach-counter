import cv2
import numpy as np

def watershed_segmentation(original, thresholded):
    dist = cv2.distanceTransform(thresholded, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(thresholded, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original, markers)
    return markers, sure_fg, sure_bg, dist
