import cv2
import numpy as np

def hsv_hue_subtract(img, bg):
    img_h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_h = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)

    h_sub = img_h[:, :, 0] - bg_h[:, :, 0]

    merged = cv2.merge([h_sub, img_h[:, :, 1], img_h[:, :, 2]])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def threshold_images(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        7,
        5
    )

    return otsu, adaptive
