import cv2
import numpy as np
import pandas as pd

from src.io_utils import load_and_crop
from src.preprocessing import hsv_hue_subtract, threshold_images
from src.segmentation import watershed_segmentation
from src.counting import count_segments, assign_components_to_df
from src.visualization import show_image, show_watershed_result

# 1. Load and preprocess images
image_path = r'E:\\UIB\\Fall\\Image and Video Analysis\\Project1\\data\\'
img_original = load_and_crop(image_path + "1660392000.jpg")
bg_img = load_and_crop(image_path + "1660370400.jpg")

subtracted = hsv_hue_subtract(img_original, bg_img)
show_image(cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB), "Hue Subtraction Result")

gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
otsu, adaptive = threshold_images(gray)

show_image(otsu, "Otsu Threshold")
show_image(adaptive, "Adaptive Threshold")

thresholded = 255 - adaptive  # Invert for watershed

# 2. Watershed segmentation
markers, sure_fg, sure_bg, dist = watershed_segmentation(img_original, thresholded)

show_watershed_result(img_original, markers)
show_image(sure_fg, "Sure Foreground")
show_image(dist, "Distance Transform")

# 3. Component analysis
num_segments = count_segments(markers)
print("Number of segments:", num_segments)

df = pd.read_csv("first.csv", names=['label','x','y','image_name','width','height'])
df = assign_components_to_df(markers, df)

unique_components = df[df['component_id'] > 0]['component_id'].unique()
print("Detected unique components in CSV:", len(unique_components))
