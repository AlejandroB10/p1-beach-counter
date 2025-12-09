import cv2
import matplotlib.pyplot as plt

def show_image(img, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_watershed_result(original, markers):
    result = original.copy()
    result[markers == -1] = [0, 0, 255]

    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Watershed Result")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.imshow(markers, cmap='jet')
    plt.title("Watershed Markers")
    plt.axis("off")
    plt.show()
