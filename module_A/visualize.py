import cv2
import numpy as np
import matplotlib.pyplot as plt

# Histogram comparison
def plot_histogram_before_after(img_before, img_after):
    lab_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2LAB)
    lab_after  = cv2.cvtColor(img_after,  cv2.COLOR_BGR2LAB)

    L_before = lab_before[:, :, 0]
    L_after  = lab_after[:, :, 0]

    plt.figure(figsize=(10, 4))
    plt.hist(L_before.flatten(), bins=256, alpha=0.6, label="Before")
    plt.hist(L_after.flatten(),  bins=256, alpha=0.6, label="After")
    plt.title("L-channel Histogram Comparison")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Visualize LAB channels
def visualize_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    titles = ["L (Lightness)", "A (Green–Red)", "B (Blue–Yellow)"]

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(lab[:, :, i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Full pipeline visualization
def visualize_pipeline(img, analysis: dict, decision: dict, output):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"Output ({decision.get('method')})")
    plt.axis("off")

    plt.suptitle(
        f"Brightness: {analysis['brightness_mean']:.1f}, "
        f"Contrast: {analysis['contrast']:.1f}, "
        f"a*: {analysis['a_mean']:.1f}, b*: {analysis['b_mean']:.1f}"
    )
    plt.tight_layout()
    plt.show()
