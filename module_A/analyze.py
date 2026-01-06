import cv2
import numpy as np

def analyze_image(img: np.ndarray):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    brightness_mean = float(np.mean(L))
    contrast  = float(np.std(L))

    a_mean = float(np.mean(A) - 128.0)
    b_mean = float(np.mean(B) - 128.0)

    L_blur = cv2.GaussianBlur(L, (5, 5), 0)
    noise_score = float(np.var(L.astype(np.float32) - L_blur.astype(np.float32)))

    analysis = {
        "brightness_mean": brightness_mean,
        "contrast": contrast,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "noise_score": noise_score,
    }

    return analysis
