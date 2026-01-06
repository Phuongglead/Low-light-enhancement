import cv2
import numpy as np

# Histogram Equalization (baseline)
def histogram_equalization(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    meta = {
        "method": "HE",
        "reason": "global contrast enhancement baseline"
    }
    return eq_bgr, meta


# CLAHE
def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )
    L_clahe = clahe.apply(L)

    lab_clahe = cv2.merge([L_clahe, A, B])
    out = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    meta = {
        "method": "CLAHE",
        "reason": "low contrast or uneven illumination"
    }
    return out, meta


# Gray World White Balance
def gray_world(img: np.ndarray):
    img = img.astype(np.float32)

    mean_b = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_r = np.mean(img[:, :, 2])
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    img[:, :, 0] *= mean_gray / (mean_b + 1e-6)
    img[:, :, 1] *= mean_gray / (mean_g + 1e-6)
    img[:, :, 2] *= mean_gray / (mean_r + 1e-6)

    img = np.clip(img, 0, 255).astype(np.uint8)

    meta = {
        "method": "GRAY_WORLD",
        "reason": "color cast detected in LAB space"
    }
    return img, meta


# Single / Multi-Scale Retinex
def retinex(img: np.ndarray, sigmas=(15, 80, 250)):
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)

    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log(img) - np.log(blur + 1e-6)

    retinex /= len(sigmas)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    retinex = retinex.astype(np.uint8)

    meta = {
        "method": "RETINEX",
        "reason": "illumination correction (bridge to DL)"
    }
    return retinex, meta
