import cv2
import numpy as np

def classical_extract(image_rgb, task="roads"):
    """Transparent heuristic baseline. Not a substitute for supervised segmentation."""
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if task == "roads":
        # Roads are often locally bright/low-saturation; adaptive threshold + morphology.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(l)
        mask = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, -3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    else:
        # Building baseline: local contrast segmentation and shape cleanup.
        blur = cv2.GaussianBlur(l, (5,5), 0)
        high = cv2.subtract(l, blur)
        _, mask = cv2.threshold(high, 12, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return mask
