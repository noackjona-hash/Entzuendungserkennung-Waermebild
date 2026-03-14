import cv2
import os

def load_and_preprocess(image_path):
    if not os.path.exists(image_path):
        return None, None
        
    img = cv2.imread(image_path)
    # Bild etwas verkleinern, damit es gut in die GUI passt (max Hoehe 600px)
    h, w = img.shape[:2]
    if h > 600:
        ratio = 600 / h
        img = cv2.resize(img, (int(w * ratio), 600))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray