import cv2
import numpy as np
import joblib
import os

MODEL_PATH = "dataset/ignite_ai_model.pkl"
ai_model = None

def load_ai_model():
    global ai_model
    if ai_model is None and os.path.exists(MODEL_PATH):
        ai_model = joblib.load(MODEL_PATH)
    return ai_model

def find_both_feet(gray_img):
    """
    Segmentiert die Füße vom Hintergrund mittels Adaptive Otsu Thresholding.
    """
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return None, None, thresh
        
    # Die zwei größten Konturen sortiert nach Größe finden
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    # Sortiere sie von links nach rechts (X-Koordinate des Bounding Rectangles)
    sorted_lr = sorted(sorted_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    return sorted_lr[0], sorted_lr[1], thresh

def extract_toes_with_ai(foot_contour, gray_img, foot_mask):
    """
    Nutzt das trainierte Modell, um die Zehen vorherzusagen und sucht dann im Umkreis
    von 20 Pixeln nach dem exakten absoluten Hitzemaximum.
    """
    model = load_ai_model()
    if model is None:
        raise ValueError("KI Modell nicht gefunden. Bitte trainiere die KI zuerst!")
        
    # Hu-Momente der Kontur berechnen (exakt wie beim Training)
    moments = cv2.moments(foot_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    for i in range(7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
            
    # Inferenz! Die KI schätzt die Positionen.
    prediction = model.predict([hu_moments])[0]
    
    predicted_toes = []
    # Da die KI 10 X und 10 Y Werte zurückgibt (wir nehmen hier einfach an, 
    # dass das Modell robust genug ist, uns die Punkte für DEN Fuß zurückzugeben, 
    # der gerade übergeben wurde, indem wir die Punkte herausfiltern, die im Bounding Box liegen)
    x_coords, y_coords, w, h = cv2.boundingRect(foot_contour)
    
    # Um es hier für die Laufzeit sicher zu machen, durchsuchen wir die 10 KI Punkte
    # nach den 5, die tatsächlich in der Bounding-Box dieses Fußes liegen.
    valid_points = []
    for i in range(10):
        px = int(prediction[i])
        py = int(prediction[i+10])
        if x_coords <= px <= x_coords + w:
            valid_points.append((px, py))
            
    # Nehmen wir die ersten 5 gefundenen Punkte
    valid_points = valid_points[:5]
    
    # Deep Sensor Logik (Suche das Hitzemaximum im Radius)
    for px, py in valid_points:
        r = 20
        x_s, x_e = max(0, px-r), min(gray_img.shape[1], px+r)
        y_s, y_e = max(0, py-r), min(gray_img.shape[0], py+r)
        
        roi_gray = gray_img[y_s:y_e, x_s:x_e]
        roi_mask = foot_mask[y_s:y_e, x_s:x_e]
        
        if roi_gray.size > 0 and np.count_nonzero(roi_mask) > 0:
            _, max_val, _, max_loc = cv2.minMaxLoc(roi_gray, mask=roi_mask)
            real_x = x_s + max_loc[0]
            real_y = y_s + max_loc[1]
            predicted_toes.append({"sensor": (real_x, real_y), "temp": max_val})
        else:
            predicted_toes.append({"sensor": (px, py), "temp": float(gray_img[py, px]) if py<gray_img.shape[0] and px<gray_img.shape[1] else 0})
            
    return predicted_toes