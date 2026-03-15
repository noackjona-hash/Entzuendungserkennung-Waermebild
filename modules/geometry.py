import cv2
import numpy as np
import os
import joblib

# --- KI MODELL LADEN ---
MODEL_PATH = os.path.join("dataset", "ignite_ai_model.pkl")
try:
    if os.path.exists(MODEL_PATH):
        ai_model = joblib.load(MODEL_PATH)
        print("Erfolg: KI-Modell wurde in den Arbeitsspeicher geladen.")
    else:
        ai_model = None
except Exception as e:
    print(f"Fehler beim Laden des KI-Modells: {e}")
    ai_model = None

def find_both_feet(gray_img):
    """Findet die Konturen der beiden Füße im Wärmebild."""
    blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return None, None, None
        
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    sorted_by_x = sorted(sorted_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    return sorted_by_x[0], sorted_by_x[1], mask

def extract_toes_with_ai(contour, gray_img, foot_mask):
    """Nutzt das trainierte Machine Learning Modell, um die Zehen zu finden."""
    if ai_model is None:
        return None # Fallback aktivieren, wenn Modell fehlt

    # 1. Feature-Extraktion (Die Fuss-DNA berechnen)
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    features = list(hu) + [w / h] # 7 Hu-Moments + Seitenverhaeltnis

    # 2. KI Vorhersage (Gibt 10 relative Koordinatenwerte zurueck)
    # [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    prediction = ai_model.predict([features])[0]

    tips_with_data = []
    
    # 3. Vorhersagen in echte Pixel umrechnen und mit Deep Sensor scannen
    for i in range(5):
        rel_x = prediction[i * 2]
        rel_y = prediction[i * 2 + 1]
        
        # Relative KI-Vorhersage auf das absolute Bild uebertragen
        abs_x = int(x + (rel_x * w))
        abs_y = int(y + (rel_y * h))
        
        # Sicherstellen, dass die Punkte im Bild bleiben
        abs_x = max(0, min(gray_img.shape[1] - 1, abs_x))
        abs_y = max(0, min(gray_img.shape[0] - 1, abs_y))

        # 4. Der Deep Sensor: Lokale Hitze um den vorhergesagten Punkt finden
        roi_size = 20
        x_start, x_end = max(0, abs_x - roi_size), min(gray_img.shape[1], abs_x + roi_size)
        y_start, y_end = max(0, abs_y - roi_size), min(gray_img.shape[0], abs_y + roi_size)
        
        roi_gray = gray_img[y_start:y_end, x_start:x_end]
        roi_mask = foot_mask[y_start:y_end, x_start:x_end] 
        
        if roi_gray.size > 0:
            _, max_val, _, max_loc_roi = cv2.minMaxLoc(roi_gray, mask=roi_mask)
            temp = int(max_val)
            meas_pt = (x_start + max_loc_roi[0], y_start + max_loc_roi[1])
        else:
            temp, meas_pt = int(gray_img[abs_y, abs_x]), (abs_x, abs_y)
            
        tips_with_data.append({"tip": (abs_x, abs_y), "temp": temp, "sensor": meas_pt})
        
    # Die KI findet die Zehen manchmal in der Reihenfolge, in der sie trainiert wurde.
    # Wir sortieren sie zur Sicherheit streng von links nach rechts, damit die Zuordnung stimmt.
    tips_with_data = sorted(tips_with_data, key=lambda item: item["sensor"][0])
    
    return tips_with_data