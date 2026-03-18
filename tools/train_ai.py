import pandas as pd
import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ROBUSTE PFADE
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_DIR = os.path.join(BASE_DIR, "dataset", "augmented_images")
AUGMENTED_LABEL_FILE = os.path.join(BASE_DIR, "dataset", "labels_augmented.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "dataset", "ignite_ai_model.pkl")

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    for i in range(7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
            
    return hu_moments

def main():
    if not os.path.exists(AUGMENTED_LABEL_FILE):
        print(f"Fehler: {AUGMENTED_LABEL_FILE} nicht gefunden.")
        print("Bitte zuerst augment_data.py ausführen!")
        return

    print("Lade Datensatz und extrahiere Features...")
    df = pd.read_csv(AUGMENTED_LABEL_FILE)
    
    X, y = [], []
    
    for index, row in df.iterrows():
        img_path = os.path.join(AUGMENTED_DIR, row['filename'])
        
        # --- ROBUSTER BILD-IMPORT FÜR UMLAUTE ---
        try:
            with open(img_path, "rb") as f:
                img_bytes = bytearray(f.read())
            nparr = np.asarray(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            continue
            
        if img is None: continue
        
        features = extract_features(img)
        if features is not None:
            X.append(features)
            coords = [row[f'x{i}'] for i in range(10)] + [row[f'y{i}'] for i in range(10)]
            y.append(coords)

    if len(X) == 0:
        print("Fehler: Konnte keine Features extrahieren. Sind die Bilder im Ordner?")
        return

    X, y = np.array(X), np.array(y)
    print(f"Features extrahiert. Datensatz-Größe: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print("Trainiere Random Forest Regressor... Das kann einen Moment dauern.")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Training abgeschlossen! MSE: {mse:.2f}")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Modell gespeichert unter {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()