import pandas as pd
import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

from modules.loader import load_and_preprocess
from modules.geometry import find_both_feet

def extract_features_and_targets(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    
    X_data = [] # Hier kommen die Fuss-Formen rein (Hu-Moments)
    Y_data = [] # Hier kommen die relativen Zehen-Koordinaten rein
    
    print(f"Lese {len(df)} Trainingsbilder ein...")
    
    for index, row in df.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        _, gray = load_and_preprocess(img_path)
        
        if gray is None:
            continue
            
        left_contour, right_contour, _ = find_both_feet(gray)
        
        if left_contour is not None and right_contour is not None:
            # --- LINKER FUSS ---
            x_l, y_l, w_l, h_l = cv2.boundingRect(left_contour)
            moments_l = cv2.moments(left_contour)
            hu_l = cv2.HuMoments(moments_l).flatten()
            
            # Features: Die 7 Hu-Moments + das Seitenverhältnis des Fusses
            features_l = list(hu_l) + [w_l / h_l]
            X_data.append(features_l)
            
            # Targets: Die relativen Koordinaten der 5 Zehen innerhalb der Bounding Box
            targets_l = []
            for i in range(1, 6):
                # Wissenschaftliche Normalisierung auf Werte zwischen 0.0 und 1.0
                rel_x = (row[f'L_Zeh{i}_x'] - x_l) / w_l
                rel_y = (row[f'L_Zeh{i}_y'] - y_l) / h_l
                targets_l.extend([rel_x, rel_y])
            Y_data.append(targets_l)
            
            # --- RECHTER FUSS ---
            x_r, y_r, w_r, h_r = cv2.boundingRect(right_contour)
            moments_r = cv2.moments(right_contour)
            hu_r = cv2.HuMoments(moments_r).flatten()
            
            features_r = list(hu_r) + [w_r / h_r]
            X_data.append(features_r)
            
            targets_r = []
            for i in range(1, 6):
                rel_x = (row[f'R_Zeh{i}_x'] - x_r) / w_r
                rel_y = (row[f'R_Zeh{i}_y'] - y_r) / h_r
                targets_r.extend([rel_x, rel_y])
            Y_data.append(targets_r)

    return np.array(X_data), np.array(Y_data)

if __name__ == "__main__":
    print("=== IGNITE AI TRAINER ===")
    dataset_dir = "dataset"
    
    # --- UPDATE: Nutze die neuen augmentierten Daten ---
    # CSV_FILE VON labels.csv AUF labels_augmented.csv GEAENDERT
    csv_file = os.path.join(dataset_dir, "labels_augmented.csv") 
    
    # IMG_DIR VON images AUF augmented_images GEAENDERT
    img_dir = os.path.join(dataset_dir, "augmented_images") 
    
    if not os.path.exists(csv_file):
        print(f"Fehler: Keine augmentierten Trainingsdaten ({csv_file}) gefunden!")
        exit()
        
    X, Y = extract_features_and_targets(csv_file, img_dir)
    
    if len(X) == 0:
        print("Fehler: Konnte keine auswertbaren Fuesse in den Bildern finden.")
        exit()
        
    print(f"Trainingsdaten erfolgreich extrahiert. {len(X)} Fuesse gefunden.")
    print("Trainiere Random Forest Regressor Model...")
    
    # Die eigentliche KI: 100 Entscheidungsbaeume lernen die Anatomie
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)
    
    # KI-Gehirn speichern
    model_path = os.path.join(dataset_dir, "ignite_ai_model.pkl")
    joblib.dump(model, model_path)
    
    print(f"ERFOLG! KI-Modell wurde trainiert und gespeichert unter: {model_path}")