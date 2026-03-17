import pandas as pd
import cv2
import numpy as np
import os
import random
import time

def rotate_image_and_landmarks(img, landmarks_x, landmarks_y, angle):
    """Dreht das Bild und berechnet die neuen Landmark-Koordinaten."""
    (h, w) = img.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    # OpenCV Rotationsmatrix holen
    # Winkel positiv: Gegen den Uhrzeigersinn, Negativ: Im Uhrzeigersinn
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # 1. Bild drehen (Kante wird schwarz gefuellt)
    rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=(0,0,0))

    # 2. Landmark-Koordinaten drehen
    new_landmarks_x = []
    new_landmarks_y = []
    
    # Formatierung fuer Matrix-Multiplikation
    coords = np.column_stack((landmarks_x, landmarks_y, np.ones(len(landmarks_x))))
    
    # Transformation anwenden: M * v
    new_coords = M.dot(coords.T).T
    
    # Bounding-Box Check: Koordinaten im Bild halten
    for i in range(len(new_coords)):
        nx = max(0, min(w - 1, int(new_coords[i][0])))
        ny = max(0, min(h - 1, int(new_coords[i][1])))
        new_landmarks_x.append(nx)
        new_landmarks_y.append(ny)
        
    return rotated_img, new_landmarks_x, new_landmarks_y

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """Simuliert unterschiedliche Thermal-Signaturen durch Helligkeit/Kontrast."""
    # alpha 1.0-3.0 (Kontrast), beta 0-100 (Helligkeit)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

if __name__ == "__main__":
    print("=== IGNITE DATA AUGMENTATION PROCESS STARTED ===")
    dataset_dir = "dataset"
    img_dir = os.path.join(dataset_dir, "images")
    label_file = os.path.join(dataset_dir, "labels.csv")
    
    # Neuer Ordner fuer augmentierte Daten
    aug_img_dir = os.path.join(dataset_dir, "augmented_images")
    if not os.path.exists(aug_img_dir):
        os.makedirs(aug_img_dir)
    
    # Originaldaten laden
    if not os.path.exists(label_file):
        print("Fehler: Keine labels.csv gefunden!")
        exit()
        
    df_original = pd.read_csv(label_file)
    print(f"{len(df_original)} Original-Bilder gefunden.")

    # Neue Liste fuer augmentierte Labels
    aug_data = []

    # Wir augmentieren jedes Originalbild mehrfach
    for index, row in df_original.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        
        # Originalbild laden (BGR fuer Farb-Anzeige, falls vorhanden, spaeter loader.py nutzen)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Original-Koordinaten extrahieren
        l_x, l_y = [], []
        r_x, r_y = [], []
        
        for i in range(1, 6):
            l_x.append(row[f'L_Zeh{i}_x'])
            l_y.append(row[f'L_Zeh{i}_y'])
            r_x.append(row[f'R_Zeh{i}_x'])
            r_y.append(row[f'R_Zeh{i}_y'])
            
        # Sammelliste fuer beide Fuesse
        all_x = l_x + r_x
        all_y = l_y + r_y
        
        base_name = os.path.splitext(row['filename'])[0]

        print(f"Augmentiere: {row['filename']}...")

        # --- AUGMENTATIONS-STRATEGIE ---
        timestamp = int(time.time() * 100)
        
        # 1. Original (unveraendert) auch im neuen Set behalten
        # aug_data.append(row) # Wir erstellen eine komplett neue Tabelle

        # 2. Leichte Rotationen (Simulation ungerader Fussstellung)
        for angle in [-15, -10, -5, 5, 10, 15]:
            # Drehen
            rot_img, new_x, new_y = rotate_image_and_landmarks(img, all_x, all_y, angle)
            
            # Helligkeit leicht variieren
            bright_factor = random.uniform(0.9, 1.1)
            contrast_factor = random.uniform(0.95, 1.05)
            final_img = adjust_brightness_contrast(rot_img, alpha=contrast_factor, beta=int(bright_factor*10))
            
            # Speichern
            new_fname = f"aug_{base_name}_rot{angle}_{timestamp}.png"
            new_fpath = os.path.join(aug_img_dir, new_fname)
            cv2.imwrite(new_fpath, final_img)
            
            # Neue Zeile fuer CSV bauen
            new_row = {"filename": new_fname}
            for i in range(10): # 10 Zehen insgesamt
                # Zuordnung L1-L5, R1-R5
                side = "L" if i < 5 else "R"
                toe_num = (i % 5) + 1
                new_row[f"{side}_Zeh{toe_num}_x"] = new_x[i]
                new_row[f"{side}_Zeh{toe_num}_y"] = new_y[i]
            
            aug_data.append(new_row)

        # 3. Nur Helligkeits-Variationen (Simulierung Temperaturwechsel)
        for b_factor in [0.85, 1.15]:
            bright_img = adjust_brightness_contrast(img, alpha=1.0, beta=int(b_factor*15))
            new_fname = f"aug_{base_name}_bright{b_factor}_{timestamp}.png"
            new_fpath = os.path.join(aug_img_dir, new_fname)
            cv2.imwrite(new_fpath, bright_img)
            
            # Koordinaten bleiben gleich!
            new_row = {"filename": new_fname}
            for i in range(10):
                side = "L" if i < 5 else "R"
                toe_num = (i % 5) + 1
                new_row[f"{side}_Zeh{toe_num}_x"] = all_x[i]
                new_row[f"{side}_Zeh{toe_num}_y"] = all_y[i]
            aug_data.append(new_row)

    # Finale Daten zusammenstellen
    df_augmented = pd.DataFrame(aug_data)
    
    # Combined: Wir fuegen Original und Augmentiert zusammen (Profi-Ansatz)
    # Da die Originale in einem anderen Ordner liegen, muessten wir die Pfade im CSV anpassen.
    # Der Einfachheit halber: Wir arbeiten NUR mit dem augmentierten Set weiter (das Originale ist da indirekt ja drin).
    
    aug_label_file = os.path.join(dataset_dir, "labels_augmented.csv")
    df_augmented.to_csv(aug_label_file, index=False)
    
    total_imgs = len(df_augmented)
    print(f"=== ERFOLG! ===")
    print(f"Das Dataset wurde von {len(df_original)} auf {total_imgs} Bilder vermehrt.")
    print(f"Augmentierte Bilder: {aug_img_dir}/")
    print(f"Neues CSV-Label-File: {aug_label_file}")
    print("Mache jetzt weiter mit train_ai.py!")