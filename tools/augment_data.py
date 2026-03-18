import cv2
import numpy as np
import pandas as pd
import os

# ROBUSTE PFADE
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")
AUGMENTED_DIR = os.path.join(BASE_DIR, "dataset", "augmented_images")
LABEL_FILE = os.path.join(BASE_DIR, "dataset", "labels.csv")
AUGMENTED_LABEL_FILE = os.path.join(BASE_DIR, "dataset", "labels_augmented.csv")

def rotate_image_and_points(image, points_x, points_y, angle):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    new_x, new_y = [], []
    for x, y in zip(points_x, points_y):
        px = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        py = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        new_x.append(int(px))
        new_y.append(int(py))
        
    return rotated_img, new_x, new_y

def safe_imwrite(filename, img):
    """Speichert Bilder auch dann, wenn der Pfad Umlaute enthält."""
    is_success, im_buf_arr = cv2.imencode(".jpeg", img)
    if is_success:
        im_buf_arr.tofile(filename)

def main():
    if not os.path.exists(AUGMENTED_DIR):
        os.makedirs(AUGMENTED_DIR)
        
    if not os.path.exists(LABEL_FILE):
        print(f"Fehler: Keine labels.csv gefunden unter: {LABEL_FILE}")
        print("Bitte zuerst annotate_dataset.py ausführen und mindestens ein Bild labeln!")
        return

    # Prüfen, ob wirklich gelabelt wurde
    df = pd.read_csv(LABEL_FILE)
    if df.empty:
        print("Die labels.csv ist leer. Du hast noch keine Bilder gelabelt!")
        return

    augmented_data = []
    print(f"Starte Augmentation von {len(df)} Original-Bildern...")
    
    for index, row in df.iterrows():
        img_name = row['filename']
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # --- ROBUSTER BILD-IMPORT FÜR UMLAUTE ---
        try:
            with open(img_path, "rb") as f:
                img_bytes = bytearray(f.read())
            nparr = np.asarray(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Überspringe {img_name} (nicht lesbar: {e})")
            continue
            
        if img is None: continue
        
        pts_x = [row[f'x{i}'] for i in range(10)]
        pts_y = [row[f'y{i}'] for i in range(10)]
        
        # Original speichern
        safe_imwrite(os.path.join(AUGMENTED_DIR, img_name), img)
        augmented_data.append(row.to_dict())
        
        # Rotation -15
        rot1_img, rx1, ry1 = rotate_image_and_points(img, pts_x, pts_y, -15)
        rot1_name = "aug_rotM15_" + img_name
        safe_imwrite(os.path.join(AUGMENTED_DIR, rot1_name), rot1_img)
        
        new_row = {'filename': rot1_name}
        for i in range(10):
            new_row[f'x{i}'] = rx1[i]
            new_row[f'y{i}'] = ry1[i]
        augmented_data.append(new_row)
        
        # Rotation +15
        rot2_img, rx2, ry2 = rotate_image_and_points(img, pts_x, pts_y, 15)
        rot2_name = "aug_rotP15_" + img_name
        safe_imwrite(os.path.join(AUGMENTED_DIR, rot2_name), rot2_img)
        
        new_row2 = {'filename': rot2_name}
        for i in range(10):
            new_row2[f'x{i}'] = rx2[i]
            new_row2[f'y{i}'] = ry2[i]
        augmented_data.append(new_row2)
        
        # Helligkeit
        bright_img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        bright_name = "aug_bright_" + img_name
        safe_imwrite(os.path.join(AUGMENTED_DIR, bright_name), bright_img)
        
        new_row3 = {'filename': bright_name}
        for i in range(10):
            new_row3[f'x{i}'] = pts_x[i]
            new_row3[f'y{i}'] = pts_y[i]
        augmented_data.append(new_row3)

    aug_df = pd.DataFrame(augmented_data)
    aug_df.to_csv(AUGMENTED_LABEL_FILE, index=False)
    print(f"Erfolg! Aus {len(df)} Bildern wurden {len(aug_df)} Trainingsdaten generiert!")

if __name__ == "__main__":
    main()