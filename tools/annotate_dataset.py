import cv2
import os
import csv
import numpy as np

# ROBUSTE PFADE: Findet den Hauptordner automatisch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")
LABEL_FILE = os.path.join(BASE_DIR, "dataset", "labels.csv")

current_points = []
img_display = None

def click_event(event, x, y, flags, param):
    global current_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 10:
            current_points.append((x, y))
            cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(img_display, str(len(current_points)), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("IGNITE Annotator", img_display)

def main():
    global current_points, img_display
    
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Ordner {IMAGE_DIR} erstellt. Bitte lege hier deine 200 Wärmebilder ab!")
        return

    labeled_images = set()
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row: labeled_images.add(row[0])

    images = [img for img in os.listdir(IMAGE_DIR) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_to_label = [img for img in images if img not in labeled_images]

    if not images_to_label:
        print("Alle Bilder sind bereits gelabelt! Starker Job, Jona!")
        return

    print(f"Es warten noch {len(images_to_label)} Bilder auf dich.")
    print("ANLEITUNG: Klicke die 10 Zehen von LINKS nach RECHTS an (5 linker Fuss, 5 rechter Fuss).")
    
    if not os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["filename"] + [f"x{i}" for i in range(10)] + [f"y{i}" for i in range(10)]
            writer.writerow(header)

    # Sicherstellen, dass die GUI-Version von OpenCV installiert ist
    try:
        cv2.namedWindow("IGNITE Annotator")
        cv2.setMouseCallback("IGNITE Annotator", click_event)
    except cv2.error:
        print("\n" + "="*60)
        print("🚨 FEHLER: FEHLENDE FENSTER-UNTERSTÜTZUNG (HEADLESS)")
        print("="*60)
        print("Du hast 'opencv-python-headless' installiert. Diese Version ist")
        print("perfekt für den Docker-Server, aber sie kann keine Fenster öffnen!")
        print("\nUm jetzt auf Windows labeln zu können, tippe ins Terminal:")
        print("  pip uninstall -y opencv-python-headless")
        print("  pip install opencv-python")
        print("="*60 + "\n")
        return

    for img_name in images_to_label:
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # --- ROBUSTER BILD-IMPORT FÜR WINDOWS UMLAUTE (ä, ö, ü) ---
        # Anstatt cv2.imread lesen wir die Bytes direkt und decodieren sie dann.
        try:
            with open(img_path, "rb") as f:
                img_bytes = bytearray(f.read())
            nparr = np.asarray(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Konnte {img_name} nicht laden: {e}")
            continue
            
        if img is None: continue
        # ------------------------------------------------------------
        
        img_display = img.copy()
        current_points = []
        
        while len(current_points) < 10:
            cv2.imshow("IGNITE Annotator", img_display)
            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                print("Labeling abgebrochen.")
                cv2.destroyAllWindows()
                return

        with open(LABEL_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            row = [img_name]
            row.extend([p[0] for p in current_points])
            row.extend([p[1] for p in current_points])
            writer.writerow(row)
            print(f"[+] Gespeichert: {img_name}")

    cv2.destroyAllWindows()
    print("Boom! Alle Bilder gelabelt. Weiter gehts mit der Augmentation.")

if __name__ == "__main__":
    main()