import cv2

def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Fehler: {path} nicht gefunden.")
        return None
    # Konvertierung für die Analyse (Graustufen = Helligkeit = Wärme)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray