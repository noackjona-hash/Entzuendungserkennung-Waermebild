import cv2
import numpy as np

def find_thermal_peaks(gray_img, mask):
    """Sucht nach lokalen Helligkeits-Maxima (Wärmespitzen) innerhalb der Maske."""
    # Bild leicht weichzeichnen, um Rauschen zu minimieren
    blurred = cv2.GaussianBlur(gray_img, (15, 15), 0)
    
    # Lokale Maxima finden
    peaks = []
    # Wir iterieren über ein Raster, um nicht jeden Pixel zu prüfen
    step = 20
    for y in range(step, blurred.shape[0] - step, step):
        for x in range(step, blurred.shape[1] - step, step):
            if mask[y, x] == 0: continue # Nur im Bereich der Hand/des Fußes suchen
            
            # Ist dieser Punkt wärmer als seine Umgebung?
            roi = blurred[y-step:y+step, x-step:x+step]
            _, max_val, _, max_loc = cv2.minMaxLoc(roi)
            
            # Wenn das Maximum in der Mitte der ROI liegt, ist es ein echter Gipfel
            if max_loc == (step, step):
                peaks.append((x, y, max_val))
                
    return peaks

def analyze_shape_and_draw_skeleton(image, gray_img):
    # 1. Objekt vom Hintergrund trennen
    _, mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, "Unbekannt"

    contour = max(contours, key=cv2.contourArea)

    # 2. Schwerpunkt berechnen
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)

    # 3. Wärmespitzen (Peaks) finden
    # Wir erstellen eine Maske, die nur den oberen Teil des Objekts zeigt (wo Finger/Zehen sind)
    top_mask = np.zeros_like(mask)
    cv2.drawContours(top_mask, [contour], -1, 255, -1)
    # Alles unterhalb des Schwerpunkts abschneiden
    top_mask[cy:, :] = 0 
    
    peaks = find_thermal_peaks(gray_img, top_mask)

    # 4. Peaks filtern (zu nah beieinander liegende zusammenfassen)
    filtered_peaks = []
    for p in peaks:
        is_new = True
        for fp in filtered_peaks:
            if np.sqrt((p[0]-fp[0])**2 + (p[1]-fp[1])**2) < 40: # 40 Pixel Mindestabstand
                is_new = False
                # Den wärmeren Punkt behalten
                if p[2] > fp[2]:
                    filtered_peaks.remove(fp)
                    filtered_peaks.append(p)
                break
        if is_new:
            filtered_peaks.append(p)

    # 5. Die wärmsten 5 Peaks auswählen und von links nach rechts sortieren
    filtered_peaks = sorted(filtered_peaks, key=lambda x: x[2], reverse=True)[:5]
    filtered_peaks = sorted(filtered_peaks, key=lambda x: x[0])

    # 6. Ergebnisse einzeichnen
    for p in filtered_peaks:
        tip = (p[0], p[1])
        temp_val = p[2]
        
        cv2.line(image, (cx, cy), tip, (0, 255, 255), 2)
        cv2.circle(image, tip, 6, (0, 0, 255), -1)
        
        # Temperaturwert direkt an die Spitze schreiben!
        cv2.putText(image, str(temp_val), (tip[0] - 15, tip[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, "Topologie-Analyse"