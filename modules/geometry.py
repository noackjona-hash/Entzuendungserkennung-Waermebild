import cv2
import numpy as np

def analyze_extremities_and_smart_heat(image, gray_img):
    # 1. Maske erstellen: Hand/Fuß vom Hintergrund trennen
    _, mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Größte Kontur ist unser Körperteil
    body_contour = max(contours, key=cv2.contourArea)
    
    # 2. Saubere Maske nur für den Fuß/die Hand zeichnen
    clean_mask = np.zeros_like(gray_img)
    cv2.drawContours(clean_mask, [body_contour], -1, 255, -1)

    # 3. Statistische Analyse der Temperatur (Helligkeit)
    # Durchschnittstemperatur des gesamten Fußes (ignoriert den kalten Hintergrund)
    mean_val = cv2.mean(gray_img, mask=clean_mask)[0]
    
    # Den absolut heißesten Punkt finden
    _, max_val, _, max_loc = cv2.minMaxLoc(gray_img, mask=clean_mask)

    # Optisches Gimmick: Durchschnitt anzeigen
    cv2.putText(image, f"Körper-Durchschnitt: {int(mean_val)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 4. SMARTE ENTZÜNDUNGS-LOGIK
    # Eine Entzündung muss deutlich heißer sein als der Rest des Körpers
    threshold_difference = 30 # Ab X Einheiten über dem Durchschnitt schlagen wir Alarm
    
    if max_val > mean_val + threshold_difference and max_val > 150:
        # 5. Den Entzündungs-Cluster isolieren
        # Wir suchen alle Pixel, die fast so heiß sind wie das Maximum
        _, hot_mask = cv2.threshold(gray_img, max_val - 25, 255, cv2.THRESH_BINARY)
        
        # Sicherstellen, dass die Hitze wirklich auf dem Fuß liegt
        hot_mask = cv2.bitwise_and(hot_mask, clean_mask)
        
        hot_contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if hot_contours:
            # Den größten Hitze-Cluster nehmen (filtert Rauschen)
            largest_hotspot = max(hot_contours, key=cv2.contourArea)
            
            # 6. Bounding Box berechnen und zeichnen
            x, y, w, h = cv2.boundingRect(largest_hotspot)
            
            # Etwas Platz (Padding) um die Box lassen
            pad = 15
            cv2.rectangle(image, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 0, 255), 3)
            
            # Warn-Label anbringen
            cv2.putText(image, f"ENTZUENDUNG! Max: {int(max_val)}", (x-pad, y-pad-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Fadenkreuz direkt auf das Epizentrum (den heißesten Pixel) setzen
            cv2.drawMarker(image, max_loc, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            
            print(f"Befund: Entzuendung erkannt! (Spitze: {int(max_val)}, Durchschnitt: {int(mean_val)})")
    else:
        print(f"Befund: Normal. (Keine signifikanten Ausreisser erkannt)")

    return image