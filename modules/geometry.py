import cv2
import numpy as np
import math

def analyze_extremities_and_heat(image, gray_img):
    # 1. Objekt vom Hintergrund trennen
    _, mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    contour = max(contours, key=cv2.contourArea)

    # 2. Schwerpunkt berechnen (Unsere Orientierung für "Oben" und "Unten")
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    cv2.circle(image, (cx, cy), 6, (255, 0, 0), -1)

    # 3. Konvexe Hülle (Das Gummiband)
    hull = cv2.convexHull(contour)
    
    raw_tips = []
    for point in hull:
        x, y = point[0]
        # Wir wollen nur Punkte, die in der oberen Hälfte (über dem Schwerpunkt) liegen
        if y < cy - 20: 
            raw_tips.append((x, y))

    # 4. Punkte zusammenfassen (Ein runder Zeh hat oft viele Punkte am Gummiband)
    merged_tips = []
    for tip in raw_tips:
        is_new = True
        for i, mt in enumerate(merged_tips):
            # Wenn ein Punkt nah an einem bereits gefundenen liegt (weniger als 35 Pixel)
            if math.sqrt((tip[0]-mt[0])**2 + (tip[1]-mt[1])**2) < 35:
                is_new = False
                # Behalte den Punkt, der weiter Oben ist (kleineres Y)
                if tip[1] < mt[1]:
                    merged_tips[i] = tip
                break
        if is_new:
            merged_tips.append(tip)

    # Von links nach rechts sortieren
    merged_tips = sorted(merged_tips, key=lambda x: x[0])

    # 5. Temperatur messen und Entzündung (Ausreißer) finden
    temperatures = []
    for tip in merged_tips:
        # Helligkeitswert (0-255) aus dem Graustufenbild an dieser Koordinate auslesen
        temp = gray_img[tip[1], tip[0]]
        temperatures.append(temp)
        
    if not temperatures:
        return image

    # Durchschnittliche Hitze aller Spitzen berechnen
    avg_temp = sum(temperatures) / len(temperatures)
    
    # 6. Einzeichnen und Diagnose stellen
    for i, tip in enumerate(merged_tips):
        temp = temperatures[i]
        
        # LOGIK FÜR ENTZÜNDUNG: Wenn die Spitze signifikant wärmer als der Durchschnitt ist
        is_inflamed = temp > avg_temp + 15 
        
        # Farbe festlegen (Rot = Entzündung, Grün = Normal)
        color = (0, 0, 255) if is_inflamed else (0, 255, 0)
        
        # Linie zum Schwerpunkt und Punkt zeichnen
        cv2.line(image, (cx, cy), tip, (255, 255, 255), 1)
        cv2.circle(image, tip, 6, color, -1)
        
        # Text ausgeben
        text = f"Wert:{temp} (HOT)" if is_inflamed else f"Wert:{temp}"
        cv2.putText(image, text, (tip[0] - 25, tip[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image