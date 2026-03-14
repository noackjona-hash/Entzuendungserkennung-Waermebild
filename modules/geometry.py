import cv2

def analyze_shape_and_draw_skeleton(image, gray_img):
    # 1. Bild binarisieren (Fuß/Hand vom Hintergrund trennen)
    _, mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, "Unbekannt"

    # 2. Größte Kontur nehmen
    contour = max(contours, key=cv2.contourArea)

    # 3. Den Schwerpunkt berechnen (Mitte der Handfläche / Fußsohle)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Schwerpunkt als Startpunkt für das Skelett einzeichnen (Blau)
    cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)

    # 4. Gummiband (Konvexe Hülle) und die Lücken (Defekte) berechnen
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    
    # Try-Except Block, falls die Kontur zu simpel ist
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except:
        defects = None

    deep_valleys = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0]) # Spitze (Finger/Zeh)
            
            # Die Distanz 'd' ist die Tiefe des Tals. 
            # OpenCV speichert das als Vielfaches von 256, also teilen wir.
            depth = d / 256.0

            # Wenn das Tal tief genug ist, ist es ein Finger-Spalt
            if depth > 20: 
                deep_valleys += 1
                
                # SKELETT-LINIE zeichnen: Vom Schwerpunkt zur Spitze (Gelb)
                cv2.line(image, (cx, cy), start, (0, 255, 255), 2)
                
                # GELENK / SPITZE markieren (Rot)
                cv2.circle(image, start, 5, (0, 0, 255), -1)

    # 5. Unterscheidung: Hände haben tiefe Täler zwischen den Fingern, Füße kaum.
    if deep_valleys >= 3:
        erkennung = "Hand"
    else:
        erkennung = "Fuss"

    return image, erkennung