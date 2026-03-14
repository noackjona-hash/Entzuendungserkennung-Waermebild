import cv2

def perform_bilateral_analysis(image, left_toes, right_toes):
    """Führt die bilaterale Symmetrie-Analyse durch und zeichnet die Ergebnisse."""
    if len(left_toes) != 5 or len(right_toes) != 5:
        return image, []

    # ANATOMIE-ABGLEICH (Die "Spiegelung")
    # Linker Fuss: Klein -> Ring -> Mittel -> Zeige -> Gross (Index 0->4)
    # Rechter Fuss: Gross -> Zeige -> Mittel -> Ring -> Klein (Index 0->4)
    
    # Wir spiegeln die rechte Liste, damit wir anatomical dieselben Zehen vergleichen
    right_toes_matched = list(reversed(right_toes))

    # Liste fuer die Temperaturdifferenzen (Delta T)
    delta_ts = []

    # Schwellenwert fuer eine Entzuendung (z.B. 15 Einheiten Temperaturdifferenz)
    threshold_diff = 15 

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        temp_l = l_data["temp"]
        temp_r = r_data["temp"]
        
        # Die wissenschaftliche Symmetrie-Formel: Delta T = T_Links - T_Rechts
        delta_t = temp_l - temp_r
        delta_ts.append(delta_t)
        
        # Sensoren (weisse Punkte) einzeichnen
        cv2.circle(image, l_data["sensor"], 3, (255, 255, 255), -1)
        cv2.circle(image, r_data["sensor"], 3, (255, 255, 255), -1)

        # DIAGNOSE: Ist die Seitendifferenz groesser als unser Schwellenwert?
        if abs(delta_t) > threshold_diff:
            # Wer ist der Entzuendungs-Ausreisser?
            if delta_t > 0:
                # Linker Zeh ist entzuendet
                draw_hotspot(image, l_data, delta_t)
                draw_normal(image, r_data)
            else:
                # Rechter Zeh ist entzuendet
                # Wir geben das Delta T positiv an fuer die Anzeige
                draw_hotspot(image, r_data, abs(delta_t))
                draw_normal(image, l_data)
        else:
            # Symmetrisch und gesund
            draw_normal(image, l_data)
            draw_normal(image, r_data)

    # Bild UND die berechneten Differenzen zurueckgeben
    return image, delta_ts

def draw_hotspot(img, data, delta_t):
    """Zeichnet eine rote Bounding Box um eine Entzuendung."""
    tip = data["tip"]
    
    # Box-Parameter
    box_width = 35
    box_height_up = 20
    box_height_down = 60
    
    # Rote Box zeichnen
    cv2.rectangle(img, (tip[0]-box_width, tip[1]-box_height_up), (tip[0]+box_width, tip[1]+box_height_down), (0, 0, 255), 2)
    
    # Text-Label: Abweichung anzeigen
    text = f"HOT (+{delta_t})"
    cv2.putText(img, text, (tip[0]-box_width, tip[1]-box_height_up-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def draw_normal(img, data):
    """Markiert einen gesunden Zeh mit einem gruenen Punkt."""
    tip = data["tip"]
    temp = data["temp"]
    
    # Gruener Punkt fuer gesund
    cv2.circle(img, tip, 5, (0, 255, 0), -1)
    
    # Temperaturwert daneben schreiben
    cv2.putText(img, str(temp), (tip[0]-15, tip[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)