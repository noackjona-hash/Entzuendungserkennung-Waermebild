import cv2

def perform_bilateral_analysis(image, left_toes, right_toes):
    if len(left_toes) != 5 or len(right_toes) != 5:
        return image

    # Anatomischer Abgleich (Rechte Liste spiegeln)
    right_toes_matched = list(reversed(right_toes))

    # MEDIZINISCHER SCHWELLENWERT
    # Erst ab 20 Einheiten Differenz zwischen LINKS und RECHTS gilt es als Befund
    clinical_threshold = 20 

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        diff = l_data["temp"] - r_data["temp"]
        abs_diff = abs(diff)

        # Sensoren zeichnen
        cv2.circle(image, l_data["sensor"], 3, (255, 255, 255), -1)
        cv2.circle(image, r_data["sensor"], 3, (255, 255, 255), -1)

        if abs_diff > clinical_threshold:
            # Nur die Seite markieren, die wirklich heisser ist
            if diff > 0:
                draw_hotspot(image, l_data, abs_diff)
                draw_normal(image, r_data)
            else:
                draw_hotspot(image, r_data, abs_diff)
                draw_normal(image, l_data)
        else:
            # Symmetrisch -> Alles okay
            draw_normal(image, l_data)
            draw_normal(image, r_data)

    return image

def draw_hotspot(img, data, diff):
    tip = data["tip"]
    # Rote Box fuer Entzuendung
    cv2.rectangle(img, (tip[0]-35, tip[1]-20), (tip[0]+35, tip[1]+60), (0, 0, 255), 2)
    cv2.putText(img, f"BEFUND (+{diff})", (tip[0]-40, tip[1]-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def draw_normal(img, data):
    tip = data["tip"]
    # Gruener Punkt fuer gesund
    cv2.circle(img, tip, 5, (0, 255, 0), -1)
    cv2.putText(img, str(data["temp"]), (tip[0]-15, tip[1]-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)