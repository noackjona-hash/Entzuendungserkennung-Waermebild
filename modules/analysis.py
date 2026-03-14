import cv2

def perform_bilateral_analysis(image, left_toes, right_toes):
    if len(left_toes) != 5 or len(right_toes) != 5:
        return image, []

    # Anatomische Spiegelung
    right_toes_matched = list(reversed(right_toes))
    delta_ts = []

    # KLINISCHE PARAMETER
    # 20 ist ein guter Wert für 'Verdacht', 35 ist 'Eindeutiger Befund'
    threshold_warn = 20 
    threshold_severe = 35

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        temp_l = l_data["temp"]
        temp_r = r_data["temp"]
        
        delta_t = temp_l - temp_r
        delta_ts.append(delta_t)
        
        cv2.circle(image, l_data["sensor"], 3, (255, 255, 255), -1)
        cv2.circle(image, r_data["sensor"], 3, (255, 255, 255), -1)

        # DIAGNOSE-LOGIK
        abs_diff = abs(delta_t)
        if abs_diff >= threshold_warn:
            # Nur markieren, wenn die Seite wirklich heißer ist
            if delta_t > 0:
                # Linker Fuß ist heißer
                color = (0, 0, 255) if abs_diff > threshold_severe else (0, 165, 255)
                draw_hotspot(image, l_data, delta_t, color)
                draw_normal(image, r_data)
            else:
                # Rechter Fuß ist heißer (hier im Bild ignorieren wir das meist)
                color = (0, 0, 255) if abs_diff > threshold_severe else (0, 165, 255)
                draw_hotspot(image, r_data, abs_diff, color)
                draw_normal(image, l_data)
        else:
            draw_normal(image, l_data)
            draw_normal(image, r_data)

    return image, delta_ts

def draw_hotspot(img, data, delta_t, color):
    tip = data["tip"]
    box_w, b_h_up, b_h_down = 35, 20, 60
    cv2.rectangle(img, (tip[0]-box_w, tip[1]-b_h_up), (tip[0]+box_w, tip[1]+b_h_down), color, 2)
    text = f"HOT (+{delta_t})" if delta_t < 35 else f"BEFUND (+{delta_t})"
    cv2.putText(img, text, (tip[0]-box_w, tip[1]-b_h_up-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_normal(img, data):
    tip, temp = data["tip"], data["temp"]
    cv2.circle(img, tip, 5, (0, 255, 0), -1)
    cv2.putText(img, str(temp), (tip[0]-15, tip[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)