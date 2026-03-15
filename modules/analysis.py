import cv2
import numpy as np

def perform_bilateral_analysis(image, left_toes, right_toes, threshold_warn, threshold_severe):
    if len(left_toes) != 5 or len(right_toes) != 5:
        return image, []

    right_toes_matched = list(reversed(right_toes))
    delta_ts = []
    
    # Overlay-Kopie fuer transparente Effekte
    overlay = image.copy()

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        temp_l = l_data["temp"]
        temp_r = r_data["temp"]
        
        delta_t = temp_l - temp_r
        delta_ts.append(delta_t)
        abs_diff = abs(delta_t)
        
        cv2.circle(overlay, l_data["sensor"], 3, (255, 255, 255), -1)
        cv2.circle(overlay, r_data["sensor"], 3, (255, 255, 255), -1)

        if abs_diff >= threshold_warn:
            hot_data = l_data if delta_t > 0 else r_data
            normal_data = r_data if delta_t > 0 else l_data
            
            # Farbe je nach Schweregrad
            if abs_diff >= threshold_severe:
                color = (0, 0, 255) # Rot (Schwer)
                label = f"SCHWER (+{abs_diff})"
            else:
                color = (0, 165, 255) # Orange (Verdacht)
                label = f"VERDACHT (+{abs_diff})"
                
            draw_hotspot(overlay, hot_data, color, label)
            draw_normal(overlay, normal_data)
        else:
            draw_normal(overlay, l_data)
            draw_normal(overlay, r_data)

    # Transparenz anwenden (60% Original, 40% Overlay)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    return image, delta_ts

def draw_hotspot(img, data, color, label):
    tip = data["tip"]
    box_w, b_h_up, b_h_down = 35, 20, 60
    
    # Gefuelltes, transparentes Rechteck (wird spaeter durch addWeighted gemischt)
    cv2.rectangle(img, (tip[0]-box_w, tip[1]-b_h_up), (tip[0]+box_w, tip[1]+b_h_down), color, cv2.FILLED)
    cv2.rectangle(img, (tip[0]-box_w, tip[1]-b_h_up), (tip[0]+box_w, tip[1]+b_h_down), (255,255,255), 2)
    
    cv2.putText(img, label, (tip[0]-box_w, tip[1]-b_h_up-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_normal(img, data):
    tip, temp = data["tip"], data["temp"]
    cv2.circle(img, tip, 5, (0, 255, 0), -1)
    cv2.putText(img, str(temp), (tip[0]-15, tip[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)