import cv2
import numpy as np

def calculate_tdi(temp_l, temp_r):
    """Berechnet den Thermal Divergence Index (TDI) in Prozent."""
    diff = abs(temp_l - temp_r)
    tdi = (diff / 255.0) * 100.0
    return round(tdi, 1)

def perform_bilateral_analysis(image, left_toes, right_toes, threshold_warn_tdi, threshold_severe_tdi):
    if len(left_toes) != 5 or len(right_toes) != 5:
        return image, []

    right_toes_matched = list(reversed(right_toes))
    analysis_results = []
    
    overlay = image.copy()

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        temp_l = l_data["temp"]
        temp_r = r_data["temp"]
        
        # TDI statt nackter Temperaturdifferenz nutzen
        tdi = calculate_tdi(temp_l, temp_r)
        analysis_results.append({
            "temp_l": temp_l,
            "temp_r": temp_r,
            "tdi": tdi,
            "is_left_hotter": temp_l > temp_r
        })
        
        cv2.circle(overlay, l_data["sensor"], 4, (255, 255, 255), -1)
        cv2.circle(overlay, r_data["sensor"], 4, (255, 255, 255), -1)

        if tdi >= threshold_warn_tdi:
            hot_data = l_data if temp_l > temp_r else r_data
            normal_data = r_data if temp_l > temp_r else l_data
            
            if tdi >= threshold_severe_tdi:
                color = (0, 0, 255) # Rot
                label = f"SCHWER (TDI: {tdi})"
            else:
                color = (0, 165, 255) # Orange
                label = f"VERDACHT (TDI: {tdi})"
                
            draw_hotspot(overlay, hot_data, color, label)
            draw_normal(overlay, normal_data)
        else:
            draw_normal(overlay, l_data)
            draw_normal(overlay, r_data)

    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    return image, analysis_results

def draw_hotspot(img, data, color, label):
    tip = data["tip"]
    box_w, b_h_up, b_h_down = 40, 20, 70
    
    cv2.rectangle(img, (tip[0]-box_w, tip[1]-b_h_up), (tip[0]+box_w, tip[1]+b_h_down), color, cv2.FILLED)
    cv2.rectangle(img, (tip[0]-box_w, tip[1]-b_h_up), (tip[0]+box_w, tip[1]+b_h_down), (255,255,255), 2)
    cv2.putText(img, label, (tip[0]-box_w, tip[1]-b_h_up-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def draw_normal(img, data):
    tip, temp = data["tip"], data["temp"]
    cv2.circle(img, tip, 5, (0, 255, 0), -1)
    cv2.putText(img, str(temp), (tip[0]-15, tip[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)