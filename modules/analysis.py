import cv2
import numpy as np

def calculate_tdi(temp_l, temp_r):
    """Berechnet den Thermal Divergence Index (TDI) auf 2 Nachkommastellen genau."""
    diff = abs(temp_l - temp_r)
    tdi = (diff / 255.0) * 100.0
    return round(tdi, 2)

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
        
        tdi = calculate_tdi(temp_l, temp_r)
        analysis_results.append({
            "temp_l": temp_l,
            "temp_r": temp_r,
            "tdi": tdi,
        })
        
        # Fadenkreuz-Markierung fuer maximale Praezision
        draw_sensor_target(overlay, l_data["sensor"])
        draw_sensor_target(overlay, r_data["sensor"])

        if tdi >= threshold_warn_tdi:
            hot_data = l_data if temp_l > temp_r else r_data
            normal_data = r_data if temp_l > temp_r else l_data
            
            if tdi >= threshold_severe_tdi:
                color = (0, 0, 255) # Rot
                label = f"SCHWER (TDI: {tdi}%)"
            else:
                color = (0, 140, 255) # Orange
                label = f"VERDACHT (TDI: {tdi}%)"
                
            draw_hotspot(overlay, hot_data, color, label)
            draw_normal(overlay, normal_data)
        else:
            draw_normal(overlay, l_data)
            draw_normal(overlay, r_data)

    # Hochwertige Transparenz ueberlagern
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    return image, analysis_results

def draw_sensor_target(img, pt):
    """Zeichnet ein wissenschaftliches Sensor-Fadenkreuz."""
    cv2.drawMarker(img, pt, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
    cv2.circle(img, pt, 2, (255, 255, 255), -1)

def draw_hotspot(img, data, color, label):
    pt = data["sensor"] # Box zentriert sich jetzt um den ECHTEN Sensor, nicht um die Kante!
    box_w, box_h = 45, 45
    
    start_pt = (pt[0]-box_w, pt[1]-box_h)
    end_pt = (pt[0]+box_w, pt[1]+box_h)
    
    cv2.rectangle(img, start_pt, end_pt, color, cv2.FILLED)
    cv2.rectangle(img, start_pt, end_pt, (255,255,255), 2)
    cv2.putText(img, label, (start_pt[0], start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_normal(img, data):
    pt, temp = data["sensor"], data["temp"]
    cv2.circle(img, pt, 6, (0, 255, 0), -1)
    cv2.circle(img, pt, 6, (255, 255, 255), 1)
    cv2.putText(img, f"{temp}", (pt[0]-15, pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)