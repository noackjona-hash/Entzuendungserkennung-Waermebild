import cv2
import numpy as np
import math

def calculate_tdi(temp_l, temp_r):
    diff = abs(temp_l - temp_r)
    return round((diff / 255.0) * 100.0, 2)

def analyze_hotspot_details(gray_img, foot_mask, pt, max_temp):
    """Berechnet tiefgreifende klinische Metriken um einen Sensorpunkt."""
    x, y = pt
    r = 20
    x_s, x_e = max(0, x-r), min(gray_img.shape[1], x+r)
    y_s, y_e = max(0, y-r), min(gray_img.shape[0], y+r)
    
    roi_gray = gray_img[y_s:y_e, x_s:x_e]
    roi_mask = foot_mask[y_s:y_e, x_s:x_e]
    
    if roi_gray.size == 0:
        return 0, 0.0, max_temp

    # 1. Flaeche (Area): Wie viele Pixel sind innerhalb von 10% des Maximums?
    threshold_temp = max_temp * 0.90
    _, hot_mask = cv2.threshold(roi_gray, threshold_temp, 255, cv2.THRESH_BINARY)
    hot_mask = cv2.bitwise_and(hot_mask, hot_mask, mask=roi_mask)
    area = cv2.countNonZero(hot_mask)

    # 2. Thermischer Gradient: Durchschnittlicher Temperaturabfall zum Rand der ROI
    mean_border_temp = np.mean(roi_gray[0,:].tolist() + roi_gray[-1,:].tolist() + roi_gray[:,0].tolist() + roi_gray[:,-1].tolist())
    gradient = round(float(max_temp - mean_border_temp) / r, 2)

    return area, gradient

def perform_deep_analysis(image, left_toes, right_toes, gray_img, foot_mask, warn_th, severe_th):
    if len(left_toes) != 5 or len(right_toes) != 5:
        return image, []

    right_toes_matched = list(reversed(right_toes))
    detailed_results = []
    overlay = image.copy()

    # Gesamtfuss-Asymmetrie berechnen (Durchschnitt ueber alle 5 Zehenpaare)
    avg_l = sum([t["temp"] for t in left_toes]) / 5
    avg_r = sum([t["temp"] for t in right_toes_matched]) / 5
    fai_index = calculate_tdi(avg_l, avg_r)

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        t_l = l_data["temp"]
        t_r = r_data["temp"]
        tdi = calculate_tdi(t_l, t_r)
        
        # Tiefen-Analyse der Hotspots
        area_l, grad_l = analyze_hotspot_details(gray_img, foot_mask, l_data["sensor"], t_l)
        area_r, grad_r = analyze_hotspot_details(gray_img, foot_mask, r_data["sensor"], t_r)

        is_left_hotter = t_l > t_r
        diff = abs(t_l - t_r)

        result_dict = {
            "toe_index": i,
            "t_l": t_l, "t_r": t_r,
            "tdi": tdi,
            "diff": diff,
            "is_left_hotter": is_left_hotter,
            "area_l": area_l, "area_r": area_r,
            "grad_l": grad_l, "grad_r": grad_r,
            "fai": fai_index,
            "status": "NORMAL"
        }

        draw_sensor_target(overlay, l_data["sensor"])
        draw_sensor_target(overlay, r_data["sensor"])

        if tdi >= warn_th:
            hot_data = l_data if is_left_hotter else r_data
            normal_data = r_data if is_left_hotter else l_data
            
            if tdi >= severe_th:
                color = (0, 0, 255) # Rot
                result_dict["status"] = "SCHWER"
            else:
                color = (0, 140, 255) # Orange
                result_dict["status"] = "VERDACHT"
                
            draw_hotspot(overlay, hot_data, color, result_dict["status"], tdi)
            draw_normal(overlay, normal_data)
        else:
            draw_normal(overlay, l_data)
            draw_normal(overlay, r_data)

        detailed_results.append(result_dict)

    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    return image, detailed_results

def draw_sensor_target(img, pt):
    cv2.drawMarker(img, pt, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
    cv2.circle(img, pt, 2, (255, 255, 255), -1)

def draw_hotspot(img, data, color, status, tdi):
    pt = data["sensor"]
    box_w, box_h = 45, 45
    start_pt = (pt[0]-box_w, pt[1]-box_h)
    end_pt = (pt[0]+box_w, pt[1]+box_h)
    
    cv2.rectangle(img, start_pt, end_pt, color, cv2.FILLED)
    cv2.rectangle(img, start_pt, end_pt, (255,255,255), 2)
    cv2.putText(img, f"{status} ({tdi}%)", (start_pt[0], start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def draw_normal(img, data):
    pt, temp = data["sensor"], data["temp"]
    cv2.circle(img, pt, 6, (0, 255, 0), -1)
    cv2.putText(img, str(temp), (pt[0]-15, pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)