import cv2
import numpy as np
import math

def calculate_tdi(temp_l, temp_r):
    """Kalkuliert den Thermal Divergence Index (TDI) als prozentuale Abweichung."""
    diff = abs(temp_l - temp_r)
    return round((diff / 255.0) * 100.0, 2)

def analyze_hotspot_details(gray_img, foot_mask, pt, max_temp):
    """
    Extrahiert tiefgreifende klinische und statistische Metriken um einen definierten Sensorpunkt.
    """
    x, y = pt
    r = 20
    x_s, x_e = max(0, x-r), min(gray_img.shape[1], x+r)
    y_s, y_e = max(0, y-r), min(gray_img.shape[0], y+r)
    
    roi_gray = gray_img[y_s:y_e, x_s:x_e]
    roi_mask = foot_mask[y_s:y_e, x_s:x_e]
    
    if roi_gray.size == 0 or np.count_nonzero(roi_mask) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    # 1. Lokale Statistik (Mean & Standard Deviation)
    valid_pixels = roi_gray[roi_mask > 0]
    mean_temp = round(float(np.mean(valid_pixels)), 2)
    std_dev = round(float(np.std(valid_pixels)), 2)

    # 2. Hotspot Area (Fläche der obersten 10% Hitze)
    threshold_temp = max_temp * 0.90
    _, hot_mask = cv2.threshold(roi_gray, threshold_temp, 255, cv2.THRESH_BINARY)
    hot_mask = cv2.bitwise_and(hot_mask, hot_mask, mask=roi_mask)
    area = cv2.countNonZero(hot_mask)

    # 3. Thermischer Gradient (Rand-Temperatur vs. Maximum)
    mean_border_temp = np.mean(roi_gray[0,:].tolist() + roi_gray[-1,:].tolist() + roi_gray[:,0].tolist() + roi_gray[:,-1].tolist())
    gradient = round(float(max_temp - mean_border_temp) / r, 2)
    
    # 4. Thermischer Schwerpunkt-Versatz (Centroid Shift)
    M = cv2.moments(hot_mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Distanz vom geometrischen Zentrum der ROI zum Hitzeschwerpunkt
        shift = round(math.sqrt((cx - r)**2 + (cy - r)**2), 2)
    else:
        shift = 0.0

    return area, gradient, mean_temp, std_dev, shift

def perform_deep_analysis(left_toes, right_toes, gray_img, foot_mask, warn_th, severe_th):
    """
    Führt die rein datengetriebene Analyse aus (API-kompatibel).
    Retourniert strukturierte JSON/Dict-Datenstände ohne Bildmanipulation.
    """
    if len(left_toes) != 5 or len(right_toes) != 5:
        return {"error": "Incomplete coordinate data", "global_metrics": {}, "regional_metrics": []}

    right_toes_matched = list(reversed(right_toes))
    detailed_results = []

    avg_l = sum([t["temp"] for t in left_toes]) / 5
    avg_r = sum([t["temp"] for t in right_toes_matched]) / 5
    fai_index = calculate_tdi(avg_l, avg_r)

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        t_l = l_data["temp"]
        t_r = r_data["temp"]
        tdi = calculate_tdi(t_l, t_r)
        
        area_l, grad_l, mean_l, std_l, shift_l = analyze_hotspot_details(gray_img, foot_mask, l_data["sensor"], t_l)
        area_r, grad_r, mean_r, std_r, shift_r = analyze_hotspot_details(gray_img, foot_mask, r_data["sensor"], t_r)

        is_left_hotter = t_l > t_r
        diff = abs(t_l - t_r)

        status = "NORMAL"
        if tdi >= severe_th:
            status = "SCHWER"
        elif tdi >= warn_th:
            status = "VERDACHT"

        result_dict = {
            "toe_index": i,
            "coordinates_left": l_data["sensor"],
            "coordinates_right": r_data["sensor"],
            "t_l": t_l, "t_r": t_r,
            "tdi": tdi,
            "diff": diff,
            "is_left_hotter": is_left_hotter,
            "area_l": area_l, "area_r": area_r,
            "grad_l": grad_l, "grad_r": grad_r,
            "mean_l": mean_l, "mean_r": mean_r,
            "std_l": std_l, "std_r": std_r,
            "shift_l": shift_l, "shift_r": shift_r,
            "status": status
        }
        detailed_results.append(result_dict)

    return {
        "global_metrics": {"fai": fai_index},
        "regional_metrics": detailed_results
    }

def render_diagnostics(image, analysis_payload):
    """
    Übernimmt den strukturierten API-Payload und generiert das visuelle Overlay.
    """
    if "error" in analysis_payload:
        return image

    overlay = image.copy()
    regional_data = analysis_payload.get("regional_metrics", [])

    for data in regional_data:
        pt_l = data["coordinates_left"]
        pt_r = data["coordinates_right"]
        status = data["status"]
        tdi = data["tdi"]
        is_left_hotter = data["is_left_hotter"]

        cv2.drawMarker(overlay, pt_l, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.circle(overlay, pt_l, 2, (255, 255, 255), -1)
        cv2.drawMarker(overlay, pt_r, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.circle(overlay, pt_r, 2, (255, 255, 255), -1)

        hot_pt = pt_l if is_left_hotter else pt_r
        normal_pt = pt_r if is_left_hotter else pt_l

        if status == "SCHWER":
            _draw_bounding_box(overlay, hot_pt, (0, 0, 255), status, tdi)
            _draw_normal_marker(overlay, normal_pt, data["t_r"] if is_left_hotter else data["t_l"])
        elif status == "VERDACHT":
            _draw_bounding_box(overlay, hot_pt, (0, 140, 255), status, tdi)
            _draw_normal_marker(overlay, normal_pt, data["t_r"] if is_left_hotter else data["t_l"])
        else:
            _draw_normal_marker(overlay, pt_l, data["t_l"])
            _draw_normal_marker(overlay, pt_r, data["t_r"])

    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    return image

def _draw_bounding_box(img, pt, color, status, tdi):
    box_w, box_h = 45, 45
    start_pt = (pt[0]-box_w, pt[1]-box_h)
    end_pt = (pt[0]+box_w, pt[1]+box_h)
    cv2.rectangle(img, start_pt, end_pt, color, cv2.FILLED)
    cv2.rectangle(img, start_pt, end_pt, (255,255,255), 2)
    cv2.putText(img, f"{status} ({tdi}%)", (start_pt[0], start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def _draw_normal_marker(img, pt, temp):
    cv2.circle(img, pt, 6, (0, 255, 0), -1)
    cv2.putText(img, str(temp), (pt[0]-15, pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)