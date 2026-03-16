import cv2
import numpy as np
import math

def calculate_tdi(temp_l: float, temp_r: float) -> float:
    """
    Kalkuliert den Thermal Divergence Index (TDI) als standardisierte, prozentuale Abweichung.
    """
    diff = abs(temp_l - temp_r)
    return round(float((diff / 255.0) * 100.0), 2)

def analyze_hotspot_comprehensive(gray_img: np.ndarray, foot_mask: np.ndarray, pt: tuple, max_temp: float) -> dict:
    """
    Extrahiert ein erschöpfendes Spektrum klinischer, statistischer und morphologischer Metriken 
    innerhalb eines definierten thermalen Radius.
    """
    x, y = pt
    r = 20
    x_s, x_e = max(0, x - r), min(gray_img.shape[1], x + r)
    y_s, y_e = max(0, y - r), min(gray_img.shape[0], y + r)
    
    roi_gray = gray_img[y_s:y_e, x_s:x_e]
    roi_mask = foot_mask[y_s:y_e, x_s:x_e]
    
    if roi_gray.size == 0 or np.count_nonzero(roi_mask) == 0:
        return _generate_empty_metrics()

    valid_pixels = roi_gray[roi_mask > 0]
    
    if valid_pixels.size == 0:
        return _generate_empty_metrics()

    # 1. Deskriptive Thermostatistik
    mean_temp = float(np.mean(valid_pixels))
    median_temp = float(np.median(valid_pixels))
    std_dev = float(np.std(valid_pixels))
    variance = float(np.var(valid_pixels))
    
    percentile_25 = float(np.percentile(valid_pixels, 25))
    percentile_75 = float(np.percentile(valid_pixels, 75))
    iqr_temp = percentile_75 - percentile_25

    # 2. Morphologische Isothermen-Analyse
    threshold_temp = max_temp * 0.90
    _, hot_mask = cv2.threshold(roi_gray, threshold_temp, 255, cv2.THRESH_BINARY)
    hot_mask = cv2.bitwise_and(hot_mask, hot_mask, mask=roi_mask)
    
    area_px = cv2.countNonZero(hot_mask)
    total_valid_area = valid_pixels.size
    isothermal_ratio = float(area_px / total_valid_area) if total_valid_area > 0 else 0.0

    # 3. Thermischer Gradient (Peripherie vs. Zentrum)
    top_border = roi_gray[0, :].tolist() if roi_gray.shape[0] > 0 else []
    bottom_border = roi_gray[-1, :].tolist() if roi_gray.shape[0] > 1 else []
    left_border = roi_gray[:, 0].tolist() if roi_gray.shape[1] > 0 else []
    right_border = roi_gray[:, -1].tolist() if roi_gray.shape[1] > 1 else []
    
    border_pixels = top_border + bottom_border + left_border + right_border
    mean_border_temp = float(np.mean(border_pixels)) if border_pixels else 0.0
    gradient = float((max_temp - mean_border_temp) / r) if r > 0 else 0.0

    # 4. Geometrische Dezentralisierung und Zirkularität
    contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shift = 0.0
    circularity = 0.0
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter > 0:
            circularity = 4 * math.pi * (area_px / (perimeter * perimeter))
            
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            shift = math.sqrt((cx - r)**2 + (cy - r)**2)

    return {
        "thermo_statistics": {
            "max_temp": round(float(max_temp), 2),
            "mean_temp": round(mean_temp, 2),
            "median_temp": round(median_temp, 2),
            "std_dev": round(std_dev, 2),
            "variance": round(variance, 2),
            "iqr": round(iqr_temp, 2)
        },
        "morphology": {
            "area_px": int(area_px),
            "isothermal_ratio": round(isothermal_ratio, 4),
            "circularity": round(circularity, 4),
            "centroid_shift_px": round(shift, 2)
        },
        "dynamics": {
            "thermal_gradient": round(gradient, 4)
        }
    }

def _generate_empty_metrics() -> dict:
    """Generiert eine standardisierte Null-Datenstruktur bei Evaluierungsversagen."""
    return {
        "thermo_statistics": {"max_temp": 0.0, "mean_temp": 0.0, "median_temp": 0.0, "std_dev": 0.0, "variance": 0.0, "iqr": 0.0},
        "morphology": {"area_px": 0, "isothermal_ratio": 0.0, "circularity": 0.0, "centroid_shift_px": 0.0},
        "dynamics": {"thermal_gradient": 0.0}
    }

def perform_deep_analysis(left_toes: list, right_toes: list, gray_img: np.ndarray, foot_mask: np.ndarray, warn_th: float, severe_th: float) -> dict:
    """
    Initiiert die vollumfängliche, datengetriebene analytische Evaluation.
    """
    if len(left_toes) != 5 or len(right_toes) != 5:
        return {"error": "Incomplete coordinate mapping. Segmentation algorithms failed.", "global_metrics": {}, "regional_metrics": []}

    right_toes_matched = list(reversed(right_toes))
    detailed_results = []

    avg_l = sum([t["temp"] for t in left_toes]) / 5.0
    avg_r = sum([t["temp"] for t in right_toes_matched]) / 5.0
    fai_index = calculate_tdi(avg_l, avg_r)

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        t_l = l_data["temp"]
        t_r = r_data["temp"]
        tdi = calculate_tdi(t_l, t_r)
        
        metrics_l = analyze_hotspot_comprehensive(gray_img, foot_mask, l_data["sensor"], t_l)
        metrics_r = analyze_hotspot_comprehensive(gray_img, foot_mask, r_data["sensor"], t_r)

        is_left_hotter = t_l > t_r
        diff = abs(t_l - t_r)

        status = "PHYSIOLOGISCH"
        if tdi >= severe_th:
            status = "PATHOLOGISCH_SCHWER"
        elif tdi >= warn_th:
            status = "PATHOLOGISCH_VERDACHT"

        result_dict = {
            "anatomical_index": i,
            "status": status,
            "bilateral_comparisons": {
                "tdi_percentage": tdi,
                "absolute_diff": round(float(diff), 2),
                "is_left_dominant": is_left_hotter
            },
            "left_hemisphere": {
                "coordinates": l_data["sensor"],
                "metrics": metrics_l
            },
            "right_hemisphere": {
                "coordinates": r_data["sensor"],
                "metrics": metrics_r
            }
        }
        detailed_results.append(result_dict)

    return {
        "global_metrics": {
            "foot_asymmetry_index_percentage": fai_index,
            "mean_temperature_left": round(float(avg_l), 2),
            "mean_temperature_right": round(float(avg_r), 2)
        },
        "regional_metrics": detailed_results
    }