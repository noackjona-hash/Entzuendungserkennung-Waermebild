import cv2
import math

def find_both_feet(gray_img):
    _, mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return None, None
        
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    sorted_by_x = sorted(sorted_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    return sorted_by_x[0], sorted_by_x[1]

def extract_toes_from_contour(contour, gray_img):
    M = cv2.moments(contour)
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    hull = cv2.convexHull(contour)
    raw_tips = [(p[0][0], p[0][1]) for p in hull if p[0][1] < cy - 20]

    merged_tips = []
    for tip in raw_tips:
        is_new = True
        for i, mt in enumerate(merged_tips):
            if math.sqrt((tip[0]-mt[0])**2 + (tip[1]-mt[1])**2) < 85:
                is_new = False
                if tip[1] < mt[1]: merged_tips[i] = tip
                break
        if is_new: merged_tips.append(tip)

    merged_tips = sorted(merged_tips, key=lambda x: x[1])[:5]
    merged_tips = sorted(merged_tips, key=lambda x: x[0]) 

    tips_with_data = []
    for tip in merged_tips:
        x, y = tip
        x_start, x_end = max(0, x - 30), min(gray_img.shape[1], x + 30)
        y_start, y_end = y, min(gray_img.shape[0], y + 60)
        
        roi = gray_img[y_start:y_end, x_start:x_end]
        if roi.size > 0:
            _, max_val, _, max_loc_roi = cv2.minMaxLoc(roi)
            temp = int(max_val)
            meas_pt = (x_start + max_loc_roi[0], y_start + max_loc_roi[1])
        else:
            temp, meas_pt = 0, (x, y)
            
        tips_with_data.append({"tip": tip, "temp": temp, "sensor": meas_pt})
        
    return tips_with_data