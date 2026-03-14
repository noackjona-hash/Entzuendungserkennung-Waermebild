import cv2

def perform_bilateral_analysis(image, left_toes, right_toes):
    if len(left_toes) != 5 or len(right_toes) != 5:
        cv2.putText(image, "FEHLER: Konnte nicht alle 10 Zehen finden!", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return image

    right_toes_matched = list(reversed(right_toes))
    threshold_diff = 15 

    for i in range(5):
        l_data = left_toes[i]
        r_data = right_toes_matched[i]

        temp_l = l_data["temp"]
        temp_r = r_data["temp"]
        delta_t = abs(temp_l - temp_r)

        cv2.circle(image, l_data["sensor"], 3, (255, 255, 255), -1)
        cv2.circle(image, r_data["sensor"], 3, (255, 255, 255), -1)

        if delta_t > threshold_diff:
            if temp_l > temp_r:
                draw_hotspot(image, l_data, delta_t)
                draw_normal(image, r_data)
            else:
                draw_hotspot(image, r_data, delta_t)
                draw_normal(image, l_data)
        else:
            draw_normal(image, l_data)
            draw_normal(image, r_data)

    return image

def draw_hotspot(img, data, delta_t):
    tip = data["tip"]
    cv2.rectangle(img, (tip[0]-35, tip[1]-20), (tip[0]+35, tip[1]+60), (0, 0, 255), 2)
    cv2.putText(img, f"HOT (+{delta_t})", (tip[0]-35, tip[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def draw_normal(img, data):
    tip = data["tip"]
    cv2.circle(img, tip, 5, (0, 255, 0), -1)
    cv2.putText(img, str(data["temp"]), (tip[0]-15, tip[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)