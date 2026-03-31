import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V13).
    Optimiert für den In-Memory Einsatz auf Servern (ohne GUI/Tkinter).
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analysiert das Wärmebild und gibt eine Liste der erkannten Zehen zurück.
        Rückgabeformat: [{"name": "Rechter Fuß - Zeh 1", "punkt": (x, y)}, ...]
        """
        # V11: Roter Kanal Trick für perfekte Trennung vom Hintergrund
        _, _, r_channel = cv2.split(image)
        blurred_r = cv2.GaussianBlur(r_channel, (11, 11), 0)
        otsu_val, _ = cv2.threshold(blurred_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = int(otsu_val * 0.8) # 80% für kalte Zehen
        _, thresh = cv2.threshold(blurred_r, final_thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_points = []
        
        if not contours:
            return detected_points
            
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in sorted_contours[:2]:
            if cv2.contourArea(contour) > 8000:
                x, y, w, h = cv2.boundingRect(contour)
                toe_zone_limit = y + int(h * 0.35) # Großzügige Zone oben
                top_points = [p[0] for p in contour if p[0][1] < toe_zone_limit]
                
                if len(top_points) > 20:
                    top_boundary_map = {}
                    for px, py in top_points:
                        if px not in top_boundary_map or py < top_boundary_map[px]:
                            top_boundary_map[px] = py
                    
                    sorted_x = sorted(top_boundary_map.keys())
                    boundary_y = [top_boundary_map[x] for x in sorted_x]
                    
                    window_size = 11 
                    conv_kernel = np.ones(window_size) / window_size
                    smoothed_y = np.convolve(boundary_y, conv_kernel, mode='same')
                    
                    # --- V13: ANATOMICAL ANCHORING ---
                    best_idx = np.argmin(smoothed_y)
                    anchor_x = sorted_x[best_idx]
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                    else:
                        center_x = x + (w / 2)
                        
                    span = h * 0.45 
                    
                    if anchor_x < center_x:
                        # Rechter Fuß
                        start_x = anchor_x - (span * 0.15)
                        end_x = anchor_x + span
                        fuss_name = "Rechter Fuß"
                    else:
                        # Linker Fuß
                        start_x = anchor_x - span
                        end_x = anchor_x + (span * 0.15)
                        fuss_name = "Linker Fuß"
                        
                    start_x = max(start_x, sorted_x[0])
                    end_x = min(end_x, sorted_x[-1])
                    
                    toe_x = []
                    toe_y = []
                    orig_toe_y = []
                    for i, sx in enumerate(sorted_x):
                        if start_x <= sx <= end_x:
                            toe_x.append(sx)
                            toe_y.append(smoothed_y[i])
                            orig_toe_y.append(boundary_y[i])
                            
                    if len(toe_x) > 10:
                        zone_width = toe_x[-1] - toe_x[0]
                        segment_width = zone_width / 5.0
                        
                        toe_count = 1
                        for i in range(5):
                            seg_min_x = toe_x[0] + i * segment_width
                            seg_max_x = toe_x[0] + (i + 1) * segment_width
                            
                            segment_indices = [idx for idx, sx in enumerate(toe_x) if seg_min_x <= sx <= seg_max_x]
                            valid_indices = [idx for idx in segment_indices if 2 < idx < len(toe_y)-2]
                            
                            if valid_indices:
                                local_best_idx = min(valid_indices, key=lambda idx: toe_y[idx])
                                final_px = toe_x[local_best_idx]
                                final_py = orig_toe_y[local_best_idx]
                                
                                detected_points.append({
                                    "name": f"{fuss_name} - Zeh {toe_count}",
                                    "punkt": (int(final_px), int(final_py))
                                })
                                toe_count += 1
                                
        return detected_points