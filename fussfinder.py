import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ThermalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wärmebild Analyse - Topographic Tracker V12")
        self.root.geometry("800x650")
        self.root.configure(bg="#2c3e50")

        header_frame = tk.Frame(root, bg="#2c3e50")
        header_frame.pack(pady=15)

        self.btn_load = tk.Button(
            header_frame, text="Wärmebild laden", 
            command=self.load_image, 
            font=("Arial", 12, "bold"), 
            bg="#e74c3c", fg="white", padx=10, pady=5
        )
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(
            header_frame, text="Warten auf Bild...", 
            font=("Arial", 11), bg="#2c3e50", fg="#bdc3c7"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(root, width=640, height=480, bg="#ecf0f1", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.image_path = None
        self.original_img = None

    def load_image(self):
        self.root.attributes('-topmost', True) 
        self.image_path = filedialog.askopenfilename(filetypes=[("Bilder", "*.png;*.jpg;*.jpeg;*.bmp")])
        self.root.attributes('-topmost', False) 
        
        if self.image_path:
            img_array = np.fromfile(self.image_path, np.uint8)
            self.original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if self.original_img is not None:
                self.original_img = cv2.resize(self.original_img, (640, 480))
                self.status_label.config(text="Bild geladen. Verarbeite...", fg="#3498db")
                self.root.update_idletasks()
                self.process_and_display()

    def process_and_display(self):
        if self.original_img is None:
            return
        
        output_img = self.original_img.copy()

        # Roter Kanal & Otsu-Threshold (aus V11) für perfekte Kanten
        _, _, r_channel = cv2.split(self.original_img)
        blurred_r = cv2.GaussianBlur(r_channel, (11, 11), 0)
        otsu_val, _ = cv2.threshold(blurred_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Toleranz beibehalten, um kalte Zehen zu erfassen
        final_thresh = int(otsu_val * 0.8)
        _, thresh = cv2.threshold(blurred_r, final_thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        feet_found = 0
        toes_marked = 0
        
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in sorted_contours[:2]:
                if cv2.contourArea(contour) > 8000:
                    feet_found += 1
                    cv2.drawContours(output_img, [contour], -1, (255, 0, 0), 2)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    toe_zone_limit = y + int(h * 0.30) # Zone für Zehen
                    top_points = [p[0] for p in contour if p[0][1] < toe_zone_limit]
                    
                    if len(top_points) > 20:
                        top_boundary_map = {}
                        for px, py in top_points:
                            if px not in top_boundary_map or py < top_boundary_map[px]:
                                top_boundary_map[px] = py
                        
                        sorted_x = sorted(top_boundary_map.keys())
                        boundary_y = [top_boundary_map[x] for x in sorted_x]
                        
                        # Glättung der Kontur (vermeidet Treppeneffekte)
                        window_size = 11 
                        conv_kernel = np.ones(window_size) / window_size
                        smoothed_y = np.convolve(boundary_y, conv_kernel, mode='same')
                        
                        # --- FIX: THE TOPOGRAPHIC WALK (Zehen-Massiv isolieren) ---
                        
                        # 1. Höchsten Punkt finden (Zeh-Spitze) -> Kleinstes Y
                        best_idx = np.argmin(smoothed_y)
                        peak_y = smoothed_y[best_idx]
                        
                        # 2. Abbruch-Kante definieren (ca. 18% der Fußhöhe nach unten)
                        # Wenn die Linie tiefer als das fällt, sind wir an der Seite des Fußes angekommen.
                        dropoff_limit = peak_y + (h * 0.18)
                        
                        # 3. Nach links laufen
                        left_bound = best_idx
                        while left_bound > 0:
                            if smoothed_y[left_bound] > dropoff_limit:
                                break # Absturz in die Schlucht -> STOP!
                            left_bound -= 1
                            
                        # 4. Nach rechts laufen
                        right_bound = best_idx
                        while right_bound < len(smoothed_y) - 1:
                            if smoothed_y[right_bound] > dropoff_limit:
                                break # Absturz in die Schlucht -> STOP!
                            right_bound += 1
                            
                        # 5. Nur noch die saubere, isolierte Zehen-Kuppe nutzen!
                        toe_x = sorted_x[left_bound:right_bound+1]
                        toe_y = smoothed_y[left_bound:right_bound+1]
                        orig_toe_y = boundary_y[left_bound:right_bound+1]
                        
                        if len(toe_x) > 10:
                            # Anatomy Forcer: Wir teilen nun diese PERFEKT isolierte Kuppe durch 5!
                            min_x = toe_x[0]
                            max_x = toe_x[-1]
                            zone_width = max_x - min_x
                            segment_width = zone_width / 5.0
                            
                            for i in range(5):
                                seg_min_x = min_x + i * segment_width
                                seg_max_x = min_x + (i + 1) * segment_width
                                
                                segment_indices = [idx for idx, sx in enumerate(toe_x) if seg_min_x <= sx <= seg_max_x]
                                
                                if segment_indices:
                                    # Finde das Minimum (höchster Punkt) in diesem der 5 Sektoren
                                    local_best_idx = min(segment_indices, key=lambda idx: toe_y[idx])
                                    
                                    final_px = toe_x[local_best_idx]
                                    final_py = orig_toe_y[local_best_idx]
                                    
                                    draw_p = (final_px, final_py + 4)
                                    cv2.circle(output_img, draw_p, 6, (0, 255, 0), -1)
                                    toes_marked += 1
        
        status_text = f"{feet_found} Füße gefunden. {toes_marked} Zehen präzise erfasst."
        self.status_label.config(text=status_text, fg="#2ecc71")
        
        output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(output_rgb)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalAnalyzerApp(root)
    root.mainloop()