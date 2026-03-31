import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ThermalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wärmebild Analyse - Anatomical Anchor V13")
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

        # V11: Roter Kanal Trick für perfekte Trennung vom Hintergrund
        _, _, r_channel = cv2.split(self.original_img)
        blurred_r = cv2.GaussianBlur(r_channel, (11, 11), 0)
        otsu_val, _ = cv2.threshold(blurred_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = int(otsu_val * 0.8) # 80% für kalte Zehen
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
                        
                        # 1. Den absoluten Gipfel finden (Anker-Zeh)
                        best_idx = np.argmin(smoothed_y)
                        anchor_x = sorted_x[best_idx]
                        
                        # 2. Schwerpunkt berechnen, um Rechts/Links zu unterscheiden
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                        else:
                            center_x = x + (w / 2) # Notfall-Fallback
                            
                        # 3. Anatomisches Fenster berechnen (ca. 45% der Fußhöhe)
                        span = h * 0.45 
                        
                        if anchor_x < center_x:
                            # Rechter Fuß (Gipfel ist links, Zehen gehen nach rechts)
                            start_x = anchor_x - (span * 0.15) # Etwas Puffer nach außen
                            end_x = anchor_x + span
                        else:
                            # Linker Fuß (Gipfel ist rechts, Zehen gehen nach links)
                            start_x = anchor_x - span
                            end_x = anchor_x + (span * 0.15)
                            
                        # Fenstergrenzen an den echten Konturen clippen
                        start_x = max(start_x, sorted_x[0])
                        end_x = min(end_x, sorted_x[-1])
                        
                        # 4. Alle Punkte außerhalb dieses Fensters gnadenlos abschneiden
                        toe_x = []
                        toe_y = []
                        orig_toe_y = []
                        for i, sx in enumerate(sorted_x):
                            if start_x <= sx <= end_x:
                                toe_x.append(sx)
                                toe_y.append(smoothed_y[i])
                                orig_toe_y.append(boundary_y[i])
                                
                        # 5. Die gefilterte, saubere Zone in 5 gleichmäßige Sektoren teilen
                        if len(toe_x) > 10:
                            zone_width = toe_x[-1] - toe_x[0]
                            segment_width = zone_width / 5.0
                            
                            for i in range(5):
                                seg_min_x = toe_x[0] + i * segment_width
                                seg_max_x = toe_x[0] + (i + 1) * segment_width
                                
                                segment_indices = [idx for idx, sx in enumerate(toe_x) if seg_min_x <= sx <= seg_max_x]
                                
                                # Randbereiche der Convolution ausklammern
                                valid_indices = [idx for idx in segment_indices if 2 < idx < len(toe_y)-2]
                                
                                if valid_indices:
                                    # Finde das absolute Minimum (Höchster Punkt) in diesem Sektor
                                    local_best_idx = min(valid_indices, key=lambda idx: toe_y[idx])
                                    
                                    final_px = toe_x[local_best_idx]
                                    final_py = orig_toe_y[local_best_idx]
                                    
                                    # Punkt einzeichnen
                                    draw_p = (final_px, final_py + 4)
                                    cv2.circle(output_img, draw_p, 6, (0, 255, 0), -1)
                                    toes_marked += 1
        
        status_text = f"{feet_found} Füße gefunden. {toes_marked} Zehen präzise markiert."
        self.status_label.config(text=status_text, fg="#2ecc71")
        
        output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(output_rgb)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalAnalyzerApp(root)
    root.mainloop()