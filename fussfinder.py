import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ThermalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wärmebild Analyse - FLIR Red-Channel Tracker V11")
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

        # --- DER FLIR IRONBOW TRICK (V11) ---
        # Spalte das Bild in Blau, Grün und Rot auf. Wir nutzen NUR den Rot-Kanal.
        b_channel, g_channel, r_channel = cv2.split(self.original_img)
        
        # Den Rot-Kanal weichzeichnen, um Rauschen zu killen
        blurred_r = cv2.GaussianBlur(r_channel, (11, 11), 0)

        # Otsu's Methode berechnet den mathematisch perfekten Schwellenwert 
        # zwischen Fuß (Rot-Anteil hoch) und Hintergrund (Rot-Anteil null).
        otsu_val, _ = cv2.threshold(blurred_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Wir nehmen 80% des perfekten Wertes zur Sicherheit, um auch eisige Zehen zu fangen,
        # aber ohne den dunklen Hintergrund anzufassen.
        final_thresh = int(otsu_val * 0.8)
        _, thresh = cv2.threshold(blurred_r, final_thresh, 255, cv2.THRESH_BINARY)

        # Letzte Krümel im Hintergrund wegradieren und Löcher stopfen
        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        feet_found = 0
        toes_marked = 0
        
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in sorted_contours[:2]:
                # Ignoriere alles, was zu klein ist (Hintergrund-Artefakte)
                if cv2.contourArea(contour) > 8000:
                    feet_found += 1
                    cv2.drawContours(output_img, [contour], -1, (255, 0, 0), 2)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    toe_zone_limit = y + int(h * 0.25)
                    top_points = [p[0] for p in contour if p[0][1] < toe_zone_limit]
                    
                    if len(top_points) > 20:
                        top_boundary_map = {}
                        for px, py in top_points:
                            if px not in top_boundary_map or py < top_boundary_map[px]:
                                top_boundary_map[px] = py
                        
                        sorted_x = sorted(top_boundary_map.keys())
                        boundary_y = [top_boundary_map[x] for x in sorted_x]
                        
                        # Die V9 Convolution-Glättung beibehalten
                        window_size = 11 
                        conv_kernel = np.ones(window_size) / window_size
                        smoothed_y = np.convolve(boundary_y, conv_kernel, mode='same')
                        
                        # 5-Sektoren Anatomie-Zwang (aus V10)
                        min_x = sorted_x[0]
                        max_x = sorted_x[-1]
                        zone_width = max_x - min_x
                        segment_width = zone_width / 5.0
                        
                        for i in range(5):
                            seg_min_x = min_x + i * segment_width
                            seg_max_x = min_x + (i + 1) * segment_width
                            
                            segment_indices = [idx for idx, sx in enumerate(sorted_x) if seg_min_x <= sx <= seg_max_x]
                            valid_indices = [idx for idx in segment_indices if 5 < idx < len(smoothed_y)-5]
                            
                            if valid_indices:
                                best_idx = min(valid_indices, key=lambda idx: smoothed_y[idx])
                                peak_x = sorted_x[best_idx]
                                peak_y = boundary_y[best_idx]
                                
                                draw_p = (peak_x, peak_y + 4)
                                cv2.circle(output_img, draw_p, 6, (0, 255, 0), -1)
                                toes_marked += 1
        
        status_text = f"{feet_found} Füße gefunden. {toes_marked} Zehen erfasst."
        self.status_label.config(text=status_text, fg="#2ecc71")
        
        output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(output_rgb)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalAnalyzerApp(root)
    root.mainloop()