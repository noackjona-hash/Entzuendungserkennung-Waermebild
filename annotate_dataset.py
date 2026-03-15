import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import csv
import shutil
import time
import numpy as np

class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignite - Rapid KI Annotator")
        self.root.geometry("1000x800")
        self.root.configure(bg="#2b2b2b")

        self.image_paths = []
        self.current_img_idx = 0
        self.current_clicks = []
        
        self.dataset_dir = "dataset"
        self.img_dir = os.path.join(self.dataset_dir, "images")
        self.label_file = os.path.join(self.dataset_dir, "labels.csv")
        self.init_dataset()

        # UI Setup
        top_frame = tk.Frame(root, bg="#2b2b2b", pady=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_load = tk.Button(top_frame, text="Bilderordner / Mehrere Bilder wählen", font=("Arial", 12), bg="#007acc", fg="white", command=self.load_images)
        self.btn_load.pack()
        
        self.status_label = tk.Label(top_frame, text="Wähle Bilder aus, um zu starten.", font=("Arial", 14), bg="#2b2b2b", fg="#00ffcc")
        self.status_label.pack(pady=5)

        self.canvas = tk.Canvas(root, bg="#000000", cursor="crosshair")
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.current_cv_img = None
        self.tk_img = None

    def init_dataset(self):
        os.makedirs(self.img_dir, exist_ok=True)
        if not os.path.exists(self.label_file):
            with open(self.label_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = ["filename"]
                for side in ["L", "R"]:
                    for i in range(1, 6): header.extend([f"{side}_Zeh{i}_x", f"{side}_Zeh{i}_y"])
                writer.writerow(header)

    def load_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Bilder", "*.jpg *.jpeg *.png")])
        if files:
            self.image_paths = list(files)
            self.current_img_idx = 0
            self.load_current_image()

    def load_current_image(self):
        if self.current_img_idx >= len(self.image_paths):
            messagebox.showinfo("Fertig!", "Alle Bilder wurden erfolgreich annotiert!")
            self.canvas.delete("all")
            self.status_label.config(text="Alle Bilder verarbeitet. Du kannst das Tool schliessen.")
            return

        img_path = self.image_paths[self.current_img_idx]
        
        # Sicherer OpenCV Load fuer Umlaute
        file_bytes = np.fromfile(img_path, np.uint8)
        self.current_cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Bild skalieren falls zu gross
        h, w = self.current_cv_img.shape[:2]
        if h > 700:
            ratio = 700 / h
            self.current_cv_img = cv2.resize(self.current_cv_img, (int(w * ratio), 700))

        self.current_clicks = []
        self.update_display()
        
        progress = f"Bild {self.current_img_idx + 1} von {len(self.image_paths)}"
        self.status_label.config(text=f"{progress} - Klicke von LINKS nach RECHTS (Klick {len(self.current_clicks)}/10)")

    def on_click(self, event):
        if self.current_cv_img is None: return
        
        # Klick registrieren
        self.current_clicks.append((event.x, event.y))
        
        # Klick ins Bild zeichnen
        cv2.drawMarker(self.current_cv_img, (event.x, event.y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(self.current_cv_img, str(len(self.current_clicks)), (event.x+5, event.y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self.update_display()
        
        progress = f"Bild {self.current_img_idx + 1} von {len(self.image_paths)}"
        self.status_label.config(text=f"{progress} - Klicke von LINKS nach RECHTS (Klick {len(self.current_clicks)}/10)")

        # Wenn 10 Zehen markiert sind -> Speichern und naechstes Bild laden!
        if len(self.current_clicks) == 10:
            self.save_and_next()

    def save_and_next(self):
        orig_path = self.image_paths[self.current_img_idx]
        timestamp = int(time.time() * 100)
        ext = os.path.splitext(orig_path)[1]
        new_filename = f"train_{timestamp}{ext}"
        new_filepath = os.path.join(self.img_dir, new_filename)
        
        # Originalbild rüberkopieren
        shutil.copy(orig_path, new_filepath)
        
        # Daten in CSV schreiben
        row = [new_filename]
        for x, y in self.current_clicks:
            row.extend([x, y])
            
        with open(self.label_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        # Nächstes Bild!
        self.current_img_idx += 1
        self.load_current_image()

    def update_display(self):
        img_rgb = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotatorApp(root)
    root.mainloop()