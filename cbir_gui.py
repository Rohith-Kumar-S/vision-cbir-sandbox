'''
    Rohith Kumar Senthil Kumar
    Spring 2024
    CS 5330
    
    GUI For the CBIR System built with tkinter.
'''


import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import os
from PIL import Image, ImageTk

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR GUI")
        self.root.geometry("690x800")
        self.photos = []
        self.paths = []
        self.selected_image_path = ""
        
        # Source display
        self.selected_image_frame = tk.Frame(root, bg="white")
        self.selected_image_frame.pack(pady=10)
        self.image_label = tk.Label(self.selected_image_frame, bg="white")
        self.image_label.pack()
        self.filename_label = tk.Label(self.selected_image_frame, text="", bg="white")
        self.filename_label.pack(pady=5)
        
        # Select image button
        self.select_btn = tk.Button(root, text="Select Query Image", 
                                     command=self.select_image)
        self.select_btn.pack(pady=5)
        
         # Feature dropdown
        row_frame = tk.Frame(root)
        row_frame.pack(padx=10, pady=10)
        
        tk.Label(row_frame, text="Color Feature Type:").grid(row=0, column=0, padx=5)
        self.color_feature_var = tk.StringVar()
        self.color_feature_dropdown = ttk.Combobox(row_frame, textvariable=self.color_feature_var,
                                             values=["None","center-patch", "rg", "rgb", "multi-rg"])
        self.color_feature_dropdown.current(0)
        self.color_feature_dropdown.grid(row=1, column=0, padx=5)
        
        tk.Label(row_frame, text="Texture Feature Type:").grid(row=0, column=1, padx=5)
        self.texture_feature_var = tk.StringVar()
        self.texture_feature_dropdown = ttk.Combobox(row_frame, textvariable=self.texture_feature_var,
                                             values=["None", "texture-grads", "texture-s5s5"])
        self.texture_feature_dropdown.current(0)
        self.texture_feature_dropdown.grid(row=1, column=1, padx=5)
        
        tk.Label(row_frame, text="DNN Feature Type:").grid(row=0, column=2, padx=5)
        self.dnn_feature_var = tk.StringVar()
        self.dnn_feature_dropdown = ttk.Combobox(row_frame, textvariable=self.dnn_feature_var,
                                             values=["None", "resnet"])
        self.dnn_feature_dropdown.current(0)
        self.dnn_feature_dropdown.grid(row=1, column=2, padx=5)
        
        # Weight sliders
        tk.Label(row_frame, text="Color Feature Weight:").grid(row=3, column=0, padx=5)
        self.color_weight = tk.Scale(row_frame, from_=0, to=1, resolution=0.05, 
                                   orient=tk.HORIZONTAL)
        self.color_weight.set(0)
        self.color_weight.grid(row=4, column=0, padx=5)
        
        tk.Label(row_frame, text="Texture Feature Weight:").grid(row=3, column=1, padx=5)
        self.texture_weight = tk.Scale(row_frame, from_=0, to=1, resolution=0.05, 
                                   orient=tk.HORIZONTAL)
        self.texture_weight.set(0)
        self.texture_weight.grid(row=4, column=1, padx=5)
        
        if self.texture_feature_var.get() != "None":
            self.texture_weight.set(1)
        
        tk.Label(row_frame, text="DNN Feature Weight:").grid(row=3, column=2, padx=5)
        self.dnn_weight = tk.Scale(row_frame, from_=0, to=1, resolution=0.05, 
                                   orient=tk.HORIZONTAL)
        self.dnn_weight.set(0)
        self.dnn_weight.grid(row=4, column=2, padx=5)
        
        if self.dnn_feature_var.get() != "None":
            self.dnn_weight.set(1)
        
        ttk.Label(row_frame, text="Direction").grid(row=5, column=0, padx=5)
        self.direction_var = tk.StringVar()
        self.direction_dropdown = ttk.Combobox(row_frame, textvariable=self.direction_var,
                                             values=["top", "bottom"])
        self.direction_dropdown.current(0)
        self.direction_dropdown.grid(row=6, column=0, padx=5)
        
        # Top-K selection
        tk.Label(row_frame, text="Number of results").grid(row=5, column=1, padx=5)
        self.resultsCount_var = tk.StringVar(value="4")
        self.resultsCount_entry = tk.Entry(row_frame, textvariable=self.resultsCount_var)
        self.resultsCount_entry.grid(row=6, column=1, padx=5)
        
        # Search button
        self.search_btn = tk.Button(root, text="Search Similar Images", 
                                     command=self.search_images, 
                                     bg="green", fg="white")
        self.search_btn.pack(pady=10)
        
        container = tk.Frame(root)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for displaying results with scrollbar
        self.canvas = tk.Canvas(container, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame inside canvas for images
        self.image_frame = tk.Frame(self.canvas, bg="white")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        

        self.image_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.query_image_path = None
        
    
    def on_frame_configure(self, event=None):
        """Update scroll region when frame size changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """Adjust frame width when canvas is resized"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def on_mousewheel(self, event):
        """Enable scrolling with mouse wheel"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def select_image(self):
        """Open file dialog to select query image and display it"""
        self.query_image_path = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if self.query_image_path:
            img = Image.open(self.query_image_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  
            self.selected_image_path = os.path.basename(self.query_image_path)
            self.filename_label.config(text=self.selected_image_path)
    
    def display_results(self):
        """Display the result images in a grid"""

        # display the result images in a grid with 4 cols.
        COLS = 4
        self.photos = []
        for idx, img_path in enumerate(self.paths):
            row = idx // COLS
            col = idx % COLS

            try:
                img = Image.open(img_path)
                img = img.resize((150, 150))
                photo = ImageTk.PhotoImage(img)
                self.photos.append(photo)

                # Image frame with image and filename
                cell = tk.Frame(self.image_frame, bg="white")
                cell.grid(row=row, column=col, padx=5, pady=5)
                label = tk.Label(cell, image=photo, bg="white")
                label.pack()
                name_label = tk.Label(cell, text=os.path.basename(img_path), bg="white")
                name_label.pack()

            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    
    
    def search_images(self):
        """Run the C++ CBIR system with the selected query image and display results"""
        
        # Call cpp executable
        cpp_executable = "build\Debug\CBIRSystem.exe"  
        
        # Build command with selected options
        cmd = [
            cpp_executable,
            self.query_image_path,
            self.direction_var.get(),
            self.resultsCount_var.get(),
            self.color_feature_var.get(),
        ]
        if self.texture_feature_var.get() != "Select a feature type":
            cmd.append(self.texture_feature_var.get())
        if self.dnn_feature_var.get() != "Select a feature type":
            cmd.append(self.dnn_feature_var.get())
        if self.color_weight.get() != 0.0 or self.texture_weight.get() != 0.0 \
            or self.dnn_weight.get() != 0.0:
            cmd.append(str(self.color_weight.get()))
            cmd.append(str(self.texture_weight.get()))
            cmd.append(str(self.dnn_weight.get()))
        
            
        print(f"Running command: {' '.join(cmd)}")
        dirname = os.getcwd()
        only_dnn = self.dnn_feature_var.get()!="None" and \
            self.texture_feature_var.get()=="None" and self.color_feature_var.get()=="None"
        try:
            # Execute CBIR cpp program
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                image_paths = result.stdout.strip().splitlines()
                self.paths = [
                    os.path.join(dirname, line.strip().split('|')[0].strip()) \
                        if not only_dnn \
                            else os.path.join(dirname,"olympus" ,line.strip(
                                ).split('|')[0].strip())
                             for line in image_paths]
                print(f"Received paths: {self.paths}")
                self.display_results()
            else:
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("Search timed out!")
        except Exception as e:
            print(f"Error: {str(e)}")
            

if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()