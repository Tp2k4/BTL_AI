import tkinter as tk
from tkinter import filedialog

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    root.destroy()
    return file_path

