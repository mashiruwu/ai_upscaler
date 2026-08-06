import sys
import os
import cv2
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QCheckBox, QProgressBar, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from upscaler import Upscaler
from processing import downscale_image, match_texture

class InferenceThread(QThread):
    progress_signal = pyqtSignal(float)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, image_path, upscaler, downscale_option, texture_option):
        super().__init__()
        self.image_path = image_path
        self.upscaler = upscaler
        self.downscale_option = downscale_option
        self.texture_option = texture_option

    def run(self):
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                pil_img = Image.open(self.image_path).convert('RGB')
                img = np.array(pil_img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            img_normalized = img.astype(np.float32) / 255.0
            
            def progress_callback(val):
                self.progress_signal.emit(val)
                
            upscaled = self.upscaler.tiled_inference(
                img_normalized, 
                tile_size=400, 
                progress_callback=progress_callback
            )
            
            if self.downscale_option == "2x (from 4x)":
                upscaled = downscale_image(upscaled, factor=0.5)
                
            if self.texture_option:
                upscaled = match_texture(img, upscaled)
                
            self.finished_signal.emit(upscaled)
        except Exception as e:
            self.error_signal.emit(str(e))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Upscaler (MPS/Apple Silicon)")
        self.setMinimumSize(500, 400)
        
        self.upscaler = Upscaler(device='mps')
        self.image_path = None
        self.output_image_np = None
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Image AI Upscaler")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Select Image
        self.btn_select = QPushButton("1. Select Image")
        self.btn_select.clicked.connect(self.select_image)
        layout.addWidget(self.btn_select)
        
        self.lbl_image = QLabel("No image selected")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_image)
        
        # Select Model
        layout.addWidget(QLabel("2. Select Model:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["RealESRGAN_x4plus"])
        layout.addWidget(self.combo_model)
        
        # Options Layout
        opt_layout = QHBoxLayout()
        
        downscale_layout = QVBoxLayout()
        downscale_layout.addWidget(QLabel("3. Optional Downscale:"))
        self.combo_downscale = QComboBox()
        self.combo_downscale.addItems(["None", "2x (from 4x)"])
        downscale_layout.addWidget(self.combo_downscale)
        opt_layout.addLayout(downscale_layout)
        
        self.chk_texture = QCheckBox("4. Texture Matching")
        opt_layout.addWidget(self.chk_texture)
        
        layout.addLayout(opt_layout)
        
        # Process
        self.btn_process = QPushButton("5. Run Tiled MPS Inference")
        self.btn_process.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px;")
        self.btn_process.clicked.connect(self.process_image)
        layout.addWidget(self.btn_process)
        
        # Progress
        self.progressbar = QProgressBar()
        self.progressbar.setValue(0)
        layout.addWidget(self.progressbar)
        
        # Export
        self.btn_export = QPushButton("6. Export")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_image)
        layout.addWidget(self.btn_export)
        
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.png *.jpeg *.heic *.HEIC)")
        if file_path:
            self.image_path = file_path
            self.lbl_image.setText(os.path.basename(file_path))
            self.btn_export.setEnabled(False)
            
    def process_image(self):
        if not self.image_path:
            return
            
        self.btn_process.setEnabled(False)
        self.btn_process.setText("Processing...")
        self.progressbar.setValue(0)
        
        self.thread = InferenceThread(
            self.image_path, 
            self.upscaler, 
            self.combo_downscale.currentText(), 
            self.chk_texture.isChecked()
        )
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.on_process_complete)
        self.thread.error_signal.connect(self.on_process_error)
        self.thread.start()
        
    def update_progress(self, val):
        self.progressbar.setValue(int(val * 100))
        
    def on_process_complete(self, output_np):
        self.output_image_np = output_np
        self.btn_process.setEnabled(True)
        self.btn_process.setText("5. Run Tiled MPS Inference")
        self.btn_export.setEnabled(True)
        self.progressbar.setValue(100)
        
    def on_process_error(self, err_msg):
        print(f"Error: {err_msg}")
        self.btn_process.setEnabled(True)
        self.btn_process.setText("Error! Try again")
        self.progressbar.setValue(0)
        
    def export_image(self):
        if self.output_image_np is None:
            return
            
        default_name = "upscaled_" + os.path.basename(self.image_path)
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", default_name, "PNG (*.png);;JPEG (*.jpg)")
        
        if file_path:
            out_bgr = cv2.cvtColor(self.output_image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, out_bgr)
