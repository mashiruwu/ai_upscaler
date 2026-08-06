import torch
import numpy as np
import math
from PIL import Image
from rrdbnet import RRDBNet
import urllib.request
import os

class Upscaler:
    def __init__(self, device='mps'):
        self.device = torch.device(device)
        self.model = None
        self.scale = 4

    def load_model(self, model_name="RealESRGAN_x4plus"):
        # We will use the standard RealESRGAN x4 plus model
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_path = os.path.join(os.path.dirname(__file__), "RealESRGAN_x4plus.pth")
        
        if not os.path.exists(model_path):
            print(f"Downloading model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Download complete.")
            
        # RealESRGAN_x4plus has 64 features, 23 blocks
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        loadnet = torch.load(model_path, map_location='cpu')
        
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.model.load_state_dict(loadnet[keyname], strict=True)
        self.model.eval()
        self.model.to(self.device)

    def tiled_inference(self, img_np, tile_size=400, tile_pad=10, progress_callback=None):
        if self.model is None:
            self.load_model()
            
        img_tensor = torch.from_numpy(np.transpose(img_np[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        b, c, h, w = img_tensor.size()
        output_h, output_w = h * self.scale, w * self.scale
        output_tensor = torch.zeros((b, c, output_h, output_w), device=self.device)
        
        tiles_x = math.ceil(w / tile_size)
        tiles_y = math.ceil(h / tile_size)
        
        total_tiles = tiles_x * tiles_y
        current_tile = 0

        with torch.no_grad():
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Compute tile boundaries
                    start_x = x * tile_size
                    start_y = y * tile_size
                    end_x = min((x + 1) * tile_size, w)
                    end_y = min((y + 1) * tile_size, h)

                    # Add padding
                    pad_start_x = max(start_x - tile_pad, 0)
                    pad_start_y = max(start_y - tile_pad, 0)
                    pad_end_x = min(end_x + tile_pad, w)
                    pad_end_y = min(end_y + tile_pad, h)

                    # Extract tile
                    tile = img_tensor[:, :, pad_start_y:pad_end_y, pad_start_x:pad_end_x]
                    
                    # Inference
                    out_tile = self.model(tile)
                    
                    # Remove padding from output
                    out_pad_start_x = (start_x - pad_start_x) * self.scale
                    out_pad_start_y = (start_y - pad_start_y) * self.scale
                    out_pad_end_x = out_tile.size(3) - (pad_end_x - end_x) * self.scale
                    out_pad_end_y = out_tile.size(2) - (pad_end_y - end_y) * self.scale
                    
                    out_tile = out_tile[:, :, out_pad_start_y:out_pad_end_y, out_pad_start_x:out_pad_end_x]
                    
                    # Place tile in output tensor
                    out_start_x = start_x * self.scale
                    out_start_y = start_y * self.scale
                    out_end_x = end_x * self.scale
                    out_end_y = end_y * self.scale
                    
                    output_tensor[:, :, out_start_y:out_end_y, out_start_x:out_end_x] = out_tile
                    
                    current_tile += 1
                    if progress_callback:
                        progress_callback(current_tile / total_tiles)
        
        # Convert back to numpy
        output = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        return output
