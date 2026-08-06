# Mac AI Upscaler

A native, locally run AI Image Upscaler explicitly designed and optimized for macOS and Apple Silicon (M1/M2/M3/M4/M5). 

This tool uses the powerful [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model running locally on your Mac's GPU (via PyTorch's Metal Performance Shaders backend) to upscale images without compromising privacy or paying for cloud subscriptions. 

It features **Tiled Inference**, which intelligently slices high-resolution images into smaller tiles during processing, preventing the "Out Of Memory" crashes that commonly occur when upscaling on unified memory architectures.

## Features
- **Apple Silicon Optimized:** Leverages PyTorch `mps` (Metal) for GPU acceleration.
- **Tiled Inference:** Safely upscales huge images without crashing your Mac.
- **Texture Matching:** (Optional) LAB-color-space histogram matching to retain original film grain and color balance, preventing the "plastic" AI look.
- **Batch Processing:** Includes a script to automatically process entire directories of images.
- **PyQt6 GUI:** A clean, native-feeling graphical interface.

## Prerequisites
- macOS (Tested on Apple Silicon, but works on Intel Macs with some performance degradation).
- Python 3.9+

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd ai_upscaler
   ```

2. Set up a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install the MPS-enabled version of PyTorch.*

## Usage

### Graphical User Interface (GUI)
To launch the desktop application, run:
```bash
source venv/bin/activate
python main.py
```
*The first time you run this, it will automatically download the RealESRGAN model weights (~60MB).*

### Batch Processing
If you have folders full of images you want to upscale automatically, you can use the batch script. Open `batch_upscale.py`, modify the `dirs_to_process` array at the bottom with your folder paths, and run:
```bash
source venv/bin/activate
python batch_upscale.py
```
This script is smart enough to skip images it has already processed!

## Architecture
- `main.py` & `ui.py`: The PyQt6 graphical frontend.
- `upscaler.py`: The AI engine handling PyTorch `mps` execution and tiled inference logic.
- `rrdbnet.py`: The Residual-in-Residual Dense Block Network architecture required to load RealESRGAN weights.
- `processing.py`: Image post-processing utilities (downscaling and LAB texture matching using OpenCV).

## License
MIT License
