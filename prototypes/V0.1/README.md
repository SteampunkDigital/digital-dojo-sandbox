# Gaussian Splat Viewer - V0.1

A Flask-based web server for viewing 3D Gaussian Splats with two rendering options:
1. **Custom WebGL2 viewer** - Basic point-sprite renderer with depth sorting
2. **PlayCanvas SuperSplat** - High-performance WebGPU renderer

## Setup

### Prerequisites
- Python 3.8+ with Flask
- Node.js 18+
- A Gaussian splat PLY file

### Installation

```bash
# Install npm dependencies (extracts SuperSplat viewer)
npm install

# Place your splat.ply file in the media folder
cp /path/to/your/splat.ply media/

# Run the server (using sam3d-objects conda environment or similar)
python app.py
```

### Usage

Open your browser to:
- `http://localhost:5000/` - Custom WebGL viewer
- `http://localhost:5000/supersplat/ui` - SuperSplat viewer (recommended)

### Controls
- **Left click + drag**: Rotate camera
- **Right click + drag**: Pan camera
- **Scroll wheel**: Zoom in/out

## Project Structure

```
V0.1/
├── app.py              # Flask server
├── package.json        # npm dependencies
├── build-viewer.mjs    # Script to extract SuperSplat viewer
├── templates/
│   └── index.html      # Custom WebGL viewer
├── static/
│   └── viewer/         # SuperSplat viewer files (generated)
└── media/
    └── splat.ply       # Your Gaussian splat file (not in git)
```

## Notes

- The `media/splat.ply` file is not included in git (too large)
- SuperSplat viewer requires WebGPU support (Chrome 113+, Edge 113+)
- Custom viewer works in any WebGL2-capable browser
