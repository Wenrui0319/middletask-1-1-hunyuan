# Region-Specific 3D Generation System

An interactive web application for image segmentation using Segment Anything Model (SAM) with region-specific 3D generation capabilities.

## Features

- **Interactive Segmentation**: Three selection modes (Point, Box, Polygon)
- **Adaptive Visualization**: Smart texture overlay that adjusts to object colors
- **Neumorphic UI**: Clean, modern interface with smooth interactions
- **Real-time Preview**: Instant segmentation feedback
- **Export Options**: Download masks and segmentation results

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd region_3d_system

# Install dependencies with uv
uv sync
```

## Usage

```bash
# Start the server
uv run python run_server.py

# Or directly
python run_server.py
```

Then open your browser and navigate to `http://localhost:8000`

## Project Structure

```
region_3d_system/
├── backend/          # FastAPI server
├── frontend/         # Web interface (HTML/JS/CSS)
├── src/             
│   └── segmentation/ # SAM integration
├── tests/           # Test scripts
└── run_server.py    # Server launcher
```

## Technologies

- **Backend**: FastAPI, Python
- **Frontend**: Vanilla JavaScript, Neumorphic CSS
- **AI Model**: Segment Anything Model (SAM)
- **Future**: Hunyuan3D integration for 3D generation

## License

MIT
