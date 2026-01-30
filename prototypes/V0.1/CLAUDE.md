# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Dojo V0.1 - A prose-driven 3D scene authoring tool. Users write natural language descriptions that are parsed into hierarchical scene graphs and rendered as Gaussian splats.

Part of the Digital Dojo initiative for asymmetric VR/AR experiences.

## Common Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies and build SuperSplat viewer
npm install

# Start MongoDB (required for scene storage)
mongod

# Start Ollama (required for text parsing)
ollama serve

# Run Flask server (default: http://127.0.0.1:5000)
python app.py
```

## URLs

- `/` - Custom WebGL2 viewer
- `/workspace` - Prose editor + 3D viewer (main interface)
- `/editor` - Standalone prose editor
- `/supersplat` - SuperSplat viewer (no UI)
- `/media/<filename>` - Serves PLY files with CORS

## API Endpoints

- `GET /api/status` - Check Ollama/MongoDB availability
- `POST /api/parse` - Parse natural language → scene graph
- `GET /api/scenes` - List saved scenes
- `POST /api/generate` - Queue generation job
- `GET /api/jobs/<id>` - Check job status

## Architecture

```
User writes prose → Ollama extracts scene graph → MongoDB stores → SD/SAM3D generates → Viewer renders
```

**Flask Backend (`app.py`):**
- Lazy-loaded services (graceful degradation if Ollama/MongoDB unavailable)
- REST API for parsing, scene management, job queue
- Environment config via `.env` file

**Services (`services/`):**
- `orchestrator.py` - GPU model swapping (SD ↔ SAM3D), VRAM management
- `ollama_client.py` - LLM/VLM integration, supports vision models (qwen2-vl) for image analysis
- `scene_parser.py` - Scene/SceneNode dataclasses, bidirectional text↔scene conversion
- `database.py` - MongoDB collections for scenes, jobs, assets

**Workspace (`templates/workspace.html`):**
- Dual-pane: prose editor (left) + SuperSplat viewer (right)
- "Parse Scene" extracts structure via Ollama
- "Generate" queues jobs for SD/SAM3D pipeline
- Status indicator shows service availability

**Prose Editor (`templates/editor.html`):**
- Monaco with serif font, no line numbers (prose-friendly)
- Debounced content change notifications
- Supports bidirectional highlighting for scene↔text sync

**GPU Pipeline (single RTX 4090):**
- Models loaded/unloaded sequentially to fit in 24GB VRAM
- Ollama runs separately (manages own memory via `keep_alive`)
- Pipeline: LLM parse → SD3.5 image → SAM (segmentation) → SAM3D (splat or mesh)
- Output format configurable via `OUTPUT_FORMAT` env var (splat = .ply, mesh = .glb)

**External Repos (integrated via sys.path):**
- `sd3.5/` - SD3.5 inference (`SD3Inferencer` class)
- `sam-3d-objects/` - SAM3D inference (`Inference` class from `notebook/inference.py`)

## Environment Variables

```
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=digital_dojo

# Ollama (vision model for text + image understanding)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:8b

# External ML Repos
SD35_REPO_PATH=c:/Users/david/Documents/GitHub/sd3.5
SD35_MODEL=sd3.5_medium.safetensors
SAM3D_REPO_PATH=c:/Users/david/Documents/GitHub/sam-3d-objects

# Output Format (splat or mesh)
OUTPUT_FORMAT=splat

# Flask
FLASK_PORT=5000
```

## Object Library System

Generate searchable catalogs of 3D objects from text descriptions.

**Create a library:**
```bash
curl -X POST http://localhost:5000/api/libraries \
  -H "Content-Type: application/json" \
  -d '{"name": "Furniture", "objects": "wooden chair\nred sofa\nblue vase"}'
```

**Process items:**
```bash
python library_worker.py        # Run continuously
python library_worker.py --once # Process available items and exit
```

**Search the library:**
```bash
curl -X POST http://localhost:5000/api/libraries/search \
  -H "Content-Type: application/json" \
  -d '{"query": "comfortable seating", "mode": "vector"}'
```

**Library Pipeline:**
```
Text list → SD3.5 (3 variants) → SAM (mask) → SAM3D (3D) → CLIP (embeddings) → Vector search
```

**Collections:**
- `libraries` - Library metadata (name, status, item count)
- `library_items` - Individual items with embeddings for vector search

**CLIP Embeddings:**
- Uses OpenCLIP ViT-L-14 (768-dim embeddings)
- Text embedding: from description
- Image embedding: from SD3.5 output image
- Supports MongoDB Atlas vector search or local brute-force fallback

## Media Files

Place `.ply` (splat) or `.glb` (mesh) files in `media/` directory. Generated assets go to `media/generated/`.
