# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Dojo V0.1 - A prose-driven 3D scene authoring tool. Users write natural language descriptions that are parsed into hierarchical scene graphs and rendered as 3D meshes.

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

# Run SD3.5 image generation worker
python worker.py

# Run library pipeline worker (SD3.5 + embeddings)
python library_worker.py

# Run Trellis2 3D mesh worker (requires trellis2 conda env)
conda activate trellis2
python trellis2_worker.py
```

## URLs

- `/` - Custom WebGL2 viewer
- `/workspace` - Prose editor + 3D viewer (main interface)
- `/editor` - Standalone prose editor
- `/supersplat` - SuperSplat viewer (no UI)
- `/media/<filename>` - Serves PLY/GLB files with CORS

## API Endpoints

- `GET /api/status` - Check Ollama/MongoDB availability
- `POST /api/parse` - Parse natural language → scene graph
- `GET /api/scenes` - List saved scenes
- `POST /api/generate` - Queue generation job
- `GET /api/jobs/<id>` - Check job status

## Architecture

```
User writes prose → Ollama extracts scene graph → MongoDB stores → SD3.5 + Trellis2 generates → Viewer renders
```

**3 independent processes, 2 conda environments:**
```
Terminal 1: Flask Server (sam3d-objects env)
  → Serves UI, monitors progress via MongoDB

Terminal 2: SD3.5 + Embedding Worker (sam3d-objects env)
  → pending → needs_3d (SD3.5 image generation)
  → needs_embedding → ready (CLIP embeddings)

Terminal 3: Trellis2 Worker (trellis2 env)
  → needs_3d → needs_embedding (library) or completed (scene)
```

**GPU coordination:** MongoDB-based lock (`gpu_lock` collection) ensures only one worker uses GPU at a time.

**Flask Backend (`app.py`):**
- Lazy-loaded services (graceful degradation if Ollama/MongoDB unavailable)
- REST API for parsing, scene management, job queue
- Environment config via `.env` file

**Services (`services/`):**
- `orchestrator.py` - SD3.5 image generation, VRAM management
- `ollama_client.py` - LLM/VLM integration, supports vision models (qwen3-vl) for image analysis
- `vlm_service.py` - VLM object description and identification (uses Ollama qwen3-vl)
- `sam_service.py` - SAM3 text-prompted segmentation + point refinement + compositing
- `scene_parser.py` - Scene/SceneNode dataclasses, bidirectional text↔scene conversion
- `database.py` - MongoDB collections for scenes, jobs, assets, libraries

**Workers:**
- `worker.py` - SD3.5 image generation for scene jobs (pending → needs_3d)
- `library_worker.py` - Library pipeline: SD3.5 images, approved→needs_3d transitions, CLIP embeddings
- `trellis2_worker.py` - Trellis2 3D mesh generation (needs_3d → completed/needs_embedding)
- `gpu_lock.py` - MongoDB-based distributed GPU lock for cross-environment coordination

**Workspace (`templates/workspace.html`):**
- Dual-pane: prose editor (left) + SuperSplat viewer (right)
- "Parse Scene" extracts structure via Ollama
- "Generate" queues jobs for SD3.5 + Trellis2 pipeline
- Status indicator shows service availability

**Prose Editor (`templates/editor.html`):**
- Monaco with serif font, no line numbers (prose-friendly)
- Debounced content change notifications
- Supports bidirectional highlighting for scene↔text sync

**GPU Pipeline (single RTX 4090):**
- Models loaded/unloaded sequentially to fit in 24GB VRAM
- Ollama runs separately (manages own memory via `keep_alive`)
- Pipeline: LLM parse → SD3.5 image → Trellis2 mesh (GLB)
- GPU lock prevents concurrent model loading across environments

**External Repos (integrated via sys.path):**
- `sd3.5/` - SD3.5 inference (`SD3Inferencer` class)
- `sam3/` - Meta SAM 3 text-prompted segmentation (`Sam3Processor`, `build_sam3_image_model`)
- `TRELLIS.2/` - Trellis2 image-to-3D pipeline (runs in separate conda env)

## Environment Variables

```
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=digital_dojo

# Ollama (vision model for text + image understanding)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:8b

# SD3.5 (image generation)
SD35_REPO_PATH=c:/Users/david/Documents/GitHub/sd3.5
SD35_MODEL=sd3.5_medium.safetensors

# SAM3 (Meta SAM 3, text-prompted segmentation)
SAM3_REPO_PATH=c:/Users/david/Documents/GitHub/sam3
SAM3_AUTO_UNLOAD_SECONDS=120

# Trellis2 (3D mesh generation, runs in separate conda env)
TRELLIS2_REPO_PATH=/mnt/g/GitHub/TRELLIS.2
TRELLIS2_MODEL=microsoft/TRELLIS.2-4B
TRELLIS2_PIPELINE_TYPE=1024_cascade

# GPU Lock
GPU_LOCK_TIMEOUT=300

# Flask
FLASK_PORT=5000
```

## State Machines

**Scene jobs** (2 stages):
```
pending --[SD3.5]--> needs_3d --[Trellis2]--> completed
```

**Library items** (with human review):
```
pending --[SD3.5]--> needs_review --[human]--> approved --[worker]--> needs_3d --[Trellis2]--> needs_embedding --[CLIP]--> ready
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
python library_worker.py        # Run continuously (SD3.5 + embeddings)
python trellis2_worker.py       # Run in trellis2 env (3D mesh generation)
```

**Search the library:**
```bash
curl -X POST http://localhost:5000/api/libraries/search \
  -H "Content-Type: application/json" \
  -d '{"query": "comfortable seating", "mode": "vector"}'
```

**Library Pipeline:**
```
Text list → SD3.5 (3 variants) → Human review → Trellis2 (GLB mesh) → CLIP (embeddings) → Vector search
```

**Collections:**
- `libraries` - Library metadata (name, status, item count)
- `library_items` - Individual items with embeddings for vector search
- `gpu_lock` - Distributed GPU mutex for cross-environment coordination

**CLIP Embeddings:**
- Uses OpenCLIP ViT-L-14 (768-dim embeddings)
- Text embedding: from description
- Image embedding: from SD3.5 output image
- Supports FAISS index or local brute-force fallback

## Capture Flow (Phone-First)

Capture real objects via phone camera → auto-describe → auto-mask → reconstruct 3D → library.

**Pipeline:**
```
Photo → VLM (qwen3-vl: describe + identify object) → SAM3 (text-prompted mask) → User review/refine → Composite on white → Trellis2 (GLB) → CLIP encode → Library
```

**Capture state machine:**
```
[upload] → review → needs_3d → reconstructed → ready
                        ↓
                     rejected (deleted)
```

**Endpoints:**
- `POST /api/capture/upload` - Upload photo, auto-describe (VLM), auto-mask (SAM3 text prompt)
- `POST /api/capture/refine-mask` - Refine mask with user touch points (SAM3 geometric prompts)
- `POST /api/capture/reconstruct` - Composite on white, send to Trellis2
- `GET /api/capture/<id>/status` - Poll for Trellis2 completion
- `POST /api/capture/<id>/approve` - CLIP encode, add to library
- `POST /api/capture/<id>/reject` - Delete item and files

**SAM3 API (services/sam_service.py):**
- `predict_text(item_id, image_path, text_prompt)` - Text-prompted segmentation
- `predict_point(item_id, image_path, points, text_prompt)` - Geometric refinement with optional text
- `composite_on_white(image_path, mask)` - Extract object onto white background

## Media Files

Place `.glb` (mesh) files in `media/` directory. Generated assets go to `media/generated/`.
