# Digital Dojo - System Architecture

**Version:** 0.1 (Prototype)
**Purpose:** Prose-driven 3D scene authoring tool that converts natural language descriptions into Gaussian splat scenes.

---

## Overview

Digital Dojo enables users to describe 3D scenes in natural language. The system parses these descriptions into a scene graph, generates 2D images for each object, segments them, and converts them to 3D Gaussian splats that can be composed into a viewable scene.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│                    "A red apple sits on a wooden table"                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OLLAMA (LLM Parser)                                  │
│                      Extracts structured scene graph                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SCENE GRAPH                                       │
│   { nodes: [{name: "apple", position: [0,1,0], prompt: "red apple..."}] }   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐
              │  Node 1  │   │  Node 2  │   │  Node N  │
              │  (apple) │   │  (table) │   │   ...    │
              └────┬─────┘   └────┬─────┘   └────┬─────┘
                   │              │              │
                   └──────────────┼──────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GPU PIPELINE (per object)                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │   SD3.5     │ →  │     SAM     │ →  │   SAM3D     │                      │
│  │  (image)    │    │   (mask)    │    │   (splat)   │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SPLAT MERGER                                        │
│              Combines splats with scene graph transforms                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SUPERSPLAT VIEWER                                    │
│                     WebGL Gaussian Splat Renderer                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Flask Web Server (`app.py`)

The main entry point. Provides REST API and serves the web interface.

**Key Routes:**

| Route | Method | Purpose |
|-------|--------|---------|
| `/workspace` | GET | Main dual-pane UI (prose editor + viewer) |
| `/api/parse` | POST | Parse natural language → scene graph |
| `/api/generate` | POST | Queue generation job for a node |
| `/api/scenes/<id>/compose` | GET | Get scene with resolved splat paths |
| `/api/scenes/<id>/merge` | POST | Merge all scene splats into one file |
| `/api/generate/sync` | POST | Synchronous generation (testing only) |
| `/media/<path>` | GET | Serve generated assets (.ply, .png) |

**Lazy Service Initialization:**
```python
def get_db():       # MongoDB connection
def get_ollama():   # LLM client
def get_parser():   # Scene parser (uses Ollama)
def get_orchestrator():  # GPU model manager
```

---

### 2. Services Module (`services/`)

#### 2.1 Ollama Client (`ollama_client.py`)

Communicates with local Ollama server for LLM inference.

**Key Methods:**
- `extract_json(text)` - Parse natural language into scene graph JSON
- `generate(prompt)` - Text completion
- `generate_with_image(prompt, image_path)` - Vision model support
- `analyze_scene_from_image(image_path)` - Extract scene graph from image

**Configuration:**
```bash
OLLAMA_URL=http://localhost:11434  # Ollama server
OLLAMA_MODEL=llama3.1              # Model to use (llama3.1, qwen3, etc.)
```

#### 2.2 Scene Parser (`scene_parser.py`)

Converts natural language to structured scene graphs.

**Data Structures:**
```python
@dataclass
class Scene:
    id: str
    name: str
    description: str        # Original prose
    root: SceneNode         # Scene graph root

@dataclass
class SceneNode:
    id: str
    name: str
    transform: Transform    # position, rotation, scale
    generator: Generator    # AI generation config
    children: List[SceneNode]

@dataclass
class Transform:
    position: [x, y, z]     # north=+z, east=+x, up=+y
    rotation: [rx, ry, rz]  # Euler angles (degrees)
    scale: [sx, sy, sz]

@dataclass
class Generator:
    type: str = "splat"     # splat or glb
    prompt: str             # Generation prompt
    status: str             # pending, generating, completed, failed
    output_path: str        # Path to generated asset
```

#### 2.3 Database Service (`database.py`)

MongoDB interface for persistent storage.

**Collections:**
- `scenes` - Scene graphs with metadata
- `jobs` - Generation job queue
- `assets` - Generated splat/image registry

**Job Statuses:**
```
pending → needs_mask → needs_splat → completed
   ↓          ↓            ↓
 failed     failed       failed
```

**Configuration:**
```bash
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=digital_dojo
```

#### 2.4 GPU Orchestrator (`orchestrator.py`)

Manages GPU resources and ML model lifecycle. Only one heavy model loaded at a time.

**Model Types:**
```python
class ModelType(Enum):
    NONE = "none"
    STABLE_DIFFUSION = "stable_diffusion"  # SD3.5
    SAM = "sam"                             # Segment Anything
    SAM3D = "sam3d"                         # Gaussian splat generation
```

**Pipeline Flow:**
```
1. generate_image(prompt)     # SD3.5 - creates isolated object on white bg
2. segment_with_sam(image)    # SAM - center-point mask extraction
3. generate_splat(image, mask) # SAM3D - 3D Gaussian splat
```

**Key Methods:**
- `load_stable_diffusion()` / `clear_gpu_memory()` - Model management
- `load_sam()` / `segment_with_sam()` / `unload_sam()` - Masking
- `load_sam3d()` / `generate_splat()` - Splat generation
- `run_pipeline(job)` - Full generation pipeline

**VRAM Management:**
- Models loaded on-demand, unloaded after use
- SAM runs alongside other models (smaller footprint)
- Aggressive memory clearing between stages

#### 2.5 Splat Merger (`splat_merger.py`)

Combines multiple Gaussian splat PLY files into a single scene.

**Key Functions:**
```python
load_ply(path) → SplatData           # Load splat from file
transform_splat(splat, pos, rot, scale)  # Apply scene transforms
merge_splats([splat1, splat2, ...])  # Concatenate splat data
save_ply(splat, path)                # Write merged result
merge_scene_splats(configs, output)  # High-level merge API
```

**PLY Format (Gaussian Splats):**
```
- position: x, y, z
- normal: nx, ny, nz
- SH coefficients: f_dc_0..2, f_rest_0..44 (spherical harmonics)
- opacity: opacity
- scale: scale_0, scale_1, scale_2
- rotation: rot_0..3 (quaternion)
```

---

### 3. Worker Process (`worker.py`)

Background job processor that polls MongoDB for pending jobs.

**Dynamic Model-Aware Processing (3 stages):**
```
1. Check for "pending" jobs → load SD3.5, process ALL, mark as "needs_mask"
2. Check for "needs_mask" jobs → load SAM, process ALL, mark as "needs_splat"
3. Check for "needs_splat" jobs → load SAM3D, process ALL, mark as "completed"
4. If no jobs at any stage → sleep and poll again
```

This minimizes GPU model switching by batching all jobs for each model before switching.
Each stage uses a different GPU model, so we process all jobs at one stage before moving on.

**Usage:**
```bash
python worker.py              # Continuous processing
python worker.py --once       # Process one batch and exit
python worker.py --batch-size 10  # Jobs per batch
```

---

### 4. Frontend

#### 4.1 Workspace (`templates/workspace.html`)

Dual-pane interface:
- **Left:** Prose editor with "Parse Scene" and "Generate All" buttons
- **Right:** SuperSplat viewer (iframe)

#### 4.2 SuperSplat Viewer (`static/viewer/`)

PlayCanvas-based WebGL Gaussian splat renderer. Loaded as a separate application.

**URL Parameters:**
- `?content=/media/path.ply` - Load specific splat file
- `?noui` - Hide UI controls

---

## External Dependencies

### Required Local Services

| Service | Purpose | Default URL |
|---------|---------|-------------|
| MongoDB | Scene/job storage | `mongodb://localhost:27017` |
| Ollama | LLM inference | `http://localhost:11434` |

### Required ML Repositories

| Repo | Purpose | Default Path |
|------|---------|--------------|
| SD3.5 | Image generation | `c:/Users/david/Documents/GitHub/sd3.5` |
| SAM | Object segmentation | `c:/Users/david/Documents/GitHub/sam3` |
| SAM3D | Splat generation | `c:/Users/david/Documents/GitHub/sam-3d-objects` |

**Environment Variables:**
```bash
SD35_REPO_PATH=...      # Path to SD3.5 repo
SAM3D_REPO_PATH=...     # Path to SAM3D repo
SAM_CHECKPOINT=...      # Path to SAM weights (.pth)
SD35_MODEL=sd3.5_medium.safetensors  # SD3.5 model variant
```

### Python Dependencies

```
flask>=3.0.0          # Web server
pymongo>=4.6.0        # Database
python-dotenv>=1.0.0  # Environment
requests>=2.31.0      # HTTP client (Ollama)
pillow>=10.0.0        # Image processing
plyfile>=1.0.0        # PLY file handling
torch + torchvision   # ML framework (CUDA)
segment-anything      # SAM model
```

---

## Data Flow

### 1. Scene Parsing

```
User types prose → POST /api/parse
                 → SceneParser.parse(text)
                 → OllamaClient.extract_json()
                 → Scene object created
                 → Saved to MongoDB
                 → Returns scene graph JSON
```

### 2. Job Queue Generation

```
User clicks "Generate All" → For each node with generator:
                           → POST /api/generate
                           → Creates job in MongoDB (status: pending)
```

### 3. Worker Processing

```
Worker polls MongoDB for pending jobs
For each pending job:
  1. SD3.5: generate_image(prompt) → image.png
  2. Update job status → needs_splat

For each needs_splat job:
  1. SAM: segment_with_sam(image) → mask
  2. SAM3D: generate_splat(image, mask) → splat.ply
  3. Register asset in MongoDB
  4. Update job status → completed
```

### 4. Scene Viewing

```
User clicks "Load in Viewer" → GET /api/scenes/{id}/compose
                             → Resolves splat paths for all nodes
                             → Returns scene + splat list

Or: "Merge Scene" → POST /api/scenes/{id}/merge
                  → Combines all splats with transforms
                  → Returns merged.ply path
                  → Viewer loads merged file
```

---

## File Structure

```
V0.1/
├── app.py                    # Flask server, API routes
├── worker.py                 # Background job processor
├── requirements.txt          # Python dependencies
├── .env                      # Environment config (create from .env.example)
│
├── services/
│   ├── __init__.py          # Service exports
│   ├── ollama_client.py     # LLM communication
│   ├── scene_parser.py      # NL → scene graph
│   ├── database.py          # MongoDB interface
│   ├── orchestrator.py      # GPU/model management
│   └── splat_merger.py      # PLY merging utilities
│
├── templates/
│   ├── index.html           # Landing page
│   ├── workspace.html       # Main editor UI
│   ├── editor.html          # Standalone prose editor
│   └── chat.html            # Ollama debug interface
│
├── static/
│   └── viewer/              # SuperSplat viewer (PlayCanvas)
│       └── index.html
│
└── media/
    ├── splat.ply            # Default test splat
    └── generated/           # Output directory
        ├── sd_xxxxx/        # SD3.5 output images
        └── splat_xxxxx.ply  # Generated splats
```

---

## Setup Guide

### 1. Install System Dependencies

```bash
# MongoDB
# Download from https://www.mongodb.com/try/download/community

# Ollama
# Download from https://ollama.ai
ollama pull llama3.1   # or qwen3, etc.
```

### 2. Clone Required ML Repos

```bash
# SD3.5 (requires model weights)
git clone https://github.com/Stability-AI/sd3.5
# Download sd3.5_medium.safetensors to sd3.5/models/

# SAM
git clone https://github.com/facebookresearch/segment-anything sam3
# Download sam_vit_h_4b8939.pth to sam3/

# SAM3D
git clone https://github.com/example/sam-3d-objects
# Follow repo setup instructions
```

### 3. Install Python Dependencies

```bash
cd prototypes/V0.1
pip install -r requirements.txt

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Configure Environment

Create `.env`:
```bash
MONGODB_URI=mongodb://localhost:27017
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
SD35_REPO_PATH=c:/path/to/sd3.5
SAM3D_REPO_PATH=c:/path/to/sam-3d-objects
SAM_CHECKPOINT=c:/path/to/sam3/sam_vit_h_4b8939.pth
```

### 5. Run

```bash
# Terminal 1: Start MongoDB
mongod

# Terminal 2: Start Ollama
ollama serve

# Terminal 3: Start Flask
python app.py

# Terminal 4: Start Worker
python worker.py
```

Access at: http://localhost:5000/workspace

---

## Known Limitations

1. **Single GPU**: Orchestrator designed for single GPU, no distributed processing
2. **Sequential Pipeline**: Each object processed one at a time through the full pipeline
3. **Single Splat Viewer**: SuperSplat loads one .ply at a time; merged scene required for multi-object
4. **No Real-time Updates**: Viewer doesn't auto-refresh; manual reload required
5. **Memory Intensive**: SD3.5 + SAM3D require significant VRAM (~16GB recommended)

---

## Future Considerations

- Mesh output support (TripoSR, InstantMesh) as alternative to splats
- Real-time scene composition without merging
- Incremental scene updates (regenerate single objects)
- Multi-GPU support for parallel processing
- Streaming generation progress to frontend
