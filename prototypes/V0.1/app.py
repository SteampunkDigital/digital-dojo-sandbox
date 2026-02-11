"""
Digital Dojo - Flask Server

A prose-driven 3D scene authoring tool.
Users write natural language descriptions that are parsed into scene graphs
and rendered as Gaussian splats.

Setup:
    pip install -r requirements.txt
    npm install
    python app.py

Requires:
    - MongoDB running locally (or configure MONGODB_URI in .env)
    - Ollama running locally (ollama serve)
"""

from flask import Flask, render_template, send_from_directory, send_file, redirect, request, jsonify
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_FOLDER = os.path.join(BASE_DIR, 'media')
VIEWER_FOLDER = os.path.join(BASE_DIR, 'static', 'viewer')

# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Initialize services (lazy loading to avoid startup errors if services unavailable)
_db = None
_ollama = None
_parser = None
_orchestrator = None


def get_db():
    """Get database service (lazy initialization)"""
    global _db
    if _db is None:
        from services import db
        try:
            db.connect()
            _db = db
        except Exception as e:
            logger.warning(f"MongoDB not available: {e}")
            return None
    return _db


def get_ollama():
    """Get Ollama client (lazy initialization)"""
    global _ollama
    if _ollama is None:
        from services import OllamaClient
        _ollama = OllamaClient(
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1")
        )
    return _ollama


def get_parser():
    """Get scene parser (lazy initialization)"""
    global _parser
    if _parser is None:
        from services import SceneParser
        _parser = SceneParser(get_ollama())
    return _parser


def get_orchestrator():
    """Get GPU orchestrator (lazy initialization)"""
    global _orchestrator
    if _orchestrator is None:
        from services import GPUOrchestrator
        _orchestrator = GPUOrchestrator(output_dir=os.path.join(MEDIA_FOLDER, 'generated'))
    return _orchestrator


# ============== Page Routes ==============

@app.route('/')
def index():
    """Landing page with viewer options"""
    return render_template('index.html')


@app.route('/workspace')
def workspace():
    """Dual-pane workspace with prose editor and scene viewer"""
    return render_template('workspace.html')


@app.route('/editor')
def editor():
    """Natural language scene editor"""
    return render_template('editor.html')


@app.route('/supersplat')
def supersplat_viewer():
    """SuperSplat viewer - redirects with content param"""
    return redirect('/static/viewer/index.html?content=/media/splat.ply&noui')


@app.route('/supersplat/ui')
def supersplat_viewer_with_ui():
    """SuperSplat viewer with UI controls"""
    return redirect('/static/viewer/index.html?content=/media/splat.ply')


# ============== API Routes ==============

@app.route('/chat')
def chat_page():
    """Simple chat interface for debugging Ollama connection"""
    return render_template('chat.html')


@app.route('/api/status')
def api_status():
    """Check status of all services"""
    ollama = get_ollama()
    db = get_db()

    return jsonify({
        "flask": True,
        "mongodb": db.is_connected if db else False,
        "ollama": ollama.is_available() if ollama else False,
        "ollama_models": ollama.list_models() if ollama and ollama.is_available() else [],
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1")
    })


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Simple chat endpoint for debugging Ollama"""
    data = request.get_json()
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "No message provided", "success": False}), 400

    ollama = get_ollama()
    if not ollama:
        return jsonify({"error": "Ollama client not available", "success": False}), 503

    try:
        logger.info(f"Chat request: {message[:50]}...")
        response = ollama.generate(message)
        logger.info(f"Chat response received ({len(response)} chars)")
        return jsonify({
            "response": response,
            "success": True
        })
    except TimeoutError as e:
        logger.error(f"Chat timeout: {e}")
        return jsonify({"error": f"Request timed out: {e}", "success": False}), 504
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/parse', methods=['POST'])
def api_parse():
    """
    Parse natural language text into a scene graph.

    Request body: { "text": "scene description..." }
    Response: { "scene": {...}, "success": true }
    """
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided", "success": False}), 400

    parser = get_parser()
    if not parser:
        return jsonify({"error": "Parser not available", "success": False}), 503

    try:
        scene = parser.parse(text)
        if scene:
            # Save to database if available
            db = get_db()
            if db:
                db.save_scene(scene.to_dict())

            return jsonify({
                "scene": scene.to_dict(),
                "success": True
            })
        else:
            return jsonify({"error": "Failed to parse scene", "success": False}), 400

    except Exception as e:
        logger.error(f"Parse error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/scenes', methods=['GET'])
def api_list_scenes():
    """List saved scenes"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "scenes": []})

    scenes = db.list_scenes(limit=50)
    # Convert ObjectId and datetime for JSON serialization
    for scene in scenes:
        scene['_id'] = str(scene['_id'])
        if 'created_at' in scene:
            scene['created_at'] = scene['created_at'].isoformat()
        if 'updated_at' in scene:
            scene['updated_at'] = scene['updated_at'].isoformat()

    return jsonify({"scenes": scenes, "success": True})


@app.route('/api/scenes/<scene_id>', methods=['GET'])
def api_get_scene(scene_id):
    """Get a specific scene"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    scene = db.get_scene(scene_id)
    if scene:
        scene['_id'] = str(scene['_id'])
        return jsonify({"scene": scene, "success": True})
    else:
        return jsonify({"error": "Scene not found"}), 404


@app.route('/api/scenes/<scene_id>/compose', methods=['GET'])
def api_compose_scene(scene_id):
    """
    Get a scene with all its generated splats resolved.
    Returns the scene graph with splat paths for rendering.

    Response: {
        "scene": {...},
        "splats": [
            {
                "node_id": "...",
                "name": "...",
                "path": "/media/generated/splat_xxx.ply",
                "transform": {"position": [x,y,z], "rotation": [...], "scale": [...]}
            }
        ],
        "success": true
    }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    scene = db.get_scene(scene_id)
    if not scene:
        return jsonify({"error": "Scene not found"}), 404

    # Get all assets for this scene
    assets = db.get_assets_for_scene(scene_id)
    asset_map = {a['node_id']: a for a in assets if a.get('node_id')}

    # Build splat list with transforms from scene graph
    splats = []

    def collect_splats(node, parent_transform=None):
        node_id = node.get('id')
        transform = node.get('transform', {})

        # Check if this node has a generated splat
        if node_id and node_id in asset_map:
            asset = asset_map[node_id]
            splat_path = asset.get('path', '')

            # Convert filesystem path to web path
            if 'media' in splat_path:
                if 'media/generated/' in splat_path:
                    web_path = '/media/' + splat_path.split('media/')[1]
                elif 'media\\generated\\' in splat_path:
                    web_path = '/media/' + splat_path.split('media\\')[1].replace('\\', '/')
                else:
                    web_path = splat_path
            else:
                web_path = splat_path

            splats.append({
                'node_id': node_id,
                'name': node.get('name', 'unnamed'),
                'path': web_path,
                'transform': {
                    'position': transform.get('position', [0, 0, 0]),
                    'rotation': transform.get('rotation', [0, 0, 0]),
                    'scale': transform.get('scale', [1, 1, 1])
                },
                'prompt': asset.get('metadata', {}).get('prompt', '')
            })

        # Recurse into children
        for child in node.get('children', []):
            collect_splats(child)

    if 'root' in scene:
        collect_splats(scene['root'])

    scene['_id'] = str(scene['_id'])

    return jsonify({
        "scene": scene,
        "splats": splats,
        "splat_count": len(splats),
        "success": True
    })


@app.route('/api/scenes/<scene_id>/merge', methods=['POST'])
def api_merge_scene(scene_id):
    """
    Merge all splats in a scene into a single .ply file with transforms applied.

    Returns the path to the merged splat file that can be viewed directly.

    Response: {
        "merged_path": "/media/generated/merged_xxx.ply",
        "splat_count": 3,
        "total_gaussians": 150000,
        "success": true
    }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    scene = db.get_scene(scene_id)
    if not scene:
        return jsonify({"error": "Scene not found"}), 404

    # Get all assets for this scene
    assets = db.get_assets_for_scene(scene_id)
    asset_map = {a['node_id']: a for a in assets if a.get('node_id')}

    if not asset_map:
        return jsonify({"error": "No splat assets found for this scene", "success": False}), 404

    # Build splat configs with transforms from scene graph
    splat_configs = []

    def collect_splat_configs(node):
        node_id = node.get('id')
        transform = node.get('transform', {})

        if node_id and node_id in asset_map:
            asset = asset_map[node_id]
            splat_path = asset.get('path', '')

            if splat_path and os.path.exists(splat_path):
                splat_configs.append({
                    'path': splat_path,
                    'transform': {
                        'position': transform.get('position', [0, 0, 0]),
                        'rotation': transform.get('rotation', [0, 0, 0]),
                        'scale': transform.get('scale', [1, 1, 1])
                    },
                    'name': node.get('name', 'unnamed')
                })

        for child in node.get('children', []):
            collect_splat_configs(child)

    if 'root' in scene:
        collect_splat_configs(scene['root'])

    if not splat_configs:
        return jsonify({"error": "No valid splat files found", "success": False}), 404

    try:
        from services.splat_merger import merge_scene_splats
        import uuid

        # Generate output path
        output_filename = f"merged_{scene_id[:8]}_{uuid.uuid4().hex[:6]}.ply"
        output_path = os.path.join(MEDIA_FOLDER, 'generated', output_filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Merge splats
        merge_scene_splats(splat_configs, output_path)

        # Get file size for info
        file_size = os.path.getsize(output_path)

        # Register as asset
        asset_id = db.register_asset(
            asset_type="merged_splat",
            path=output_path,
            scene_id=scene_id,
            node_id="merged",
            metadata={
                "source_splats": len(splat_configs),
                "source_nodes": [c['name'] for c in splat_configs]
            }
        )

        # Web path for viewer
        web_path = f"/media/generated/{output_filename}"

        return jsonify({
            "merged_path": web_path,
            "file_path": output_path,
            "splat_count": len(splat_configs),
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "asset_id": asset_id,
            "success": True
        })

    except ImportError as e:
        logger.error(f"Splat merger import error: {e}")
        return jsonify({"error": f"plyfile not installed: {e}", "success": False}), 500
    except Exception as e:
        logger.error(f"Merge error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    Queue a generation job for a scene node.

    Request body: { "scene_id": "...", "node_id": "...", "prompt": "..." }
    Response: { "job_id": "...", "success": true }
    """
    data = request.get_json()
    scene_id = data.get('scene_id')
    node_id = data.get('node_id')
    prompt = data.get('prompt', '')

    if not all([scene_id, node_id, prompt]):
        return jsonify({"error": "Missing required fields", "success": False}), 400

    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    job_id = db.create_job(scene_id, node_id, prompt)

    return jsonify({
        "job_id": job_id,
        "success": True,
        "message": "Job queued for processing"
    })


@app.route('/api/jobs/<job_id>', methods=['GET'])
def api_job_status(job_id):
    """Get job status"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    job = db.get_job(job_id)
    if job:
        job['_id'] = str(job['_id'])
        return jsonify({"job": job, "success": True})
    else:
        return jsonify({"error": "Job not found"}), 404


@app.route('/api/jobs', methods=['GET'])
def api_list_jobs():
    """List jobs by status (default: pending)"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "jobs": []})

    status = request.args.get('status', 'pending')
    jobs = db.get_jobs_by_stage(status, limit=50)
    for job in jobs:
        job['_id'] = str(job['_id'])
        if 'created_at' in job:
            job['created_at'] = job['created_at'].isoformat()
        if 'completed_at' in job and job['completed_at']:
            job['completed_at'] = job['completed_at'].isoformat()

    return jsonify({"jobs": jobs, "count": len(jobs), "status": status, "success": True})


@app.route('/api/assets', methods=['GET'])
def api_list_assets():
    """List generated assets (splats, meshes, images)"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "assets": []})

    scene_id = request.args.get('scene_id')
    asset_type = request.args.get('type', 'all')  # Default to 'all' for both splat and mesh

    if scene_id:
        assets = db.get_assets_for_scene(scene_id)
    elif asset_type == 'all':
        # Get both splat and mesh assets (3D assets)
        assets = list(db.assets.find({"type": {"$in": ["splat", "mesh"]}}).sort("created_at", -1).limit(50))
    else:
        # Get all assets of the specified type
        assets = list(db.assets.find({"type": asset_type}).sort("created_at", -1).limit(50))

    # Filter to only include assets where the file actually exists
    valid_assets = []
    for asset in assets:
        file_path = asset.get('path')
        if file_path and os.path.exists(file_path):
            asset['_id'] = str(asset['_id'])
            if 'created_at' in asset:
                asset['created_at'] = asset['created_at'].isoformat()
            valid_assets.append(asset)

    return jsonify({"assets": valid_assets, "count": len(valid_assets), "success": True})


@app.route('/api/assets/<asset_id>/download')
def api_download_asset(asset_id):
    """Download an asset file"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    asset = db.assets.find_one({"_id": asset_id})
    if not asset:
        return jsonify({"error": "Asset not found"}), 404

    file_path = asset.get('path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Asset file not found"}), 404

    return send_file(file_path, as_attachment=False)


@app.route('/api/generate/sync', methods=['POST'])
def api_generate_sync():
    """
    Run image generation synchronously (for testing).
    This bypasses the job queue and runs immediately.

    WARNING: This blocks the request until generation completes.
    Only use for testing with small prompts.

    3D mesh generation is handled by the Trellis2 worker asynchronously.

    Request body: { "prompt": "..." }
    """
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided", "success": False}), 400

    orchestrator = get_orchestrator()
    if not orchestrator:
        return jsonify({"error": "Orchestrator not available", "success": False}), 503

    try:
        output_path = orchestrator.run_image_only(prompt)
        return jsonify({
            "image_path": output_path,
            "success": True
        })
    except Exception as e:
        logger.error(f"Sync generation error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/narrative', methods=['POST'])
def api_to_narrative():
    """
    Convert a scene graph back to natural language.
    Used for bidirectional editing.

    Request body: { "scene_id": "..." }
    Response: { "narrative": "...", "success": true }
    """
    data = request.get_json()
    scene_id = data.get('scene_id')

    if not scene_id:
        return jsonify({"error": "No scene_id provided", "success": False}), 400

    db = get_db()
    parser = get_parser()

    if not db or not parser:
        return jsonify({"error": "Services not available", "success": False}), 503

    scene_data = db.get_scene(scene_id)
    if not scene_data:
        return jsonify({"error": "Scene not found", "success": False}), 404

    from services import Scene
    scene = Scene.from_dict(scene_data)
    narrative = parser.scene_to_narrative(scene)

    return jsonify({
        "narrative": narrative,
        "success": True
    })


# ============== Library Routes ==============

@app.route('/api/libraries', methods=['POST'])
def api_create_library():
    """
    Create a new object library from a text list.

    Request body:
    {
        "name": "My Objects",
        "objects": "red apple\\nwooden chair\\nblue vase"
    }

    Response:
    {
        "library_id": "abc123",
        "items_created": 3,
        "jobs_queued": 9,
        "success": true
    }
    """
    data = request.get_json()
    name = data.get('name', 'Untitled Library')
    objects_text = data.get('objects', '')
    variants = int(data.get('variants', 3))

    if not objects_text.strip():
        return jsonify({"error": "No objects provided", "success": False}), 400

    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    try:
        library_id = db.create_library(name, objects_text, variants_per_item=variants)
        library = db.get_library(library_id)

        return jsonify({
            "library_id": library_id,
            "name": name,
            "items_created": library["item_count"],
            "jobs_queued": library["total_variants"],
            "success": True
        })
    except Exception as e:
        logger.error(f"Failed to create library: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries', methods=['GET'])
def api_list_libraries():
    """List all libraries"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "libraries": []})

    libraries = db.list_libraries()
    for lib in libraries:
        lib['_id'] = str(lib['_id'])
        if 'created_at' in lib:
            lib['created_at'] = lib['created_at'].isoformat()
        if 'completed_at' in lib and lib['completed_at']:
            lib['completed_at'] = lib['completed_at'].isoformat()

    return jsonify({"libraries": libraries, "count": len(libraries), "success": True})


@app.route('/api/libraries/<library_id>', methods=['GET'])
def api_get_library(library_id):
    """Get a library with its items"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available"}), 503

    library = db.get_library(library_id)
    if not library:
        return jsonify({"error": "Library not found"}), 404

    items = db.get_library_items(library_id)

    # Format for JSON
    library['_id'] = str(library['_id'])
    if 'created_at' in library:
        library['created_at'] = library['created_at'].isoformat()
    if 'completed_at' in library and library['completed_at']:
        library['completed_at'] = library['completed_at'].isoformat()

    for item in items:
        item['_id'] = str(item['_id'])
        if 'created_at' in item:
            item['created_at'] = item['created_at'].isoformat()
        # Remove embeddings from response (too large)
        item.pop('text_embedding', None)
        item.pop('image_embedding', None)

    # Count items by status
    status_counts = {}
    for item in items:
        status = item.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1

    return jsonify({
        "library": library,
        "items": items,
        "status_counts": status_counts,
        "success": True
    })


def _path_to_url(path):
    """Convert a filesystem path to a web URL."""
    if not path:
        return None

    # Normalize slashes to forward slashes
    path = path.replace("\\", "/")

    # Already a URL
    if path.startswith("/media/"):
        return path

    # Find the media/generated portion and extract relative path
    if "media/generated/" in path:
        idx = path.index("media/generated/")
        return "/" + path[idx:]

    # Path is just "generated/..." without media prefix
    if path.startswith("generated/"):
        return "/media/" + path

    # Path starts with sd_ or mesh_ directly (relative to generated/)
    if path.startswith("sd_") or path.startswith("mesh_") or path.startswith("splat_"):
        return "/media/generated/" + path

    # Fallback: assume it's relative to media
    return "/media/" + path


def _item_to_api_response(item):
    """
    Convert a library item to a clean API response format.
    Transforms filesystem paths to web URLs and removes internal fields.
    """
    result = {
        "id": str(item.get("_id", "")),
        "description": item.get("description", ""),
        "status": item.get("status", ""),
        "seed": item.get("seed"),
        "library_id": item.get("library_id", ""),
        "variant_group_id": item.get("variant_group_id", ""),
    }

    # Convert image path to web URL
    image_url = _path_to_url(item.get("image_path"))
    if image_url:
        result["image_url"] = image_url

    # Convert asset path to web URL
    asset_url = _path_to_url(item.get("asset_path"))
    if asset_url:
        result["asset_url"] = asset_url
        result["asset_type"] = item.get("asset_type", "mesh")

    # Include similarity score if present (from search results)
    if "similarity" in item:
        result["similarity"] = item["similarity"]

    return result


@app.route('/api/library/items/<item_id>', methods=['GET'])
def api_get_library_item(item_id):
    """
    Get a single library item by ID.

    Response:
    {
        "item": {
            "id": "abc123",
            "description": "red apple",
            "status": "ready",
            "image_url": "/media/generated/sd_xxx/image.png",
            "asset_url": "/media/generated/mesh_xxx.glb",
            "asset_type": "mesh"
        },
        "success": true
    }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    return jsonify({
        "item": _item_to_api_response(item),
        "success": True
    })


@app.route('/api/libraries/search', methods=['POST'])
def api_search_libraries():
    """
    Search libraries using text, image, or combined vector similarity.

    Request body:
    {
        "query": "comfortable seating",      // Text query (optional if image provided)
        "image_url": "/media/...",           // Image URL for visual search (optional)
        "image_base64": "...",               // Base64-encoded image (optional)
        "mode": "vector",                    // "text", "vector", or "image"
        "embedding_field": "text_embedding", // "text_embedding" or "image_embedding"
        "use_faiss": true,                   // Use FAISS index for speed (default: true)
        "limit": 10,
        "library_id": "optional",
        "status": "ready"
    }

    Response:
    {
        "results": [
            {
                "id": "abc123",
                "description": "red sofa",
                "image_url": "/media/generated/...",
                "asset_url": "/media/generated/...",
                "asset_type": "mesh",
                "similarity": 0.85
            }
        ],
        "count": 1,
        "mode": "vector",
        "success": true
    }
    """
    data = request.get_json()
    query = data.get('query', '')
    image_url = data.get('image_url', '')
    image_base64 = data.get('image_base64', '')
    mode = data.get('mode', 'vector')
    embedding_field = data.get('embedding_field', 'text_embedding')
    use_faiss = data.get('use_faiss', True)
    limit = int(data.get('limit', 10))
    library_id = data.get('library_id')
    status_filter = data.get('status', 'ready')

    # Validate input
    has_text = bool(query.strip())
    has_image = bool(image_url or image_base64)

    if mode == 'text' and not has_text:
        return jsonify({"error": "No query provided for text search", "success": False}), 400
    if mode == 'image' and not has_image:
        return jsonify({"error": "No image provided for image search", "success": False}), 400
    if mode == 'vector' and not has_text and not has_image:
        return jsonify({"error": "No query or image provided", "success": False}), 400

    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    try:
        if mode == 'text':
            # Simple text search
            results = db.search_library_text(query, limit=limit, library_id=library_id)
        else:
            # Vector similarity search
            from services.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()

            query_embedding = None

            if has_image:
                # Image-based search
                embedding_field = 'image_embedding'
                query_embedding = _encode_image_query(embedding_service, image_url, image_base64)
            elif has_text:
                # Text-based search
                query_embedding = embedding_service.encode_text(query)

            if query_embedding is None:
                return jsonify({"error": "Failed to encode query", "success": False}), 500

            # Try FAISS if available and requested
            if use_faiss:
                try:
                    from services.faiss_service import get_faiss_service
                    faiss_svc = get_faiss_service()

                    # FAISS search returns {id, similarity}
                    faiss_results = faiss_svc.search(
                        query_embedding,
                        k=limit * 2,  # Over-fetch to account for status filtering
                        library_id=library_id,
                        embedding_field=embedding_field
                    )

                    # Fetch full items from database
                    item_ids = [r['id'] for r in faiss_results]
                    similarity_map = {r['id']: r['similarity'] for r in faiss_results}

                    results = []
                    for item_id in item_ids:
                        item = db.library_items.find_one({"_id": item_id})
                        if item:
                            item['similarity'] = similarity_map.get(item_id, 0)
                            results.append(item)

                except Exception as faiss_err:
                    logger.warning(f"FAISS search failed, falling back to brute force: {faiss_err}")
                    results = db.search_library_vector(
                        query_embedding,
                        embedding_field=embedding_field,
                        limit=limit,
                        library_id=library_id
                    )
            else:
                # Direct database search (brute force)
                results = db.search_library_vector(
                    query_embedding,
                    embedding_field=embedding_field,
                    limit=limit,
                    library_id=library_id
                )

        # Filter by status if specified
        if status_filter:
            results = [r for r in results if r.get('status') == status_filter]

        # Limit results
        results = results[:limit]

        # Convert to clean API format
        formatted_results = [_item_to_api_response(item) for item in results]

        return jsonify({
            "results": formatted_results,
            "count": len(formatted_results),
            "mode": mode,
            "embedding_field": embedding_field if mode != 'text' else None,
            "success": True
        })

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({"error": str(e), "success": False}), 500


def _encode_image_query(embedding_service, image_url: str, image_base64: str):
    """
    Encode an image query to embedding vector.
    Supports local file URLs and base64-encoded images.
    """
    import tempfile
    import base64

    if image_base64:
        # Decode base64 image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(base64.b64decode(image_base64))
            temp_path = f.name

        try:
            return embedding_service.encode_image(temp_path)
        finally:
            os.unlink(temp_path)

    elif image_url:
        # Handle local media URLs
        if image_url.startswith('/media/'):
            local_path = os.path.join(MEDIA_FOLDER, image_url[7:])
            if os.path.exists(local_path):
                return embedding_service.encode_image(local_path)
            else:
                raise ValueError(f"Image not found: {local_path}")
        else:
            # Download remote image
            import requests as req
            import tempfile

            response = req.get(image_url, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            try:
                return embedding_service.encode_image(temp_path)
            finally:
                os.unlink(temp_path)

    return None


# ============== Library Review Routes ==============

@app.route('/api/libraries/<library_id>/review', methods=['GET'])
def api_get_review_items(library_id):
    """
    Get items needing review, grouped by description.

    Query params:
        status: Filter by status (default: needs_review)

    Response:
    {
        "groups": [
            {
                "variant_group_id": "abc123",
                "description": "red apple",
                "variants": [...]
            }
        ],
        "total_pending": 15,
        "success": true
    }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    status = request.args.get('status', 'needs_review')

    try:
        groups = db.get_items_for_review(library_id, status=status)

        # Format for JSON
        for group in groups:
            for variant in group.get('variants', []):
                variant['_id'] = str(variant['_id'])
                if 'created_at' in variant:
                    variant['created_at'] = variant['created_at'].isoformat()
                if 'reviewed_at' in variant and variant['reviewed_at']:
                    variant['reviewed_at'] = variant['reviewed_at'].isoformat()
                # Convert filesystem paths to web URLs
                if variant.get('image_path'):
                    variant['image_path'] = _path_to_url(variant['image_path'])
                if variant.get('asset_path'):
                    variant['asset_path'] = _path_to_url(variant['asset_path'])
                if variant.get('original_image_path'):
                    variant['original_image_path'] = _path_to_url(variant['original_image_path'])
                # Remove embeddings from response
                variant.pop('text_embedding', None)
                variant.pop('image_embedding', None)

        total_pending = db.count_library_items_by_stage("needs_review")

        return jsonify({
            "groups": groups,
            "total_pending": total_pending,
            "success": True
        })

    except Exception as e:
        logger.error(f"Failed to get review items: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/<library_id>/stats', methods=['GET'])
def api_library_stats(library_id):
    """
    Get processing statistics for a library.

    Response:
    {
        "pending": 0,
        "needs_review": 15,
        "approved": 3,
        ...
        "success": true
    }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    try:
        stats = db.get_library_stats(library_id)
        return jsonify({**stats, "success": True})

    except Exception as e:
        logger.error(f"Failed to get library stats: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/items/<item_id>/approve', methods=['POST'])
def api_approve_item(item_id):
    """
    Approve an item for continued processing.

    Request body (optional):
    { "reviewer": "name" }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json() or {}
    reviewer = data.get('reviewer')

    try:
        success = db.approve_item(item_id, reviewer=reviewer)
        if success:
            return jsonify({"status": "approved", "success": True})
        else:
            return jsonify({"error": "Item not found or not in needs_review status", "success": False}), 404

    except Exception as e:
        logger.error(f"Failed to approve item: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/items/<item_id>/reject', methods=['POST'])
def api_reject_item(item_id):
    """
    Reject an item.

    Request body (optional):
    { "reviewer": "name", "notes": "reason" }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json() or {}
    reviewer = data.get('reviewer')
    notes = data.get('notes')

    try:
        success = db.reject_item(item_id, reviewer=reviewer, notes=notes)
        if success:
            return jsonify({"status": "rejected", "success": True})
        else:
            return jsonify({"error": "Item not found", "success": False}), 404

    except Exception as e:
        logger.error(f"Failed to reject item: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/items/<item_id>/reset', methods=['POST'])
def api_reset_item(item_id):
    """
    Reset a rejected or failed item for regeneration.
    Clears image/asset and assigns a new seed.
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json() or {}
    new_seed = data.get('new_seed', True)

    try:
        success = db.reset_item_for_regeneration(item_id, new_seed=new_seed)
        if success:
            return jsonify({"status": "pending", "success": True})
        else:
            return jsonify({"error": "Item not found", "success": False}), 404

    except Exception as e:
        logger.error(f"Failed to reset item: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/items/<item_id>/retry', methods=['POST'])
def api_retry_item(item_id):
    """
    Retry a stuck processing item (keeps image, resets to needs_3d).
    Use this for items stuck at needs_3d or needs_embedding.
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    try:
        success = db.retry_stuck_item(item_id)
        if success:
            return jsonify({"status": "approved", "success": True})
        else:
            return jsonify({"error": "Item not found or not in processing state", "success": False}), 404

    except Exception as e:
        logger.error(f"Failed to retry item: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/groups/<group_id>/reset', methods=['POST'])
def api_reset_group(group_id):
    """
    Reset all rejected/failed items in a group for regeneration.
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json() or {}
    new_seeds = data.get('new_seeds', True)

    try:
        count = db.reset_group_for_regeneration(group_id, new_seeds=new_seeds)
        return jsonify({"reset_count": count, "success": True})

    except Exception as e:
        logger.error(f"Failed to reset group: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/items/<item_id>/description', methods=['PUT'])
def api_update_description(item_id):
    """
    Update item description/prompt.

    Request body:
    { "description": "new prompt text" }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({"error": "description required", "success": False}), 400

    try:
        result = db.update_item_description(item_id, data['description'])
        return jsonify({**result, "success": True})

    except Exception as e:
        logger.error(f"Failed to update description: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/items/<item_id>/regenerate', methods=['POST'])
def api_regenerate_item(item_id):
    """
    Create a new variant with different seed.

    Request body (optional):
    { "seed": 12345 }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json() or {}
    seed = data.get('seed')

    try:
        new_item_id = db.create_variant(item_id, seed=seed)
        return jsonify({
            "new_item_id": new_item_id,
            "success": True
        })

    except ValueError as e:
        return jsonify({"error": str(e), "success": False}), 404
    except Exception as e:
        logger.error(f"Failed to create variant: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/items/<item_id>', methods=['DELETE'])
def api_delete_item(item_id):
    """Delete an item from the library."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    try:
        success = db.delete_item(item_id)
        if success:
            return jsonify({"deleted": True, "success": True})
        else:
            return jsonify({"error": "Item not found", "success": False}), 404

    except Exception as e:
        logger.error(f"Failed to delete item: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/groups/<group_id>', methods=['DELETE'])
def api_delete_group(group_id):
    """Delete all variants in a group."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    try:
        count = db.delete_group(group_id)
        return jsonify({
            "deleted_count": count,
            "success": True
        })

    except Exception as e:
        logger.error(f"Failed to delete group: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/groups/<group_id>/approve-all', methods=['POST'])
def api_approve_group(group_id):
    """Approve all variants in a group."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json() or {}
    reviewer = data.get('reviewer')

    try:
        count = db.approve_group(group_id, reviewer=reviewer)
        return jsonify({
            "approved_count": count,
            "success": True
        })

    except Exception as e:
        logger.error(f"Failed to approve group: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/groups/<group_id>/reject-all', methods=['POST'])
def api_reject_group(group_id):
    """Reject all variants in a group."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json() or {}
    reviewer = data.get('reviewer')
    notes = data.get('notes')

    try:
        count = db.reject_group(group_id, reviewer=reviewer, notes=notes)
        return jsonify({
            "rejected_count": count,
            "success": True
        })

    except Exception as e:
        logger.error(f"Failed to reject group: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/libraries/groups/<group_id>/description', methods=['PUT'])
def api_update_group_description(group_id):
    """Update description for all items in a variant group."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({"error": "Missing description", "success": False}), 400

    try:
        result = db.update_group_description(group_id, data['description'])
        if result.get("updated"):
            return jsonify({
                "updated_count": result.get("updated_count", 0),
                "success": True
            })
        else:
            return jsonify({"error": result.get("error", "Update failed"), "success": False}), 404

    except Exception as e:
        logger.error(f"Failed to update group description: {e}")
        return jsonify({"error": str(e), "success": False}), 500



# ============== Image Upload + SAM Masking ==============

_sam_service = None

def get_sam():
    """Lazy-load SAM service."""
    global _sam_service
    if _sam_service is None:
        from services.sam_service import get_sam_service
        _sam_service = get_sam_service()
    return _sam_service


@app.route('/api/libraries/<library_id>/upload', methods=['POST'])
def api_upload_images(library_id):
    """
    Upload images to a library for interactive SAM masking.

    Accepts multipart/form-data with 'images' files and optional 'descriptions' JSON.
    Creates library_items with status 'needs_masking'.
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    library = db.get_library(library_id)
    if not library:
        return jsonify({"error": "Library not found", "success": False}), 404

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No images uploaded", "success": False}), 400

    # Parse descriptions (JSON array or single string)
    import json as json_module
    descriptions_raw = request.form.get('descriptions', '[]')
    try:
        descriptions = json_module.loads(descriptions_raw)
        if isinstance(descriptions, str):
            descriptions = [descriptions]
    except (json_module.JSONDecodeError, TypeError):
        descriptions = []

    single_description = request.form.get('description', '')

    import uuid
    from PIL import Image as PILImage

    output_dir = os.path.join(os.path.dirname(__file__), 'media', 'generated')
    item_ids = []

    for i, file in enumerate(files):
        if not file.filename:
            continue

        # Determine description
        if i < len(descriptions) and descriptions[i]:
            desc = descriptions[i]
        elif single_description:
            desc = single_description
        else:
            # Use filename without extension
            desc = os.path.splitext(file.filename)[0].replace('_', ' ').replace('-', ' ')

        # Create upload directory
        upload_id = uuid.uuid4().hex[:8]
        upload_dir = os.path.join(output_dir, f"upload_{upload_id}")
        os.makedirs(upload_dir, exist_ok=True)

        # Save and resize image
        try:
            from PIL import ImageOps
            img = PILImage.open(file.stream)
            img = ImageOps.exif_transpose(img)  # Apply EXIF rotation (phone cameras)
            img = img.convert("RGB")

            # Resize if too large (max 2048px on long edge)
            max_edge = 2048
            if max(img.size) > max_edge:
                ratio = max_edge / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, PILImage.LANCZOS)

            original_path = os.path.join(upload_dir, "original.png")
            img.save(original_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded image: {e}")
            continue

        # Create library item
        item_id = db.add_uploaded_item(library_id, desc, original_path)
        item_ids.append(item_id)

    return jsonify({
        "items_created": len(item_ids),
        "item_ids": item_ids,
        "success": True
    })


@app.route('/api/sam/predict', methods=['POST'])
def api_sam_predict():
    """
    Run SAM prediction from point prompts.

    Request body:
    {
        "item_id": "abc123",
        "points": [{"x": 100, "y": 200, "label": 1}, ...]
    }

    Returns mask as base64 PNG + confidence score.
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json()
    item_id = data.get('item_id')
    points = data.get('points', [])

    if not item_id or not points:
        return jsonify({"error": "item_id and points required", "success": False}), 400

    # Look up item
    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    image_path = item.get('original_image_path') or item.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Image not found", "success": False}), 404

    # Acquire GPU lock briefly
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gpu_lock import GPULock

    gpu_lock = GPULock("flask")
    if not gpu_lock.acquire(timeout=30):
        return jsonify({"error": "GPU is busy (another worker is running). Please retry in a moment.", "success": False}), 503

    try:
        sam = get_sam()
        sam.check_auto_unload()

        # Use object_name from item as text prompt if available
        text_prompt = item.get('object_name', '') if item else ''
        mask, score = sam.predict_point(item_id, image_path, points, text_prompt=text_prompt or None)

        # Encode mask as base64 PNG
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        mask_uint8 = (mask * 255).astype('uint8')
        mask_img = PILImage.fromarray(mask_uint8, mode='L')
        buffer = BytesIO()
        mask_img.save(buffer, format='PNG')
        mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            "mask_base64": mask_b64,
            "score": score,
            "width": mask.shape[1],
            "height": mask.shape[0],
            "success": True
        })

    except Exception as e:
        logger.error(f"SAM predict error: {e}")
        return jsonify({"error": str(e), "success": False}), 500
    finally:
        gpu_lock.release()


@app.route('/api/sam/accept', methods=['POST'])
def api_sam_accept():
    """
    Accept a SAM mask: composite object on white background and advance to needs_3d.

    Request body:
    {
        "item_id": "abc123",
        "points": [{"x": 100, "y": 200, "label": 1}, ...]
    }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json()
    item_id = data.get('item_id')
    points = data.get('points', [])

    if not item_id or not points:
        return jsonify({"error": "item_id and points required", "success": False}), 400

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    image_path = item.get('original_image_path') or item.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Image not found", "success": False}), 404

    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gpu_lock import GPULock

    gpu_lock = GPULock("flask")
    if not gpu_lock.acquire(timeout=30):
        return jsonify({"error": "GPU is busy. Please retry.", "success": False}), 503

    try:
        sam = get_sam()
        sam.check_auto_unload()

        # Use object_name from item as text prompt if available
        text_prompt = item.get('object_name', '') if item else ''
        mask, score = sam.predict_point(item_id, image_path, points, text_prompt=text_prompt or None)

        # Composite on white background
        result_image = sam.composite_on_white(image_path, mask)

        # Save masked image
        upload_dir = os.path.dirname(image_path)
        masked_path = os.path.join(upload_dir, "masked.png")
        result_image.save(masked_path)

        # Update item: set image_path to masked version, advance to needs_3d
        db.library_items.update_one(
            {"_id": item_id},
            {"$set": {
                "image_path": masked_path,
                "status": "needs_3d",
                "sam_points": points,
                "sam_score": score,
            }}
        )

        masked_url = _path_to_url(masked_path)
        return jsonify({
            "image_url": masked_url,
            "score": score,
            "status": "needs_3d",
            "success": True
        })

    except Exception as e:
        logger.error(f"SAM accept error: {e}")
        return jsonify({"error": str(e), "success": False}), 500
    finally:
        gpu_lock.release()


# ============== Page Routes ==============

@app.route('/library-review/<library_id>')
def library_review_page(library_id):
    """Library review UI page."""
    db = get_db()
    if not db:
        return "Database not available", 503

    library = db.get_library(library_id)
    if not library:
        return "Library not found", 404

    return render_template('library_review.html', library=library)


@app.route('/library-mask/<item_id>')
def library_mask_page(item_id):
    """Interactive SAM masking page for an uploaded library item."""
    db = get_db()
    if not db:
        return "Database not available", 503

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return "Item not found", 404

    library = db.get_library(item["library_id"])
    image_url = _path_to_url(item.get("original_image_path") or item.get("image_path"))

    return render_template('library_mask.html', item=item, library=library, image_url=image_url)


@app.route('/library-search')
def library_search_page():
    """Library search test page - text and image search with gallery view."""
    return render_template('library_search.html')


# ============== FAISS Index Management ==============

@app.route('/api/faiss/rebuild', methods=['POST'])
def api_faiss_rebuild():
    """
    Rebuild FAISS indices from database.

    Request body (optional):
    {
        "library_id": "optional - rebuild for specific library",
        "embedding_field": "text_embedding"  // or "image_embedding" or "all"
    }
    """
    data = request.get_json() or {}
    library_id = data.get('library_id')
    embedding_field = data.get('embedding_field', 'all')

    try:
        from services.faiss_service import get_faiss_service, reset_faiss_service

        # Reset to get fresh service
        reset_faiss_service()
        faiss_svc = get_faiss_service()

        if embedding_field == 'all':
            faiss_svc.build_index(library_id, "text_embedding")
            faiss_svc.build_index(library_id, "image_embedding")
        else:
            faiss_svc.build_index(library_id, embedding_field)

        stats = faiss_svc.get_stats()

        return jsonify({
            "rebuilt": True,
            "stats": stats,
            "success": True
        })

    except Exception as e:
        logger.error(f"FAISS rebuild failed: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/faiss/stats', methods=['GET'])
def api_faiss_stats():
    """Get FAISS index statistics."""
    try:
        from services.faiss_service import get_faiss_service
        faiss_svc = get_faiss_service()
        stats = faiss_svc.get_stats()

        return jsonify({
            "stats": stats,
            "success": True
        })

    except Exception as e:
        logger.error(f"Failed to get FAISS stats: {e}")
        return jsonify({"error": str(e), "success": False}), 500


# ============== Environment Panoramas ==============

@app.route('/api/environments')
def list_environments():
    """List available equirectangular environment panoramas."""
    env_dir = os.path.join(MEDIA_FOLDER, 'environments')
    if not os.path.isdir(env_dir):
        return jsonify([])
    exts = {'.jpg', '.jpeg', '.png', '.hdr', '.exr'}
    envs = []
    for f in sorted(os.listdir(env_dir)):
        if os.path.splitext(f)[1].lower() in exts:
            envs.append({
                'name': os.path.splitext(f)[0],
                'url': f'/media/environments/{f}',
            })
    return jsonify(envs)


# ============== Catalog ==============

@app.route('/catalog')
def catalog_page():
    """Flat catalog of all items across all libraries."""
    return render_template('catalog.html')


@app.route('/api/catalog/items', methods=['GET'])
def api_catalog_items():
    """
    Get all items as a flat list, newest first.

    Query params:
        status: Filter by status (default: all)
        limit: Max items (default: 100)
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    status = request.args.get('status', '')
    limit = int(request.args.get('limit', 100))

    query = {}
    if status:
        if status == 'processing':
            query["status"] = {"$in": ["approved", "needs_3d", "needs_embedding"]}
        else:
            query["status"] = status

    try:
        items = list(
            db.library_items.find(query)
            .sort([("created_at", -1)])
            .limit(limit)
        )

        for item in items:
            item['_id'] = str(item['_id'])
            if 'created_at' in item and item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
            if item.get('image_path'):
                item['image_path'] = _path_to_url(item['image_path'])
            if item.get('asset_path'):
                item['asset_path'] = _path_to_url(item['asset_path'])
            if item.get('original_image_path'):
                item['original_image_path'] = _path_to_url(item['original_image_path'])
            item.pop('text_embedding', None)
            item.pop('image_embedding', None)

        return jsonify({"items": items, "count": len(items), "success": True})

    except Exception as e:
        logger.error(f"Failed to get catalog items: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/catalog/stats', methods=['GET'])
def api_catalog_stats():
    """Get status counts across all items."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    try:
        stats = db.get_library_stats()
        return jsonify({**stats, "success": True})
    except Exception as e:
        logger.error(f"Failed to get catalog stats: {e}")
        return jsonify({"error": str(e), "success": False}), 500


# ============== Capture Flow ==============

@app.route('/capture')
def capture_page():
    """Phone-first capture UI: photograph → describe → mask → reconstruct → library."""
    return render_template('capture.html')


@app.route('/api/capture/upload', methods=['POST'])
def api_capture_upload():
    """
    Upload a photo, auto-describe with VLM, auto-mask with SAM center-point.

    Accepts multipart/form-data with 'image' file.
    Returns: { item_id, description, mask_base64, score, image_url }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    file = request.files.get('image')
    if not file or not file.filename:
        return jsonify({"error": "No image uploaded", "success": False}), 400

    import uuid
    from PIL import Image as PILImage

    # Save uploaded image
    output_dir = os.path.join(MEDIA_FOLDER, 'generated')
    capture_id = uuid.uuid4().hex[:8]
    capture_dir = os.path.join(output_dir, f"capture_{capture_id}")
    os.makedirs(capture_dir, exist_ok=True)

    try:
        from PIL import ImageOps
        img = PILImage.open(file.stream)
        img = ImageOps.exif_transpose(img)  # Apply EXIF rotation (phone cameras)
        img = img.convert("RGB")

        # Resize if too large (max 2048px on long edge)
        max_edge = 2048
        if max(img.size) > max_edge:
            ratio = max_edge / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, PILImage.LANCZOS)

        original_path = os.path.join(capture_dir, "original.png")
        img.save(original_path)
        img_width, img_height = img.size
    except Exception as e:
        logger.error(f"Failed to save captured image: {e}")
        return jsonify({"error": f"Failed to process image: {e}", "success": False}), 500

    # Acquire GPU lock for VLM + SAM3 (both use GPU)
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gpu_lock import GPULock

    description = ""
    object_name = ""
    mask_b64 = None
    score = 0.0

    gpu_lock = GPULock("flask")
    if not gpu_lock.acquire(timeout=60):
        # GPU busy — create item without VLM/SAM, user can still edit manually
        logger.warning("GPU busy, skipping VLM + SAM3 auto-processing")
        item_id = db.add_capture_item("", original_path)
    else:
        try:
            # Step 1: VLM — describe object and identify main object name
            try:
                from services.vlm_service import describe_and_identify
                ollama = get_ollama()
                vlm_result = describe_and_identify(ollama, original_path)
                description = vlm_result.get("description", "")
                object_name = vlm_result.get("object_name", "")
                # Unload VLM from VRAM before loading SAM3
                ollama.unload_model()
            except Exception as e:
                logger.warning(f"VLM failed (continuing without description): {e}")

            # Create database item
            item_id = db.add_capture_item(description, original_path)

            # Store object_name on item for later refinement
            if object_name:
                db.library_items.update_one(
                    {"_id": item_id},
                    {"$set": {"object_name": object_name}}
                )

            # Step 2: SAM3 — auto-mask using VLM-identified object name as text prompt
            try:
                sam = get_sam()
                sam.check_auto_unload()

                text_prompt = object_name or "object"
                mask, score = sam.predict_text(item_id, original_path, text_prompt)

                # Encode mask as base64 PNG
                import base64
                from io import BytesIO
                mask_uint8 = (mask * 255).astype('uint8')
                mask_img = PILImage.fromarray(mask_uint8, mode='L')
                buffer = BytesIO()
                mask_img.save(buffer, format='PNG')
                mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                logger.warning(f"SAM3 auto-mask failed (continuing without): {e}")
        finally:
            gpu_lock.release()

    image_url = _path_to_url(original_path)

    return jsonify({
        "item_id": item_id,
        "description": description,
        "object_name": object_name,
        "mask_base64": mask_b64,
        "score": float(score),
        "image_url": image_url,
        "width": img_width,
        "height": img_height,
        "success": True
    })


@app.route('/api/capture/refine-mask', methods=['POST'])
def api_capture_refine_mask():
    """
    Refine SAM mask with user-provided points.

    Request body: { "item_id": "abc", "points": [{"x":100,"y":200,"label":1}, ...] }
    Returns: { mask_base64, score }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json()
    item_id = data.get('item_id')
    points = data.get('points', [])

    if not item_id or not points:
        return jsonify({"error": "item_id and points required", "success": False}), 400

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    image_path = item.get('original_image_path') or item.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Image not found", "success": False}), 404

    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gpu_lock import GPULock

    gpu_lock = GPULock("flask")
    if not gpu_lock.acquire(timeout=30):
        return jsonify({"error": "GPU is busy. Please retry.", "success": False}), 503

    try:
        sam = get_sam()
        sam.check_auto_unload()

        # Use stored object_name as text prompt for SAM3 refinement
        text_prompt = item.get('object_name', '')
        mask, score = sam.predict_point(item_id, image_path, points, text_prompt=text_prompt or None)

        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        mask_uint8 = (mask * 255).astype('uint8')
        mask_img = PILImage.fromarray(mask_uint8, mode='L')
        buffer = BytesIO()
        mask_img.save(buffer, format='PNG')
        mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            "mask_base64": mask_b64,
            "score": float(score),
            "width": mask.shape[1],
            "height": mask.shape[0],
            "success": True
        })
    except Exception as e:
        logger.error(f"Capture refine-mask error: {e}")
        return jsonify({"error": str(e), "success": False}), 500
    finally:
        gpu_lock.release()


@app.route('/api/capture/reconstruct', methods=['POST'])
def api_capture_reconstruct():
    """
    Accept mask, composite on white, set status to needs_3d for Trellis2.

    Request body: { "item_id": "abc", "description": "edited desc", "points": [...] }
    """
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    data = request.get_json()
    item_id = data.get('item_id')
    description = data.get('description', '').strip()
    points = data.get('points', [])

    if not item_id:
        return jsonify({"error": "item_id required", "success": False}), 400

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    image_path = item.get('original_image_path') or item.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Image not found", "success": False}), 404

    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gpu_lock import GPULock

    gpu_lock = GPULock("flask")
    if not gpu_lock.acquire(timeout=30):
        return jsonify({"error": "GPU is busy. Please retry.", "success": False}), 503

    try:
        sam = get_sam()
        sam.check_auto_unload()

        # Use stored object_name as text prompt for SAM3
        text_prompt = item.get('object_name', '')

        if points:
            mask, score = sam.predict_point(item_id, image_path, points, text_prompt=text_prompt or None)
        else:
            # No refinement points — re-run text-only prediction
            mask, score = sam.predict_text(item_id, image_path, text_prompt or "object")

        # Composite on white background
        result_image = sam.composite_on_white(image_path, mask)

        # Save masked image
        upload_dir = os.path.dirname(image_path)
        masked_path = os.path.join(upload_dir, "masked.png")
        result_image.save(masked_path)

        # Update item: set masked image, description, advance to needs_3d
        update = {
            "image_path": masked_path,
            "status": "needs_3d",
            "sam_score": float(score),
        }
        if points:
            update["sam_points"] = points
        if description:
            update["description"] = description

        db.library_items.update_one({"_id": item_id}, {"$set": update})

        return jsonify({
            "item_id": item_id,
            "status": "needs_3d",
            "image_url": _path_to_url(masked_path),
            "score": float(score),
            "success": True
        })
    except Exception as e:
        logger.error(f"Capture reconstruct error: {e}")
        return jsonify({"error": str(e), "success": False}), 500
    finally:
        gpu_lock.release()


@app.route('/api/capture/<item_id>/status', methods=['GET'])
def api_capture_status(item_id):
    """Poll item status during Trellis2 processing."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    status = item.get('status', 'unknown')
    result = {
        "item_id": item_id,
        "status": status,
        "description": item.get('description', ''),
        "success": True
    }

    # Trellis2 done: capture items → 'reconstructed', other items → 'needs_embedding'
    if status in ('reconstructed', 'needs_embedding', 'ready'):
        asset_path = item.get('asset_path')
        if asset_path:
            result["asset_url"] = _path_to_url(asset_path)
            result["asset_type"] = item.get('asset_type', 'mesh')
        # Normalize to 'reconstructed' for the capture UI
        if status in ('needs_embedding', 'reconstructed'):
            result["status"] = "reconstructed"

    if item.get('error'):
        result["error"] = item['error']

    return jsonify(result)


@app.route('/api/capture/<item_id>/approve', methods=['POST'])
def api_capture_approve(item_id):
    """Approve 3D model: run inline CLIP encoding and set status to ready."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    status = item.get('status')
    if status == 'ready':
        # Already approved and encoded
        return jsonify({"item_id": item_id, "status": "ready", "success": True})
    if status not in ('reconstructed', 'needs_embedding'):
        return jsonify({"error": f"Item not ready for approval (status: {status})", "success": False}), 400

    # Inline CLIP/FLAIR encoding
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from gpu_lock import GPULock

    gpu_lock = GPULock("flask")
    if not gpu_lock.acquire(timeout=30):
        return jsonify({"error": "GPU is busy. Please retry.", "success": False}), 503

    try:
        from services.embedding_service import get_embedding_service
        embed_svc = get_embedding_service()

        description = item.get('description', '')
        image_path = item.get('image_path') or item.get('original_image_path')

        text_emb = None
        image_emb = None

        if description:
            text_emb = embed_svc.encode_text(description)
            if text_emb is not None and not isinstance(text_emb, list):
                text_emb = text_emb.tolist()

        if image_path and os.path.exists(image_path):
            image_emb = embed_svc.encode_image(image_path)
            if image_emb is not None and not isinstance(image_emb, list):
                image_emb = image_emb.tolist()

        db.library_items.update_one(
            {"_id": item_id},
            {"$set": {
                "status": "ready",
                "text_embedding": text_emb,
                "image_embedding": image_emb,
            }}
        )

        return jsonify({
            "item_id": item_id,
            "status": "ready",
            "has_text_embedding": text_emb is not None,
            "has_image_embedding": image_emb is not None,
            "success": True
        })
    except Exception as e:
        logger.error(f"Capture approve/embed error: {e}")
        return jsonify({"error": str(e), "success": False}), 500
    finally:
        gpu_lock.release()


@app.route('/api/capture/<item_id>/reject', methods=['POST'])
def api_capture_reject(item_id):
    """Reject and delete a capture item and its files."""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "success": False}), 503

    item = db.library_items.find_one({"_id": item_id})
    if not item:
        return jsonify({"error": "Item not found", "success": False}), 404

    # Delete files
    import shutil
    for path_key in ('original_image_path', 'image_path'):
        path = item.get(path_key)
        if path and os.path.exists(path):
            parent_dir = os.path.dirname(path)
            if os.path.basename(parent_dir).startswith('capture_'):
                try:
                    shutil.rmtree(parent_dir)
                    logger.info(f"Deleted capture directory: {parent_dir}")
                except Exception as e:
                    logger.warning(f"Failed to delete {parent_dir}: {e}")
                break

    # Delete asset file if exists
    asset_path = item.get('asset_path')
    if asset_path and os.path.exists(asset_path):
        try:
            os.remove(asset_path)
        except Exception as e:
            logger.warning(f"Failed to delete asset {asset_path}: {e}")

    # Delete from database
    db.library_items.delete_one({"_id": item_id})

    return jsonify({"deleted": True, "success": True})


# ============== Media Routes ==============

@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve files from the media folder"""
    response = send_from_directory(MEDIA_FOLDER, filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
    if filename.endswith('.ply'):
        response.headers['Content-Type'] = 'application/octet-stream'
    return response


@app.route('/download-cert')
def download_cert():
    """Download the self-signed certificate for mobile device installation."""
    cert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certs', 'cert.pem')
    if not os.path.exists(cert_path):
        return "No certificate found. Start server with FLASK_HTTPS=true first.", 404
    return send_file(cert_path, mimetype='application/x-pem-file',
                     as_attachment=True, download_name='digital-dojo.pem')


@app.route('/media/<path:filename>', methods=['OPTIONS'])
def serve_media_options(filename):
    """Handle CORS preflight for media files"""
    from flask import Response
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
    return response


# ============== Main ==============

if __name__ == '__main__':
    # Check if viewer files exist
    if not os.path.exists(os.path.join(VIEWER_FOLDER, 'index.html')):
        print("WARNING: SuperSplat viewer files not found!")
        print("Run 'npm install' to build the viewer files.")
        print("")

    print("=" * 60)
    print("  Digital Dojo - Prose-Driven 3D Scene Authoring")
    print("=" * 60)
    print(f"Media folder: {MEDIA_FOLDER}")
    print(f"Viewer folder: {VIEWER_FOLDER}")
    print("")
    print("Available routes:")
    print("  http://localhost:5000/           - Custom WebGL viewer")
    print("  http://localhost:5000/workspace  - Prose Editor + Scene Viewer")
    print("  http://localhost:5000/editor     - Prose Editor (standalone)")
    print("  http://localhost:5000/supersplat - SuperSplat viewer (no UI)")
    print("  http://localhost:5000/chat       - Ollama Chat (debug)")
    print("")
    print("API endpoints:")
    print("  GET  /api/status        - Check service status")
    print("  POST /api/parse         - Parse text to scene graph")
    print("  GET  /api/scenes        - List saved scenes")
    print("  POST /api/generate      - Queue generation job")
    print("  GET  /api/jobs          - List pending jobs")
    print("  POST /api/generate/sync - Run generation immediately (testing)")
    print("")
    print("To process queued jobs, run workers in other terminals:")
    print("  python worker.py              # SD3.5 image generation")
    print("  python library_worker.py      # Library pipeline (SD3.5 + embeddings)")
    print("  conda activate trellis2")
    print("  python trellis2_worker.py     # 3D mesh generation (Trellis2)")
    print("")

    # Get port from environment or default
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    use_https = os.getenv("FLASK_HTTPS", "false").lower() == "true"

    ssl_ctx = None
    if use_https:
        import socket
        cert_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certs')
        cert_file = os.path.join(cert_dir, 'cert.pem')
        key_file = os.path.join(cert_dir, 'key.pem')
        if os.path.exists(cert_file) and os.path.exists(key_file):
            ssl_ctx = (cert_file, key_file)
            print(f"  HTTPS enabled (cert: {cert_dir}/)")
        else:
            try:
                from OpenSSL import crypto
                os.makedirs(cert_dir, exist_ok=True)

                # Detect local IP addresses for SANs
                san_entries = ['DNS:localhost', 'DNS:digital-dojo',
                               'IP:127.0.0.1', 'IP:::1']
                # Add extra IPs from env var (comma-separated)
                extra_ips = os.getenv('FLASK_HTTPS_IPS', '')
                for ip in extra_ips.split(','):
                    ip = ip.strip()
                    if ip:
                        entry = f'IP:{ip}'
                        if entry not in san_entries:
                            san_entries.append(entry)
                try:
                    hostname = socket.gethostname()
                    for info in socket.getaddrinfo(hostname, None):
                        ip = info[4][0]
                        entry = f'IP:{ip}'
                        if entry not in san_entries:
                            san_entries.append(entry)
                except Exception:
                    pass

                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                cert = crypto.X509()
                cert.get_subject().CN = 'digital-dojo'
                cert.get_subject().O = 'Digital Dojo Dev'
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                cert.add_extensions([
                    crypto.X509Extension(b'basicConstraints', True, b'CA:TRUE'),
                    crypto.X509Extension(b'subjectAltName', False,
                                         ','.join(san_entries).encode()),
                ])
                cert.sign(k, 'sha256')
                with open(cert_file, 'wb') as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                with open(key_file, 'wb') as f:
                    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                ssl_ctx = (cert_file, key_file)
                san_str = ', '.join(san_entries)
                print(f"  HTTPS enabled (generated cert with SANs: {san_str})")
            except ImportError:
                print("  HTTPS requested but pyopenssl not installed.")
                print("  Install with: pip install pyopenssl")
        if ssl_ctx:
            print(f"  Access via: https://<your-wsl-ip>:{port}/")
            print(f"  Install cert on mobile: https://<ip>:{port}/download-cert")
            print("  (Accept the browser security warning on first visit)")
            print("")

    app.run(debug=debug, host='0.0.0.0', port=port, ssl_context=ssl_ctx)
