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
    """List generated assets (splats, images)"""
    db = get_db()
    if not db:
        return jsonify({"error": "Database not available", "assets": []})

    scene_id = request.args.get('scene_id')
    asset_type = request.args.get('type', 'splat')

    if scene_id:
        assets = db.get_assets_for_scene(scene_id)
    else:
        # Get all assets of the specified type
        assets = list(db.assets.find({"type": asset_type}).sort("created_at", -1).limit(50))

    for asset in assets:
        asset['_id'] = str(asset['_id'])
        if 'created_at' in asset:
            asset['created_at'] = asset['created_at'].isoformat()

    return jsonify({"assets": assets, "count": len(assets), "success": True})


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
    Run generation synchronously (for testing).
    This bypasses the job queue and runs immediately.

    WARNING: This blocks the request until generation completes.
    Only use for testing with small prompts.

    Request body: { "prompt": "...", "stage": "image|splat|full" }
    """
    data = request.get_json()
    prompt = data.get('prompt', '')
    stage = data.get('stage', 'image')  # image, splat, or full

    if not prompt:
        return jsonify({"error": "No prompt provided", "success": False}), 400

    orchestrator = get_orchestrator()
    if not orchestrator:
        return jsonify({"error": "Orchestrator not available", "success": False}), 503

    try:
        if stage == 'image':
            # Just generate image
            output_path = orchestrator.run_image_only(prompt)
            return jsonify({
                "image_path": output_path,
                "success": True
            })
        elif stage == 'splat':
            # Need an image path
            image_path = data.get('image_path')
            if not image_path:
                return jsonify({"error": "image_path required for splat stage", "success": False}), 400
            output_path = orchestrator.run_splat_only(image_path)
            return jsonify({
                "splat_path": output_path,
                "success": True
            })
        else:
            # Full pipeline
            from services.orchestrator import GenerationJob
            job = GenerationJob(id="sync", prompt=prompt)
            result = orchestrator.run_pipeline(job)
            return jsonify({
                "image_path": result.image_path,
                "splat_path": result.splat_path,
                "status": result.status,
                "error": result.error,
                "success": result.status == "completed"
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
    print("To process queued jobs, run the worker in another terminal:")
    print("  python worker.py")
    print("")

    # Get port from environment or default
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    app.run(debug=debug, host='0.0.0.0', port=port)
