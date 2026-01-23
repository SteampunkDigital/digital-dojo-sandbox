"""
Flask server for Gaussian Splat Viewer
Uses sam3d-objects conda environment

Setup:
    npm install
    python app.py
"""

from flask import Flask, render_template, send_from_directory, redirect
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_FOLDER = os.path.join(BASE_DIR, 'media')
VIEWER_FOLDER = os.path.join(BASE_DIR, 'static', 'viewer')

# Create Flask app with custom static folder
app = Flask(__name__, static_folder='static', static_url_path='/static')

@app.route('/')
def index():
    """Landing page with viewer options"""
    return render_template('index.html')

@app.route('/supersplat')
def supersplat_viewer():
    """Serve the SuperSplat viewer - redirects to static viewer with content param"""
    return redirect('/static/viewer/index.html?content=/media/splat.ply&noui')

@app.route('/supersplat/ui')
def supersplat_viewer_with_ui():
    """Serve the SuperSplat viewer with UI controls"""
    return redirect('/static/viewer/index.html?content=/media/splat.ply')

@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve files from the media folder"""
    response = send_from_directory(MEDIA_FOLDER, filename)
    # Enable CORS for media files (needed by SuperSplat viewer)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
    # Set correct content type for PLY files
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

if __name__ == '__main__':
    # Check if viewer files exist
    if not os.path.exists(os.path.join(VIEWER_FOLDER, 'index.html')):
        print("WARNING: SuperSplat viewer files not found!")
        print("Run 'npm install' to build the viewer files.")
        print("")

    print("Starting Gaussian Splat Viewer Server...")
    print(f"Media folder: {MEDIA_FOLDER}")
    print(f"Viewer folder: {VIEWER_FOLDER}")
    print("")
    print("Available routes:")
    print("  http://localhost:5000/           - Custom WebGL viewer")
    print("  http://localhost:5000/supersplat - SuperSplat viewer (no UI)")
    print("  http://localhost:5000/supersplat/ui - SuperSplat viewer (with UI)")
    print("")
    app.run(debug=True, host='0.0.0.0', port=5000)
