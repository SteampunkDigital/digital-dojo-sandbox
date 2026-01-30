# Library REST API Documentation

Access the 3D object library for searching and retrieving assets.

## Base URL

```
http://localhost:5000
```

## Endpoints

### Get Item by ID

Retrieve a specific library item by its unique identifier.

```
GET /api/library/items/{item_id}
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `item_id` | string | The unique identifier of the library item |

**Response:**
```json
{
  "item": {
    "id": "a1b2c3d4e5f6",
    "description": "A ceramic coffee mug with a thick, curved handle",
    "status": "ready",
    "seed": 1234567890,
    "library_id": "lib_abc123",
    "variant_group_id": "grp_xyz789",
    "image_url": "/media/generated/sd_abc123/00000.png",
    "asset_url": "/media/generated/mesh_def456.glb",
    "asset_type": "mesh"
  },
  "success": true
}
```

**Error Responses:**
- `404` - Item not found
- `503` - Database unavailable

---

### Search Library

Search for objects using text, images, or combined queries. Supports text matching, semantic vector search, and visual similarity search.

```
POST /api/libraries/search
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "comfortable seating furniture",
  "image_url": "/media/generated/sd_xxx/00000.png",
  "image_base64": "iVBORw0KGgo...",
  "mode": "vector",
  "embedding_field": "text_embedding",
  "use_faiss": true,
  "limit": 10,
  "library_id": "optional_library_id",
  "status": "ready"
}
```

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | No* | - | Search text (natural language for vector, substring for text) |
| `image_url` | string | No* | - | URL to image for visual similarity search |
| `image_base64` | string | No* | - | Base64-encoded image for visual search |
| `mode` | string | No | `"vector"` | Search mode: `"vector"`, `"text"`, or `"image"` |
| `embedding_field` | string | No | `"text_embedding"` | Which embeddings to search: `"text_embedding"` or `"image_embedding"` |
| `use_faiss` | boolean | No | `true` | Use FAISS index for faster search |
| `limit` | integer | No | `10` | Maximum number of results to return |
| `library_id` | string | No | - | Limit search to a specific library |
| `status` | string | No | `"ready"` | Filter by item status (use `null` for all statuses) |

*At least one of `query`, `image_url`, or `image_base64` is required.

**Response:**
```json
{
  "results": [
    {
      "id": "item_123",
      "description": "A fabric-upholstered sectional sofa with deep cushions",
      "status": "ready",
      "seed": 987654321,
      "library_id": "lib_abc123",
      "variant_group_id": "grp_sofa01",
      "image_url": "/media/generated/sd_xxx/00000.png",
      "asset_url": "/media/generated/mesh_yyy.glb",
      "asset_type": "mesh",
      "similarity": 0.847
    }
  ],
  "count": 1,
  "mode": "vector",
  "embedding_field": "text_embedding",
  "success": true
}
```

**Search Modes:**
| Mode | Description |
|------|-------------|
| `"text"` | Simple substring matching on description field |
| `"vector"` | Semantic similarity using text or image embeddings |
| `"image"` | Visual similarity (automatically uses image_embedding field) |

**Notes:**
- Vector search uses CLIP embeddings (768 dimensions for OpenCLIP, 512 for FLAIR)
- Similarity scores range from 0 to 1 (higher = more similar)
- Results are sorted by similarity (descending) for vector search
- FAISS provides faster search for large libraries; falls back to brute-force if unavailable
- Image search requires providing either `image_url` (local or remote) or `image_base64`

---

### List Libraries

Get all available libraries.

```
GET /api/libraries
```

**Response:**
```json
{
  "libraries": [
    {
      "_id": "lib_abc123",
      "name": "Household Objects",
      "item_count": 54,
      "total_variants": 162,
      "status": "ready",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "count": 1,
  "success": true
}
```

---

### Get Library Details

Get a specific library with all its items.

```
GET /api/libraries/{library_id}
```

**Response:**
```json
{
  "library": {
    "_id": "lib_abc123",
    "name": "Household Objects",
    "item_count": 54,
    "status": "ready"
  },
  "items": [...],
  "status_counts": {
    "ready": 150,
    "failed": 12,
    "pending": 0
  },
  "success": true
}
```

---

### Rebuild FAISS Index

Rebuild the FAISS search index from database. Use this after adding many items or changing embedding backends.

```
POST /api/faiss/rebuild
Content-Type: application/json
```

**Request Body (optional):**
```json
{
  "library_id": "optional_library_id",
  "embedding_field": "all"
}
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `library_id` | string | - | Rebuild for specific library only |
| `embedding_field` | string | `"all"` | `"text_embedding"`, `"image_embedding"`, or `"all"` |

**Response:**
```json
{
  "rebuilt": true,
  "stats": {
    "embedding_dim": 768,
    "use_gpu": true,
    "combined_text_count": 150,
    "combined_image_count": 145
  },
  "success": true
}
```

---

### Get FAISS Stats

Get information about the current FAISS indices.

```
GET /api/faiss/stats
```

**Response:**
```json
{
  "stats": {
    "embedding_dim": 768,
    "use_gpu": true,
    "text_indices": {},
    "image_indices": {},
    "combined_text_count": 150,
    "combined_image_count": 145
  },
  "success": true
}
```

---

## Response Fields

### Item Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `description` | string | Text description of the object |
| `status` | string | Processing status (see below) |
| `seed` | integer | Random seed used for generation |
| `library_id` | string | Parent library ID |
| `variant_group_id` | string | Groups variants of the same description |
| `image_url` | string | URL to the generated 2D image |
| `asset_url` | string | URL to the 3D model file |
| `asset_type` | string | `"mesh"` (.glb) or `"splat"` (.ply) |
| `similarity` | float | Similarity score (search results only) |

### Status Values

| Status | Description |
|--------|-------------|
| `ready` | Fully processed, available for use |
| `pending` | Awaiting image generation |
| `needs_review` | Image generated, awaiting human approval |
| `approved` | Approved, awaiting 3D processing |
| `needs_3d` | Mask generated, awaiting 3D generation |
| `needs_embedding` | 3D generated, awaiting embeddings |
| `failed` | Processing failed |
| `rejected` | Rejected during review |

---

## Usage Examples

### JavaScript (Fetch API)

**Get item by ID:**
```javascript
async function getLibraryItem(itemId) {
  const response = await fetch(`/api/library/items/${itemId}`);
  const data = await response.json();

  if (data.success) {
    console.log('Description:', data.item.description);
    console.log('3D Model:', data.item.asset_url);
    console.log('Image:', data.item.image_url);
  }
  return data.item;
}
```

**Search for objects by text:**
```javascript
async function searchLibrary(query, limit = 10) {
  const response = await fetch('/api/libraries/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query,
      mode: 'vector',
      limit: limit
    })
  });
  const data = await response.json();

  if (data.success) {
    data.results.forEach(item => {
      console.log(`${item.description} (${item.similarity.toFixed(2)})`);
    });
  }
  return data.results;
}

// Example usage
const chairs = await searchLibrary('wooden dining chair');
```

**Search by image (visual similarity):**
```javascript
async function searchByImage(imageUrl, limit = 10) {
  const response = await fetch('/api/libraries/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_url: imageUrl,
      mode: 'image',
      limit: limit
    })
  });
  const data = await response.json();
  return data.results;
}

// Find similar items to an existing library item
const similar = await searchByImage('/media/generated/sd_abc123/00000.png');
```

**Search with base64 image (e.g., from canvas):**
```javascript
async function searchByCanvasImage(canvas, limit = 10) {
  const base64 = canvas.toDataURL('image/png').split(',')[1];

  const response = await fetch('/api/libraries/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: base64,
      mode: 'image',
      limit: limit
    })
  });
  return (await response.json()).results;
}
```

**Load 3D model in Three.js:**
```javascript
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

async function loadLibraryModel(itemId) {
  // Get item details
  const response = await fetch(`/api/library/items/${itemId}`);
  const { item } = await response.json();

  if (item.asset_type !== 'mesh') {
    throw new Error('Item is not a mesh (GLB) format');
  }

  // Load GLB model
  const loader = new GLTFLoader();
  return new Promise((resolve, reject) => {
    loader.load(
      item.asset_url,
      (gltf) => resolve(gltf.scene),
      undefined,
      reject
    );
  });
}
```

### Python (Requests)

```python
import requests
import base64

BASE_URL = "http://localhost:5000"

def get_item(item_id):
    """Get a library item by ID"""
    response = requests.get(f"{BASE_URL}/api/library/items/{item_id}")
    data = response.json()
    if data["success"]:
        return data["item"]
    raise Exception(data.get("error", "Unknown error"))

def search_library(query, limit=10, mode="vector"):
    """Search the library using natural language"""
    response = requests.post(
        f"{BASE_URL}/api/libraries/search",
        json={
            "query": query,
            "mode": mode,
            "limit": limit
        }
    )
    data = response.json()
    if data["success"]:
        return data["results"]
    raise Exception(data.get("error", "Unknown error"))

def search_by_image(image_path, limit=10):
    """Search the library using visual similarity"""
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    response = requests.post(
        f"{BASE_URL}/api/libraries/search",
        json={
            "image_base64": image_base64,
            "mode": "image",
            "limit": limit
        }
    )
    data = response.json()
    if data["success"]:
        return data["results"]
    raise Exception(data.get("error", "Unknown error"))

def rebuild_faiss_index():
    """Rebuild FAISS index after adding items"""
    response = requests.post(f"{BASE_URL}/api/faiss/rebuild")
    return response.json()

# Example usage
results = search_library("kitchen appliances")
for item in results:
    print(f"{item['description']}: {item['asset_url']}")

# Find visually similar items
similar = search_by_image("my_reference_image.png")
```

### cURL

**Get item:**
```bash
curl http://localhost:5000/api/library/items/abc123
```

**Search:**
```bash
curl -X POST http://localhost:5000/api/libraries/search \
  -H "Content-Type: application/json" \
  -d '{"query": "red sofa", "mode": "vector", "limit": 5}'
```

---

## Asset URLs

All URLs returned by the API are relative paths that can be appended to the base URL:

- **Images**: `/media/generated/sd_xxxxx/00000.png` - PNG format, typically 1024x1024
- **Meshes**: `/media/generated/mesh_xxxxx.glb` - GLB format with vertex colors
- **Splats**: `/media/generated/splat_xxxxx.ply` - Gaussian splat PLY format

**Example full URLs:**
```
http://localhost:5000/media/generated/sd_abc123/00000.png
http://localhost:5000/media/generated/mesh_def456.glb
```

---

## Error Handling

All endpoints return errors in a consistent format:

```json
{
  "error": "Error message describing what went wrong",
  "success": false
}
```

**HTTP Status Codes:**
| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Bad request (missing/invalid parameters) |
| `404` | Resource not found |
| `500` | Server error |
| `503` | Service unavailable (database not connected) |

---

## CORS

The `/media/` endpoints include CORS headers allowing cross-origin requests:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, OPTIONS`

API endpoints do not include CORS headers by default. If accessing from a different origin, configure a proxy or add CORS middleware.
