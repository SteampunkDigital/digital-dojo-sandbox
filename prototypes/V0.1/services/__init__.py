# Digital Dojo Services
# Lazy imports to handle missing dependencies gracefully

import logging
logger = logging.getLogger(__name__)

# Always available (only needs requests)
from .ollama_client import OllamaClient

# Try to import others, but don't fail if dependencies missing
try:
    from .orchestrator import GPUOrchestrator, GenerationJob
except ImportError as e:
    logger.warning(f"GPUOrchestrator not available: {e}")
    GPUOrchestrator = None
    GenerationJob = None

try:
    from .scene_parser import SceneParser, Scene, SceneNode
except ImportError as e:
    logger.warning(f"SceneParser not available: {e}")
    SceneParser = None
    Scene = None
    SceneNode = None

try:
    from .database import DatabaseService, db
except ImportError as e:
    logger.warning(f"DatabaseService not available: {e}")
    DatabaseService = None
    db = None

__all__ = [
    'GPUOrchestrator',
    'GenerationJob',
    'OllamaClient',
    'SceneParser',
    'Scene',
    'SceneNode',
    'DatabaseService',
    'db'
]
