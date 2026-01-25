"""
Scene Parser

Converts natural language descriptions into structured scene graphs.
Uses Ollama for NLP extraction, stores results in MongoDB.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Transform:
    """3D transform for a scene node"""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class Generator:
    """AI generation configuration for a node"""
    type: str = "splat"  # "splat" or "glb"
    prompt: str = ""
    status: str = "pending"  # pending, generating, completed, failed
    output_path: Optional[str] = None


@dataclass
class SceneNode:
    """A node in the scene graph hierarchy"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "unnamed"
    transform: Transform = field(default_factory=Transform)
    generator: Optional[Generator] = None
    children: List["SceneNode"] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        result = {
            "id": self.id,
            "name": self.name,
            "transform": asdict(self.transform),
            "properties": self.properties,
            "children": [child.to_dict() for child in self.children]
        }
        if self.generator:
            result["generator"] = asdict(self.generator)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneNode":
        """Create from dictionary"""
        node = cls(
            id=data.get("id", uuid.uuid4().hex[:8]),
            name=data.get("name", "unnamed"),
            properties=data.get("properties", {})
        )

        if "transform" in data:
            t = data["transform"]
            node.transform = Transform(
                position=t.get("position", [0, 0, 0]),
                rotation=t.get("rotation", [0, 0, 0]),
                scale=t.get("scale", [1, 1, 1])
            )

        if "generator" in data:
            g = data["generator"]
            node.generator = Generator(
                type=g.get("type", "splat"),
                prompt=g.get("prompt", ""),
                status=g.get("status", "pending"),
                output_path=g.get("output_path")
            )

        if "children" in data:
            node.children = [cls.from_dict(child) for child in data["children"]]

        return node


@dataclass
class Scene:
    """Complete scene with metadata"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Untitled Scene"
    description: str = ""
    root: SceneNode = field(default_factory=lambda: SceneNode(name="root"))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.id,
            "name": self.name,
            "description": self.description,
            "root": self.root.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        return cls(
            id=data.get("_id", uuid.uuid4().hex),
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            root=SceneNode.from_dict(data.get("root", {})),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow())
        )


class SceneParser:
    """
    Parses natural language into structured scene graphs.
    """

    def __init__(self, ollama_client):
        self.ollama = ollama_client

    def parse(self, text: str) -> Optional[Scene]:
        """
        Parse natural language text into a Scene object.
        """
        logger.info(f"Parsing scene from text ({len(text)} chars)")

        # Use Ollama to extract structured data
        scene_data = self.ollama.extract_json(text)

        if not scene_data:
            logger.error("Failed to extract scene data from text")
            return None

        # Build scene from extracted data
        scene = Scene(
            description=text,
            name=scene_data.get("name", "Parsed Scene")
        )

        # Parse the root node and its children
        if "scene" in scene_data:
            scene_nodes = scene_data["scene"]
            if isinstance(scene_nodes, list) and len(scene_nodes) > 0:
                scene.root = SceneNode.from_dict(scene_nodes[0])
            elif isinstance(scene_nodes, dict):
                scene.root = SceneNode.from_dict(scene_nodes)
        elif "root" in scene_data:
            scene.root = SceneNode.from_dict(scene_data["root"])
        elif "nodes" in scene_data:
            # Flat list of nodes - make them children of root
            for node_data in scene_data["nodes"]:
                scene.root.children.append(SceneNode.from_dict(node_data))
        else:
            # Try to parse the whole response as a single node
            scene.root = SceneNode.from_dict(scene_data)

        logger.info(f"Parsed scene with {self._count_nodes(scene.root)} nodes")
        return scene

    def _count_nodes(self, node: SceneNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def get_generation_queue(self, scene: Scene) -> List[SceneNode]:
        """
        Get list of nodes that need generation, in depth-first order.
        """
        queue = []
        self._collect_generators(scene.root, queue)
        return queue

    def _collect_generators(self, node: SceneNode, queue: List[SceneNode]):
        """Recursively collect nodes with generators"""
        if node.generator and node.generator.status == "pending":
            queue.append(node)
        for child in node.children:
            self._collect_generators(child, queue)

    def scene_to_narrative(self, scene: Scene) -> str:
        """
        Convert a scene graph back to natural language description.
        Used for bidirectional editing.
        """
        prompt = f"""Convert this scene graph JSON back into a natural, flowing English description.
Describe the spatial relationships, objects, and their properties in a way that reads like prose.

Scene JSON:
{scene.root.to_dict()}

Write the description:"""

        return self.ollama.generate(prompt, temperature=0.7)
