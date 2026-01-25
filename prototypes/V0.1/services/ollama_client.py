"""
Ollama Client

Handles communication with local Ollama server for LLM inference.
Used for parsing natural language into structured scene data.

Supports vision models (qwen2-vl, llava, etc.) for image analysis.
"""

import requests
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url
        self.model = model

    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2048,
                 strip_thinking: bool = True) -> str:
        """
        Generate text completion from Ollama.

        Args:
            strip_thinking: If True, removes <think>...</think> tags from response (default True)
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        try:
            logger.info(f"Sending generate request to {self.base_url} (model: {self.model})")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # 5 min - first request loads model into VRAM
            )
            response.raise_for_status()
            response_json = response.json()
            logger.info(f"Ollama API response keys: {list(response_json.keys())}")
            if 'error' in response_json:
                logger.error(f"Ollama error: {response_json['error']}")

            # Get response - qwen3 models may put content in 'thinking' field instead of 'response'
            result = response_json.get("response", "")

            # If response is empty but thinking exists, use thinking content
            if not result and "thinking" in response_json:
                thinking = response_json.get("thinking", "")
                logger.info(f"Response empty, checking thinking field ({len(thinking)} chars)")
                # The thinking field may contain the actual JSON we need
                result = thinking

            # Debug: log raw response before any processing
            logger.info(f"Raw response ({len(result)} chars): {result[:500] if result else '(empty)'}...")

            # Strip thinking tags if requested (qwen3 models output these)
            if strip_thinking:
                result = self._strip_thinking_tags(result)
                logger.info(f"After stripping thinking tags ({len(result)} chars)")

            return result

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            raise TimeoutError("Ollama request timed out")

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama server")
            raise ConnectionError("Ollama server not running. Start with: ollama serve")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7,
             strip_thinking: bool = True, format_schema: Optional[Dict] = None) -> str:
        """
        Chat completion with message history.
        Messages format: [{"role": "user", "content": "..."}, ...]

        Args:
            strip_thinking: If True, removes <think>...</think> tags from response (default True)
            format_schema: Optional JSON schema to constrain output format (structured output)
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        # Add format constraint if schema provided
        if format_schema:
            payload["format"] = format_schema

        try:
            logger.info(f"Sending chat request to {self.base_url} (model: {self.model})")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300  # 5 min - first request loads model into VRAM
            )
            response.raise_for_status()
            result = response.json().get("message", {}).get("content", "")

            if strip_thinking:
                result = self._strip_thinking_tags(result)

            return result

        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise

    def extract_json(self, text: str, schema_hint: str = "") -> Optional[Dict[str, Any]]:
        """
        Use LLM to extract structured JSON from natural language text.
        Uses the chat endpoint which works better with qwen3 models.
        """
        # Use chat endpoint - it works for the chat page
        messages = [
            {
                "role": "system",
                "content": "You are a JSON extraction assistant. Return ONLY valid JSON, no explanation."
            },
            {
                "role": "user",
                "content": f"""Extract scene data from this description. Return ONLY a JSON object.

Description: {text}

Return this exact format:
{{"name": "scene name", "nodes": [{{"name": "object name", "transform": {{"position": [x,y,z]}}, "generator": {{"type": "splat", "prompt": "description for AI generation"}}}}]}}

Position rules: north=+z, east=+x, up=+y, center=[0,0,0]

Respond with ONLY the JSON object, nothing else:"""
            }
        ]

        try:
            logger.info(f"Using chat endpoint for JSON extraction...")

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                }
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            response_json = response.json()

            result = response_json.get("message", {}).get("content", "")
            logger.info(f"Chat response ({len(result)} chars): {result[:500] if result else '(empty)'}...")

            # Strip thinking tags
            result = self._strip_thinking_tags(result)

            if not result:
                logger.error("Empty response from chat endpoint")
                return None

            # Try direct parse first
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                # Fallback: extract JSON from response
                logger.info(f"Direct parse failed, trying to extract JSON...")
                json_str = self._extract_json_from_response(result)
                if json_str:
                    logger.info(f"Found JSON ({len(json_str)} chars)")
                    return json.loads(json_str)
                logger.error(f"Could not find JSON. Response: {result[:300]}...")
                return None

        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return None

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON from LLM response, handling various formats:
        - Raw JSON
        - Markdown code blocks
        - Qwen3 thinking tags (<think>...</think>)
        - Mixed content
        """
        import re

        text = response.strip()

        # Handle Qwen3 thinking tags - remove everything in <think>...</think>
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()

        # Try to find JSON in markdown code blocks first
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate

        # Try the whole text as JSON
        if self._is_valid_json(text):
            return text

        # Try to find JSON object anywhere in text (starts with { ends with })
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            candidate = json_match.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate

        # Try to find JSON array
        json_array_match = re.search(r'(\[[\s\S]*\])', text)
        if json_array_match:
            candidate = json_array_match.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate

        return None

    def _is_valid_json(self, text: str) -> bool:
        """Check if text is valid JSON"""
        try:
            json.loads(text)
            return True
        except:
            return False

    def _strip_thinking_tags(self, text: str) -> str:
        """
        Remove <think>...</think> tags from qwen3 model responses.
        Returns the text without thinking content.
        """
        import re
        # Remove thinking tags and their content
        result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return result.strip()

    def unload_model(self):
        """
        Request Ollama to unload the current model from memory.
        This frees VRAM for other models.
        """
        try:
            # Ollama doesn't have explicit unload, but we can set keep_alive to 0
            payload = {
                "model": self.model,
                "prompt": "",
                "keep_alive": 0
            }
            requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            logger.info(f"Requested Ollama to unload {self.model}")
        except Exception as e:
            logger.warning(f"Failed to unload Ollama model: {e}")

    # ==================== Vision Model Support ====================

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for Ollama API"""
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_with_image(self, prompt: str, image_path: Union[str, Path],
                            system: Optional[str] = None,
                            temperature: float = 0.7) -> str:
        """
        Generate text from prompt + image using a vision model.
        Requires a vision-capable model like qwen2-vl, llava, etc.
        """
        image_b64 = self._encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180  # Vision models can be slower
            )
            response.raise_for_status()
            return response.json().get("response", "")

        except requests.exceptions.Timeout:
            logger.error("Ollama vision request timed out")
            raise TimeoutError("Ollama vision request timed out")

        except Exception as e:
            logger.error(f"Vision request failed: {e}")
            raise

    def describe_image(self, image_path: Union[str, Path],
                       detail_level: str = "detailed") -> str:
        """
        Describe an image for scene understanding.

        Args:
            image_path: Path to the image
            detail_level: "brief", "detailed", or "technical"
        """
        prompts = {
            "brief": "Describe this image in one sentence.",
            "detailed": "Describe this image in detail. Include objects, their positions, colors, lighting, and overall composition.",
            "technical": "Provide a technical description of this image suitable for 3D scene reconstruction. List all objects, their approximate positions (use directions like north, south, east, west, center), sizes, materials, and spatial relationships."
        }

        prompt = prompts.get(detail_level, prompts["detailed"])
        return self.generate_with_image(prompt, image_path)

    def analyze_scene_from_image(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Analyze an image and extract scene graph data.
        Returns structured JSON similar to extract_json but from visual input.
        """
        system_prompt = """You are a 3D scene analysis assistant. Analyze the image and return ONLY valid JSON describing the scene structure.

Return a scene graph with:
- name: scene identifier
- description: brief scene description
- nodes: array of objects, each with:
  - name: object identifier
  - type: object category (tree, rock, building, etc.)
  - position: approximate [x, y, z] where center=[0,0,0], north=+z, east=+x, up=+y
  - properties: {color, material, size, etc.}
  - generator: {type: "splat", prompt: "detailed description for regenerating this object"}

Be specific about spatial relationships and object properties."""

        prompt = "Analyze this image and extract the scene structure as JSON. Respond with ONLY the JSON:"

        try:
            response = self.generate_with_image(
                prompt, image_path,
                system=system_prompt,
                temperature=0.3
            )

            # Use robust JSON extraction
            json_str = self._extract_json_from_response(response)
            if json_str:
                return json.loads(json_str)
            else:
                logger.error("Could not find valid JSON in vision response")
                logger.debug(f"Raw response: {response}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from vision response: {e}")
            logger.debug(f"Raw response: {response}")
            return None

        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return None

    def compare_images(self, image_path1: Union[str, Path],
                       image_path2: Union[str, Path],
                       aspect: str = "similarity") -> str:
        """
        Compare two images. Useful for checking generation quality.

        Note: Ollama API sends images in order, prompt should reference "first" and "second".
        """
        # Encode both images
        img1_b64 = self._encode_image(image_path1)
        img2_b64 = self._encode_image(image_path2)

        prompts = {
            "similarity": "Compare these two images. How similar are they? What are the key differences?",
            "quality": "Compare the quality of these two images. Which has better detail, lighting, and composition?",
            "consistency": "Are these two images consistent in style, lighting, and content? Describe any inconsistencies."
        }

        prompt = prompts.get(aspect, prompts["similarity"])

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img1_b64, img2_b64],
            "stream": False,
            "options": {"temperature": 0.5}
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            return response.json().get("response", "")

        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            raise
