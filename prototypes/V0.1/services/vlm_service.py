"""
VLM Service - Uses Ollama vision models to describe and identify objects in photos.
"""

import json
import logging

logger = logging.getLogger(__name__)


def describe_and_identify(ollama_client, image_path: str) -> dict:
    """
    Ask the VLM to describe the scene and identify the main object.

    Returns:
        {
            "description": "A red ceramic coffee mug with a white handle",
            "object_name": "coffee mug",
        }
    """
    prompt = (
        "Look at this photo and identify the most prominent object.\n"
        "Respond with ONLY valid JSON, no other text:\n"
        "{\n"
        '  "description": "one sentence describing the object - what it is, color, material, style",\n'
        '  "object_name": "short 1-3 word name for the object"\n'
        "}\n"
    )

    result = {
        "description": "",
        "object_name": "",
    }

    try:
        response = ollama_client.generate_with_image(
            prompt=prompt,
            image_path=image_path,
            temperature=0.2
        )

        # Parse JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(text)

        description = data.get("description", "").strip().strip('"').strip("'")
        object_name = data.get("object_name", "").strip().strip('"').strip("'")

        result["description"] = description
        result["object_name"] = object_name

        logger.info(f"VLM: '{object_name}' - {description[:60]}")

    except json.JSONDecodeError as e:
        logger.warning(f"VLM returned non-JSON, falling back to description-only: {e}")
        result["description"] = _describe_fallback(ollama_client, image_path)
    except Exception as e:
        logger.error(f"VLM describe_and_identify failed: {e}")

    return result


def _describe_fallback(ollama_client, image_path: str) -> str:
    """Simple description fallback if structured output fails."""
    prompt = (
        "Describe the main object in this photo in one sentence. "
        "Focus on what the object IS, its color, material, and style. "
        "Be concise and specific. Do not describe the background or setting."
    )
    try:
        response = ollama_client.generate_with_image(
            prompt=prompt,
            image_path=image_path,
            temperature=0.3
        )
        return response.strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"VLM fallback failed: {e}")
        return ""
