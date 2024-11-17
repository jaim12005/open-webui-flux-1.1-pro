"""
title: FLUX.1.1 Pro Ultra Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 1.4
license: MIT
requirements: pydantic, requests
environment_variables: REPLICATE_API_TOKEN
supported providers: replicate.com
"""

import base64
import os
from typing import Any, Dict, Generator, Iterator, List, Union, Optional, Literal
import requests
from open_webui.utils.misc import get_last_user_message
from pydantic import BaseModel, Field, validator

AspectRatioType = Literal[
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"
]
OutputFormatType = Literal["jpg", "png"]
SafetyToleranceType = Literal[1, 2, 3, 4, 5, 6]


class Pipe:
    """
    Class representing the FLUX.1.1-pro-ultra Manifold Function.

    Available Aspect Ratios:
        - Wide formats: "21:9", "16:9"
        - Standard formats: "3:2", "4:3", "5:4", "1:1"
        - Portrait formats: "4:5", "3:4", "2:3", "9:16", "9:21"

    Available Output Formats:
        - "jpg": Standard JPEG format
        - "png": Lossless PNG format

    Safety Tolerance Levels:
        1: Maximum safety (strictest content filtering)
        2: Very high safety (default)
        3: High safety
        4: Medium safety
        5: Low safety
        6: Minimum safety (most permissive)
    """

    # Available options for configuration
    AVAILABLE_ASPECT_RATIOS = [
        "21:9",  # Ultra-wide
        "16:9",  # Wide screen
        "3:2",  # Standard photo
        "4:3",  # Standard monitor
        "5:4",  # Square-ish
        "1:1",  # Perfect square
        "4:5",  # Portrait
        "3:4",  # Portrait
        "2:3",  # Portrait
        "9:16",  # Mobile portrait
        "9:21",  # Ultra portrait
    ]

    AVAILABLE_OUTPUT_FORMATS = [
        "jpg",  # Standard format, smaller file size
        "png",  # Lossless format, larger file size
    ]

    AVAILABLE_SAFETY_LEVELS = [
        1,  # Maximum safety (strictest filtering)
        2,  # Very high safety (default)
        3,  # High safety
        4,  # Medium safety
        5,  # Low safety
        6,  # Minimum safety (most permissive)
    ]

    SAFETY_TOLERANCE_LEVELS = {
        1: "Maximum safety (strictest filtering)",
        2: "Very high safety (default)",
        3: "High safety",
        4: "Medium safety",
        5: "Low safety",
        6: "Minimum safety (most permissive)",
    }

    class Valves(BaseModel):
        """
        Pydantic model for storing API keys and base URLs.

        Environment Variables:
            REPLICATE_API_TOKEN: Your API Token for Replicate
            FLUX_RAW_MODE: Set to "true" for less processed images
            FLUX_SAFETY_TOLERANCE: Integer 1-6 where:
                1: Maximum safety (strictest filtering)
                2: Very high safety (default)
                3: High safety
                4: Medium safety
                5: Low safety
                6: Minimum safety (most permissive)
            FLUX_SEED: Integer for reproducible generation (optional)
            FLUX_ASPECT_RATIO: One of the available aspect ratios
            FLUX_OUTPUT_FORMAT: Either "jpg" or "png"
        """

        REPLICATE_API_TOKEN: str = Field(
            default="", description="Your API Token for Replicate"
        )
        REPLICATE_API_BASE_URL: str = Field(
            default="https://api.replicate.com/v1",
            description="Base URL for the Replicate API",
        )
        FLUX_RAW_MODE: bool = Field(
            default=False,
            description="Generate less processed, more natural-looking images",
        )
        FLUX_SAFETY_TOLERANCE: SafetyToleranceType = Field(
            default=1, description="Safety tolerance levels for content filtering"
        )
        FLUX_SEED: Optional[int] = Field(
            default=None,
            description="Random seed for image generation. None/blank for random",
        )
        FLUX_ASPECT_RATIO: AspectRatioType = Field(
            default="1:1", description="Aspect ratio for the generated image"
        )
        FLUX_OUTPUT_FORMAT: OutputFormatType = Field(
            default="jpg", description="Output format for the generated image"
        )

        @validator("FLUX_SAFETY_TOLERANCE")
        def validate_safety_tolerance(cls, v):
            if not 1 <= v <= 6:
                raise ValueError("Safety tolerance must be between 1 and 6")
            return v

    def __init__(self):
        """
        Initialize the Pipe class with default values and environment variables.
        """
        self.type = "manifold"
        self.id = "FLUX_1_1__PRO_ULTRA"
        self.name = "FLUX.1.1-pro-ultra: "
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN", ""),
            REPLICATE_API_BASE_URL=os.getenv(
                "REPLICATE_API_BASE_URL",
                "https://api.replicate.com/v1",
            ),
            FLUX_RAW_MODE=os.getenv("FLUX_RAW_MODE", "false").lower() == "true",
            FLUX_SAFETY_TOLERANCE=int(os.getenv("FLUX_SAFETY_TOLERANCE", "1")),
            FLUX_SEED=int(os.getenv("FLUX_SEED")) if os.getenv("FLUX_SEED") else None,
            FLUX_ASPECT_RATIO=os.getenv("FLUX_ASPECT_RATIO", "1:1"),
            FLUX_OUTPUT_FORMAT=os.getenv("FLUX_OUTPUT_FORMAT", "jpg"),
        )

    def get_img_extension(self, img_data: str) -> Union[str, None]:
        """
        Get the image extension based on the base64-encoded data.
        """
        if img_data.startswith("/9j/"):
            return "jpeg"
        elif img_data.startswith("iVBOR"):
            return "png"
        elif img_data.startswith("R0lG"):
            return "gif"
        elif img_data.startswith("UklGR"):
            return "webp"
        return None

    def handle_image_response(self, response: requests.Response) -> str:
        """
        Handle image response from the API.
        """
        content_type = response.headers.get("Content-Type", "")
        img_ext = "png"
        if "image/" in content_type:
            img_ext = content_type.split("/")[-1]
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return f"![Image](data:{content_type};base64,{image_base64})\n`GeneratedImage.{img_ext}`"

    def stream_response(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Generator[str, None, None]:
        yield self.non_stream_response(headers, payload)

    def non_stream_response(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> str:
        """
        Get a non-streaming response from the API.
        """
        try:
            # Create prediction with correct URL
            create_url = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro-ultra/predictions"
            response = requests.post(
                url=create_url,
                headers=headers,
                json=payload,
                timeout=(3.05, 60),
            )
            response.raise_for_status()
            prediction = response.json()

            # Poll for completion with correct URL
            while prediction["status"] not in ["succeeded", "failed", "canceled"]:
                poll_url = (
                    f"https://api.replicate.com/v1/predictions/{prediction['id']}"
                )
                response = requests.get(
                    poll_url,
                    headers={
                        "Authorization": f"Bearer {self.valves.REPLICATE_API_TOKEN}"
                    },
                )
                prediction = response.json()
                if prediction["status"] == "failed":
                    return f"Error: Generation failed: {prediction.get('error', 'Unknown error')}"

            # Handle the completed prediction
            if prediction["status"] == "succeeded":
                image_url = prediction["output"]
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                return self.handle_image_response(img_response)

            return "Error: Image generation failed"

        except requests.exceptions.RequestException as e:
            return f"Error: Request failed: {e}"
        except Exception as e:
            return f"Error: {e}"

    def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        """
        Process the pipe request.
        """
        headers = {
            "Authorization": f"Bearer {self.valves.REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }

        prompt = get_last_user_message(body["messages"])

        # Build base payload
        payload = {
            "input": {
                "prompt": prompt,
                "aspect_ratio": self.valves.FLUX_ASPECT_RATIO,
                "output_format": self.valves.FLUX_OUTPUT_FORMAT,
                "safety_tolerance": self.valves.FLUX_SAFETY_TOLERANCE,
                "raw": self.valves.FLUX_RAW_MODE,
            }
        }

        # Only add seed if it's not None
        if self.valves.FLUX_SEED is not None:
            payload["input"]["seed"] = self.valves.FLUX_SEED

        try:
            if body.get("stream", False):
                return self.stream_response(headers, payload)
            else:
                return self.non_stream_response(headers, payload)
        except requests.exceptions.RequestException as e:
            return f"Error: Request failed: {e}"
        except Exception as e:
            return f"Error: {e}"

    def pipes(self) -> List[Dict[str, str]]:
        """
        Get the list of available pipes.
        """
        return [{"id": "flux_1_1_pro_ultra", "name": "Flux 1.1 Pro Ultra"}]


# Add environment variable validation helper
def validate_env_int(value: str, default: int, min_val: int, max_val: int) -> int:
    """Helper function to validate integer environment variables."""
    try:
        val = int(value)
        return max(min_val, min(val, max_val))
    except (TypeError, ValueError):
        return default