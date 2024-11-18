"""
title: FLUX.1.1 Pro Ultra Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 1.5
license: MIT
requirements: pydantic, requests
environment_variables: REPLICATE_API_TOKEN
supported providers: replicate.com
"""

import base64
import os
import time
from enum import Enum
from typing import Any, Dict, Generator, Iterator, List, Union, Optional, Literal
import requests
from open_webui.utils.misc import get_last_user_message
from pydantic import BaseModel, Field, validator, HttpUrl

# Type definitions
AspectRatioType = Literal[
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"
]
OutputFormatType = Literal["jpg", "png"]
SafetyToleranceType = Literal[1, 2, 3, 4, 5, 6]

class PredictionStatus(str, Enum):
    """Enum for prediction status values"""
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    PROCESSING = "processing"

class APIEndpoints:
    """Class containing API endpoints"""
    BASE_URL = "https://api.replicate.com/v1"
    PREDICTIONS = f"{BASE_URL}/models/black-forest-labs/flux-1.1-pro-ultra/predictions"
    
    @staticmethod
    def get_prediction_url(prediction_id: str) -> str:
        """Generate prediction URL for a given ID"""
        return f"{APIEndpoints.BASE_URL}/predictions/{prediction_id}"

class ImageFormat:
    """Class containing image format identifiers"""
    JPEG_IDENTIFIER = "/9j/"
    PNG_IDENTIFIER = "iVBOR"
    GIF_IDENTIFIER = "R0lG"
    WEBP_IDENTIFIER = "UklGR"

class Pipe:
    """
    FLUX.1.1-pro-ultra Manifold Function for image generation.
    
    This class provides an interface to the Black Forest Labs image generation model,
    allowing for customizable image generation with various parameters and safety controls.

    Features:
        - Multiple aspect ratio support
        - Configurable safety tolerance levels
        - Multiple output format options
        - Seed control for reproducible generations
        - Raw mode for less processed images

    Available Aspect Ratios:
        - Wide formats: "21:9", "16:9"
        - Standard formats: "3:2", "4:3", "5:4", "1:1"
        - Portrait formats: "4:5", "3:4", "2:3", "9:16", "9:21"

    Available Output Formats:
        - "jpg": Standard JPEG format (smaller file size)
        - "png": Lossless PNG format (larger file size, better quality)

    Safety Tolerance Levels:
        1: Maximum safety (strictest content filtering)
        2: Very high safety (default)
        3: High safety
        4: Medium safety
        5: Low safety
        6: Minimum safety (most permissive)
    """

    # Class constants
    AVAILABLE_ASPECT_RATIOS: List[AspectRatioType] = [
        "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", 
        "4:5", "3:4", "2:3", "9:16", "9:21"
    ]

    AVAILABLE_OUTPUT_FORMATS: List[OutputFormatType] = ["jpg", "png"]

    AVAILABLE_SAFETY_LEVELS: List[int] = list(range(1, 7))

    SAFETY_TOLERANCE_LEVELS: Dict[int, str] = {
        1: "Maximum safety (strictest filtering)",
        2: "Very high safety (default)",
        3: "High safety",
        4: "Medium safety",
        5: "Low safety",
        6: "Minimum safety (most permissive)",
    }

    # Request timeout settings
    CONNECT_TIMEOUT = 3.05  # seconds
    READ_TIMEOUT = 60  # seconds
    POLL_INTERVAL = 1  # seconds

    class Valves(BaseModel):
        """
        Configuration model for the Pipe class.

        This model validates and stores all configuration parameters needed for
        the image generation process.

        Attributes:
            REPLICATE_API_TOKEN (Optional[str]): Authentication token for Replicate API
            REPLICATE_API_BASE_URL (HttpUrl): Base URL for API requests
            FLUX_RAW_MODE (bool): Toggle for less processed image generation
            FLUX_SAFETY_TOLERANCE (int): Safety level for content filtering (1-6)
            FLUX_SEED (Optional[int]): Seed for reproducible generation
            FLUX_ASPECT_RATIO (str): Output image aspect ratio
            FLUX_OUTPUT_FORMAT (str): Output image format (jpg/png)
        """

        REPLICATE_API_TOKEN: Optional[str] = Field(
            default=None,
            description="Your API Token for Replicate"
        )
        REPLICATE_API_BASE_URL: HttpUrl = Field(
            default="https://api.replicate.com/v1",
            description="Base URL for the Replicate API"
        )
        FLUX_RAW_MODE: bool = Field(
            default=False,
            description="Generate less processed, more natural-looking images"
        )
        FLUX_SAFETY_TOLERANCE: SafetyToleranceType = Field(
            default=1,
            description="Safety tolerance levels for content filtering"
        )
        FLUX_SEED: Optional[int] = Field(
            default=None,
            description="Random seed for image generation. None/blank for random"
        )
        FLUX_ASPECT_RATIO: AspectRatioType = Field(
            default="1:1",
            description="Aspect ratio for the generated image"
        )
        FLUX_OUTPUT_FORMAT: OutputFormatType = Field(
            default="jpg",
            description="Output format for the generated image"
        )

        @validator("FLUX_SAFETY_TOLERANCE")
        def validate_safety_tolerance(cls, v: int) -> int:
            """Validate safety tolerance is within acceptable range"""
            if not 1 <= v <= 6:
                raise ValueError("Safety tolerance must be between 1 and 6")
            return v

        @validator("FLUX_ASPECT_RATIO")
        def validate_aspect_ratio(cls, v: str) -> str:
            """Validate aspect ratio is in available options"""
            if v not in Pipe.AVAILABLE_ASPECT_RATIOS:
                raise ValueError(f"Invalid aspect ratio. Must be one of: {Pipe.AVAILABLE_ASPECT_RATIOS}")
            return v

        @validator("FLUX_OUTPUT_FORMAT")
        def validate_output_format(cls, v: str) -> str:
            """Validate output format is in available options"""
            if v not in Pipe.AVAILABLE_OUTPUT_FORMATS:
                raise ValueError(f"Invalid output format. Must be one of: {Pipe.AVAILABLE_OUTPUT_FORMATS}")
            return v

    def __init__(self):
        """Initialize the Pipe instance with configuration from environment variables."""
        self.type = "manifold"
        self.id = "FLUX_1_1_PRO_ULTRA"
        self.name = "FLUX.1.1-pro-ultra: "
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN"),
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

    def get_img_extension(self, img_data: str) -> Optional[str]:
        """
        Determine the image extension based on the base64-encoded data.

        Args:
            img_data: Base64 encoded image data

        Returns:
            str or None: Image extension if recognized, None otherwise
        """
        if img_data.startswith(ImageFormat.JPEG_IDENTIFIER):
            return "jpeg"
        elif img_data.startswith(ImageFormat.PNG_IDENTIFIER):
            return "png"
        elif img_data.startswith(ImageFormat.GIF_IDENTIFIER):
            return "gif"
        elif img_data.startswith(ImageFormat.WEBP_IDENTIFIER):
            return "webp"
        return None

    def handle_image_response(self, response: requests.Response) -> str:
        """
        Process and format the image response from the API.

        Args:
            response: HTTP response containing image data

        Returns:
            str: Markdown formatted image string with base64 encoded data
        """
        content_type = response.headers.get("Content-Type", "")
        img_ext = "png"  # default extension
        if "image/" in content_type:
            img_ext = content_type.split("/")[-1]
        
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return f"![Image](data:{content_type};base64,{image_base64})\n`GeneratedImage.{img_ext}`"

    def create_headers(self) -> Dict[str, str]:
        """Create headers for API requests"""
        if not self.valves.REPLICATE_API_TOKEN:
            raise ValueError("REPLICATE_API_TOKEN is required but not provided")
        
        return {
            "Authorization": f"Bearer {self.valves.REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }

    def create_payload(self, prompt: str) -> Dict[str, Any]:
        """Create payload for API requests"""
        payload = {
            "input": {
                "prompt": prompt,
                "aspect_ratio": self.valves.FLUX_ASPECT_RATIO,
                "output_format": self.valves.FLUX_OUTPUT_FORMAT,
                "safety_tolerance": self.valves.FLUX_SAFETY_TOLERANCE,
                "raw": self.valves.FLUX_RAW_MODE,
            }
        }
        if self.valves.FLUX_SEED is not None:
            payload["input"]["seed"] = self.valves.FLUX_SEED
        return payload

    def poll_prediction(self, prediction_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Poll the API for prediction status until completion.

        Args:
            prediction_id: ID of the prediction to poll
            headers: Request headers

        Returns:
            Dict containing the prediction result
        """
        while True:
            response = requests.get(
                APIEndpoints.get_prediction_url(prediction_id),
                headers=headers,
            )
            prediction = response.json()
            status = prediction["status"]
            
            if status == PredictionStatus.FAILED:
                raise RuntimeError(f"Generation failed: {prediction.get('error', 'Unknown error')}")
            elif status in [PredictionStatus.SUCCEEDED, PredictionStatus.CANCELED]:
                return prediction
                
            time.sleep(self.POLL_INTERVAL)

    def stream_response(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Handle streaming response"""
        yield self.non_stream_response(headers, payload)

    def non_stream_response(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> str:
        """
        Process non-streaming response from the API.

        Args:
            headers: Request headers
            payload: Request payload

        Returns:
            str: Generated image in markdown format or error message
        """
        try:
            # Create prediction
            response = requests.post(
                url=APIEndpoints.PREDICTIONS,
                headers=headers,
                json=payload,
                timeout=(self.CONNECT_TIMEOUT, self.READ_TIMEOUT),
            )
            response.raise_for_status()
            prediction = response.json()

            # Poll for completion
            prediction = self.poll_prediction(prediction["id"], headers)

            # Handle the completed prediction
            if prediction["status"] == PredictionStatus.SUCCEEDED:
                image_url = prediction["output"]
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                return self.handle_image_response(img_response)

            return "Error: Image generation failed"

        except requests.exceptions.RequestException as e:
            return f"Error: Request failed: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def pipe(
        self, body: Dict[str, Any]
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        """
        Process the pipe request.

        Args:
            body: Request body containing messages and stream flag

        Returns:
            Generated image result or error message
        """
        try:
            headers = self.create_headers()
            prompt = get_last_user_message(body["messages"])
            payload = self.create_payload(prompt)

            if body.get("stream", False):
                return self.stream_response(headers, payload)
            else:
                return self.non_stream_response(headers, payload)
        except Exception as e:
            return f"Error: {str(e)}"

    def pipes(self) -> List[Dict[str, str]]:
        """Get available pipes configuration"""
        return [{"id": "flux_1_1_pro_ultra", "name": "Flux 1.1 Pro Ultra"}]


def validate_env_int(value: str, default: int, min_val: int, max_val: int) -> int:
    """
    Validate and convert environment variable to integer within specified range.

    Args:
        value: String value to validate
        default: Default value if conversion fails
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        int: Validated integer value
    """
    try:
        val = int(value)
        return max(min_val, min(val, max_val))
    except (TypeError, ValueError):
        return default
