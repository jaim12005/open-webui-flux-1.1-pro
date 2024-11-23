"""
title: FLUX.1.1 Pro Ultra Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 2.4
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.0
environment_variables: 
    - REPLICATE_API_TOKEN (required)
    - FLUX_RAW_MODE (optional, default: false)
    - FLUX_SAFETY_TOLERANCE (optional, default: 2)
    - FLUX_SEED (optional)
    - FLUX_ASPECT_RATIO (optional, default: "1:1")
    - FLUX_OUTPUT_FORMAT (optional, default: "jpg")
    - FLUX_IMAGE_SIZE (optional, default: "1024x1024")
supported providers: replicate.com

NOTE: Due to the asynchronous nature of the Replicate API, each image generation will make 2-3 (rare occasion 4) API requests:
1. Initial request to start generation
2. Follow-up request(s) to check completion status
This is normal behavior and required by the API design. You will typically see only 2 requests after the first generation.

NOTE: If you first image is a PNG file - your thread will appear blank on the left hand side. Overall PNG files slow down the 
interface for some reason. Generate and save your images, then delete the thread.

NOTE: "Fluidly stream large external response chunks" must be set to OFF in the interface.

NOTE: This version supports both fluid streaming and non-streaming modes.
"""

from typing import Dict, Generator, Iterator, Union, Optional, Literal, Tuple, List
from pydantic import BaseModel, Field
import os
import base64
import aiohttp
import asyncio
import json
import uuid
import time

AspectRatioType = Literal[
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"
]
OutputFormatType = Literal["jpg", "png"]
SafetyToleranceType = Literal[1, 2, 3, 4, 5, 6]


class Pipe:
    """A pipe that generates images using Black Forest Lab's Image Generation Models."""

    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(
            default="", description="Your Replicate API token"
        )
        FLUX_RAW_MODE: bool = Field(
            default=False, description="Enable raw mode for direct prompt input"
        )
        FLUX_SAFETY_TOLERANCE: SafetyToleranceType = Field(
            default=2, description="Safety filter strength (1-6)"
        )
        FLUX_SEED: Optional[int] = Field(
            default=None, description="Random seed for reproducible generations"
        )
        FLUX_ASPECT_RATIO: AspectRatioType = Field(
            default="1:1", description="Output image aspect ratio"
        )
        FLUX_OUTPUT_FORMAT: OutputFormatType = Field(
            default="jpg", description="Output image format"
        )
        FLUX_IMAGE_SIZE: str = Field(
            default="1024x1024", description="Output image size (width x height)"
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "flux-1.1-pro-ultra"
        self.name = "Flux 1.1 Pro Ultra"
        self.MODEL_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro-ultra/predictions"
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN", ""),
            FLUX_RAW_MODE=bool(os.getenv("FLUX_RAW_MODE", False)),
            FLUX_SAFETY_TOLERANCE=int(os.getenv("FLUX_SAFETY_TOLERANCE", "2")),
            FLUX_SEED=int(os.getenv("FLUX_SEED")) if os.getenv("FLUX_SEED") else None,
            FLUX_ASPECT_RATIO=os.getenv("FLUX_ASPECT_RATIO", "1:1"),
            FLUX_OUTPUT_FORMAT=os.getenv("FLUX_OUTPUT_FORMAT", "jpg"),
            FLUX_IMAGE_SIZE=os.getenv("FLUX_IMAGE_SIZE", "1024x1024"),
        )

    async def _process_image(self, url_or_data: str, prompt: str, params: Dict, stream: bool = True) -> Union[str, List[str]]:
        """Process image data and return it in SSE format.
        
        Args:
            url_or_data (str): Either a URL to an image or base64 encoded image data
            prompt (str): The original prompt used to generate the image
            params (Dict): Generation parameters used
            stream (bool): Whether to stream the response in chunks
            
        Returns:
            Union[str, List[str]]: SSE formatted response(s)
        """
        if url_or_data.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.get(url_or_data, timeout=30) as response:
                    response.raise_for_status()
                    image_data = base64.b64encode(await response.read()).decode("utf-8")
                    content_type = response.headers.get(
                        "Content-Type", f"image/{self.valves.FLUX_OUTPUT_FORMAT}"
                    )
                    image_url = f"data:{content_type};base64,{image_data}"
        else:
            image_url = url_or_data

        # Split the response into smaller chunks for fluid streaming
        responses = []
        
        # First chunk: Image container opening and image
        image_chunk = f'''
<div class="generated-image-container">
    <img src="{image_url}" alt="Generated Image" style="max-width: 100%; height: auto; border-radius: 8px; margin-bottom: 8px;" />'''
        
        chunk_data = {
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "flux-1.1-pro-ultra",
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": image_chunk,
                    "content_type": "text/html"
                },
                "index": 0,
                "finish_reason": None
            }]
        }
        responses.append(f"data: {json.dumps(chunk_data)}\n\n")
        
        # Second chunk: Metadata section
        metadata_chunk = f'''
    <div class="image-metadata" style="font-size: 0.9em; color: var(--text-gray-600); dark:color: var(--text-gray-400);">
        <details>
            <summary style="cursor: pointer; user-select: none;">Generation Details</summary>
            <div style="padding: 8px; margin-top: 4px; background: var(--bg-gray-50); dark:background: var(--bg-gray-800); border-radius: 6px;">
                <p><strong>Prompt:</strong> {prompt}</p>
                <p><strong>Parameters:</strong></p>
                <ul style="list-style-type: none; padding-left: 12px;">
                    <li>• Size: {params.get("width", "1024")}x{params.get("height", "1024")}</li>
                    <li>• Aspect Ratio: {params.get("aspect_ratio", self.valves.FLUX_ASPECT_RATIO)}</li>
                    <li>• Format: {params.get("output_format", self.valves.FLUX_OUTPUT_FORMAT)}</li>
                    <li>• Safety Level: {params.get("safety_tolerance", self.valves.FLUX_SAFETY_TOLERANCE)}</li>
                    <li>• Seed: {params.get("seed", "Random")}</li>
                </ul>
            </div>
        </details>
    </div>
</div>'''
        
        chunk_data = {
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "flux-1.1-pro-ultra",
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": metadata_chunk,
                    "content_type": "text/html"
                },
                "index": 0,
                "finish_reason": None
            }]
        }
        responses.append(f"data: {json.dumps(chunk_data)}\n\n")
        
        # Final chunk: Finish marker
        finish_data = {
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "flux-1.1-pro-ultra",
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        responses.append(f"data: {json.dumps(finish_data)}\n\n")
        responses.append("data: [DONE]\n\n")
        
        # If fluid streaming is enabled, return list of responses
        # Otherwise concatenate them into a single response
        return responses if stream else "".join(responses)

    async def _wait_for_completion(
        self, prediction_url: str, __event_emitter__=None
    ) -> Dict:
        headers = {
            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
            "Accept": "application/json",
            "Prefer": "wait=30",  # Tell API to wait up to 30 seconds before responding
        }

        async with aiohttp.ClientSession() as session:
            # Initial delay with buffer for network latency and variation
            await asyncio.sleep(12)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Checking generation status...",
                            "done": False,
                        },
                    }
                )

            # Check after initial delay
            async with session.get(
                prediction_url, headers=headers, timeout=35
            ) as response:
                response.raise_for_status()
                result = await response.json()
                status = result.get("status")

                if status in ["succeeded", "failed", "canceled"]:
                    return result

                # If not complete, wait a bit longer and try again
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Generation in progress... almost there!",
                                "done": False,
                            },
                        }
                    )

                await asyncio.sleep(5)  # Extended wait for final check

                async with session.get(
                    prediction_url, headers=headers, timeout=35
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    status = result.get("status")

                    if status in ["succeeded", "failed", "canceled"]:
                        return result

                    # One last attempt if still not complete
                    await asyncio.sleep(5)
                    async with session.get(
                        prediction_url, headers=headers, timeout=35
                    ) as response:
                        response.raise_for_status()
                        final_result = await response.json()
                        final_status = final_result.get("status")

                        if final_status in ["succeeded", "failed", "canceled"]:
                            return final_result
                        else:
                            raise Exception(
                                f"Generation incomplete after {final_status} status"
                            )

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, Generator[str, None, None]]:
        if not self.valves.REPLICATE_API_TOKEN:
            yield "Error: REPLICATE_API_TOKEN is required"
            return

        try:
            prompt = body.get("messages", [{}])[-1].get("content", "")
            if not prompt:
                yield "Error: No prompt provided"
                return

            # Parse image size from environment variable
            try:
                width, height = map(int, self.valves.FLUX_IMAGE_SIZE.split("x"))
            except (ValueError, AttributeError):
                width = height = 1024  # Default to 1024x1024 if parsing fails
                
            input_params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "aspect_ratio": self.valves.FLUX_ASPECT_RATIO,
                "output_format": self.valves.FLUX_OUTPUT_FORMAT,
                "safety_tolerance": self.valves.FLUX_SAFETY_TOLERANCE,
            }

            if self.valves.FLUX_SEED:
                input_params["seed"] = int(self.valves.FLUX_SEED)

            if self.valves.FLUX_RAW_MODE:
                input_params["raw"] = True

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Starting image generation...",
                            "done": False,
                        },
                    }
                )

            async with aiohttp.ClientSession() as session:
                # Create prediction
                headers = {
                    "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                }
                
                async with session.post(
                    self.MODEL_URL,
                    headers=headers,
                    json={"input": input_params},
                ) as response:
                    if response.status != 201:
                        error_msg = f"Error creating prediction: {await response.text()}"
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": error_msg, "done": True},
                                }
                            )
                        yield error_msg
                        return

                    prediction = await response.json()

                # Get prediction result
                prediction_result = await self._wait_for_completion(
                    prediction["urls"]["get"], __event_emitter__
                )

                if prediction_result.get("status") == "failed":
                    error_msg = f"Generation failed: {prediction_result.get('error')}"
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": error_msg, "done": True},
                            }
                        )
                    yield error_msg
                    return

                # Debug the prediction result
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Debug - Result: {prediction_result}",
                                "done": False
                            },
                        }
                    )

                # Get the output URL directly from the prediction result
                image_url = prediction_result.get("output")
                if not image_url:
                    error_msg = "No output URL in prediction result"
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": error_msg, "done": True},
                            }
                        )
                    yield error_msg
                    return

                # Send the image as a message
                if __event_emitter__:
                    # First send the image
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"![Generated Image]({image_url})",
                                "content_type": "text/markdown"
                            },
                        }
                    )
                    
                    # Then send the metadata
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"""<details>
<summary>Generation Details</summary>

- **Prompt:** {prompt}
- **Size:** {width}x{height}
- **Aspect Ratio:** {input_params["aspect_ratio"]}
- **Format:** {input_params["output_format"]}
- **Safety Level:** {input_params["safety_tolerance"]}
- **Seed:** {input_params.get("seed", "Random")}
</details>""",
                                "content_type": "text/markdown"
                            },
                        }
                    )

                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Image generated successfully!",
                                "done": True,
                            },
                        }
                    )

                # Return empty string since we've emitted the message
                yield ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            yield error_msg
