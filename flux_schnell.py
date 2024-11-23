"""
title: FLUX Schnell Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 1.2
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.1
environment_variables: 
    - REPLICATE_API_TOKEN (required)
    - FLUX_SEED (optional)
    - FLUX_ASPECT_RATIO (optional, default: "1:1")
    - FLUX_OUTPUT_FORMAT (optional, default: "webp")
    - FLUX_GO_FAST (optional, default: true)
    - FLUX_MEGAPIXELS (optional, default: "1")
    - FLUX_NUM_OUTPUTS (optional, default: 1)
    - FLUX_OUTPUT_QUALITY (optional, default: 80)
    - FLUX_NUM_INFERENCE_STEPS (optional, default: 4)
    - FLUX_DISABLE_SAFETY_CHECKER (optional, default: false)
supported providers: replicate.com

NOTE: Due to the asynchronous nature of the Replicate API, each image generation will make 2-3 (rare occasion 4) API requests:
1. Initial request to start generation
2. Follow-up request(s) to check completion status
This is normal behavior and required by the API design. You will typically see only 2 requests after the first generation.

NOTE: If you first image is a PNG file - your thread will appear blank on the left hand side. Overall PNG files slow down the 
interface for some reason. Generate and save your images, then delete the thread.

NOTE: "Fluidly stream large external response chunks" must be set to OFF in the interface.
"""

from typing import Dict, AsyncIterator, Optional, Literal, cast, Union
from pydantic import BaseModel, Field
import os
import base64
import aiohttp
import asyncio
from datetime import datetime

AspectRatioType = Literal["1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4", "4:3", "9:16", "9:21"]
OutputFormatType = Literal["webp", "jpg", "png"]
MegapixelsType = Literal["1", "0.25"]

class Pipe:
    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(
            default="",
            description="API token for Replicate.com"
        )
        FLUX_SEED: Optional[int] = Field(
            default=None,
            description="Seed for reproducible generations"
        )
        FLUX_ASPECT_RATIO: AspectRatioType = Field(
            default="1:1",
            description="Aspect ratio of generated images"
        )
        FLUX_OUTPUT_FORMAT: OutputFormatType = Field(
            default="webp",
            description="Output format for generated images"
        )
        FLUX_GO_FAST: bool = Field(
            default=True,
            description="Enable fast mode for quicker generations"
        )
        FLUX_MEGAPIXELS: MegapixelsType = Field(
            default="1",
            description="Resolution in megapixels"
        )
        FLUX_NUM_OUTPUTS: int = Field(
            default=1,
            ge=1,
            le=4,
            description="Number of images to generate (1-4)"
        )
        FLUX_OUTPUT_QUALITY: int = Field(
            default=80,
            ge=0,
            le=100,
            description="Output image quality (0-100)"
        )
        FLUX_NUM_INFERENCE_STEPS: int = Field(
            default=4,
            ge=1,
            le=4,
            description="Number of inference steps (1-4)"
        )
        FLUX_DISABLE_SAFETY_CHECKER: bool = Field(
            default=False,
            description="Disable safety checker for image generation"
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "flux_schnell"
        self.name = "Flux Schnell"
        self.MODEL_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN", ""),
            FLUX_SEED=int(os.getenv("FLUX_SEED")) if os.getenv("FLUX_SEED") else None,
            FLUX_ASPECT_RATIO=cast(AspectRatioType, os.getenv("FLUX_ASPECT_RATIO", "1:1")),
            FLUX_OUTPUT_FORMAT=cast(OutputFormatType, os.getenv("FLUX_OUTPUT_FORMAT", "webp")),
            FLUX_GO_FAST=bool(os.getenv("FLUX_GO_FAST", True)),
            FLUX_MEGAPIXELS=cast(MegapixelsType, os.getenv("FLUX_MEGAPIXELS", "1")),
            FLUX_NUM_OUTPUTS=int(os.getenv("FLUX_NUM_OUTPUTS", "1")),
            FLUX_OUTPUT_QUALITY=int(os.getenv("FLUX_OUTPUT_QUALITY", "80")),
            FLUX_NUM_INFERENCE_STEPS=int(os.getenv("FLUX_NUM_INFERENCE_STEPS", "4")),
            FLUX_DISABLE_SAFETY_CHECKER=bool(os.getenv("FLUX_DISABLE_SAFETY_CHECKER", False))
        )

    def _get_status(self, message: str) -> str:
        """Format a status message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {message}"

    async def _process_image(self, url: str, session: aiohttp.ClientSession) -> str:
        """Process and format image data."""
        try:
            async with session.get(url, timeout=30) as response:
                response.raise_for_status()
                image_data = base64.b64encode(await response.read()).decode('utf-8')
                content_type = response.headers.get("Content-Type", f"image/{self.valves.FLUX_OUTPUT_FORMAT}")
                return f"![Image](data:{content_type};base64,{image_data})\n`GeneratedImage.{self.valves.FLUX_OUTPUT_FORMAT}`"
        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")

    async def _wait_for_completion(self, prediction_url: str, __event_emitter__=None) -> Dict:
        headers = {
            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
            "Accept": "application/json",
            "Prefer": "wait=30"  # Tell API to wait up to 30 seconds before responding
        }
        
        async with aiohttp.ClientSession() as session:
            # Initial delay with buffer for network latency and variation
            await asyncio.sleep(6)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Checking generation status...", "done": False}
                })
            
            # Check after initial delay
            async with session.get(prediction_url, headers=headers, timeout=35) as response:
                response.raise_for_status()
                result = await response.json()
                status = result.get("status")
                
                if status in ["succeeded", "failed", "canceled"]:
                    return result
                
                # If not complete, wait a bit longer and try again
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Generation in progress... almost there!", "done": False}
                    })
                
                await asyncio.sleep(5)  # Extended wait for final check
                
                async with session.get(prediction_url, headers=headers, timeout=35) as response:
                    response.raise_for_status()
                    result = await response.json()
                    status = result.get("status")
                    
                    if status in ["succeeded", "failed", "canceled"]:
                        return result
                    
                    # One last attempt if still not complete
                    await asyncio.sleep(5)
                    async with session.get(prediction_url, headers=headers, timeout=35) as response:
                        response.raise_for_status()
                        final_result = await response.json()
                        final_status = final_result.get("status")
                        
                        if final_status in ["succeeded", "failed", "canceled"]:
                            return final_result
                        else:
                            raise Exception(f"Generation incomplete after {final_status} status")

    async def pipe(self, body: Dict, __event_emitter__=None) -> Union[str, AsyncIterator[str]]:
        """Generate images using the Flux model."""
        if not self.valves.REPLICATE_API_TOKEN:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Error: REPLICATE_API_TOKEN is required", "done": True}
                })
            return "Error: REPLICATE_API_TOKEN is required"

        try:
            # Get prompt from messages
            prompt = body.get("messages", [{}])[-1].get("content", "")
            if not prompt:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Error: No prompt provided", "done": True}
                    })
                return "Error: No prompt provided"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Starting image generation...", "done": False}
                })

            # Prepare API parameters
            input_params = {
                "prompt": prompt,
                "seed": self.valves.FLUX_SEED,
                "aspect_ratio": self.valves.FLUX_ASPECT_RATIO,
                "output_format": self.valves.FLUX_OUTPUT_FORMAT,
                "go_fast": self.valves.FLUX_GO_FAST,
                "megapixels": self.valves.FLUX_MEGAPIXELS,
                "num_outputs": self.valves.FLUX_NUM_OUTPUTS,
                "output_quality": self.valves.FLUX_OUTPUT_QUALITY,
                "num_inference_steps": self.valves.FLUX_NUM_INFERENCE_STEPS,
                "disable_safety_checker": self.valves.FLUX_DISABLE_SAFETY_CHECKER
            }
            input_params = {k: v for k, v in input_params.items() if v is not None}

            headers = {
                "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Start generation
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Sending request to Replicate API...", "done": False}
                    })

                try:
                    async with session.post(
                        self.MODEL_URL,
                        headers=headers,
                        json={"input": input_params}
                    ) as response:
                        response.raise_for_status()
                        prediction = await response.json()
                except aiohttp.ClientError as e:
                    if hasattr(e, 'status') and e.status == 422:
                        error_detail = await response.json()
                        raise Exception(f"API Error: {error_detail.get('detail', str(e))}")
                    raise Exception(f"API Error: {str(e)}")

                # Poll for completion
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Request accepted, waiting for generation...", "done": False}
                    })

                prediction_url = prediction["urls"]["get"]
                headers["Prefer"] = "wait=30"  # Tell API to wait up to 30 seconds before responding
                
                result = await self._wait_for_completion(prediction_url, __event_emitter__)

                if result.get("status") == "succeeded":
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "Generation completed successfully!", "done": False}
                        })

                    outputs = result.get("output", [])
                    if not outputs:
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {"description": "Error: No output generated", "done": True}
                            })
                        return "Error: No output generated"

                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "Processing generated images...", "done": False}
                        })

                    # Process all images and combine them
                    processed_images = []
                    for output in outputs:
                        processed_image = await self._process_image(output, session)
                        processed_images.append(processed_image)

                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "Generation complete!", "done": True}
                        })

                    # Join all images with newlines
                    return "\n".join(processed_images)

                elif result.get("status") in ["failed", "canceled"]:
                    error = result.get("error", "Unknown error")
                    raise Exception(f"Generation {result.get('status')}: {error}")
                else:
                    raise Exception("Generation is taking longer than expected")

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Error: {str(e)}", "done": True}
                })
            return f"Error: {str(e)}"
