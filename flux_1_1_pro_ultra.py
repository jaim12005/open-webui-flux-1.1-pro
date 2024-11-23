"""
title: FLUX.1.1 Pro Ultra Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 2.2
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.0
environment_variables: 
    - REPLICATE_API_TOKEN (required)
    - FLUX_RAW_MODE (optional, default: false)
    - FLUX_SAFETY_TOLERANCE (optional, default: 2)
    - FLUX_SEED (optional)
    - FLUX_ASPECT_RATIO (optional, default: "1:1")
    - FLUX_OUTPUT_FORMAT (optional, default: "jpg")
supported providers: replicate.com

NOTE: Due to the asynchronous nature of the Replicate API, each image generation will make 2-3 (rare occasion 4) API requests:
1. Initial request to start generation
2. Follow-up request(s) to check completion status
This is normal behavior and required by the API design. You will typically see only 2 requests after the first generation.

NOTE: If you first image is a PNG file - your thread will appear blank on the left hand side. Overall PNG files slow down the 
interface for some reason. Generate and save your images, then delete the thread.

NOTE: "Fluidly stream large external response chunks" must be set to OFF in the interface.
"""

from typing import Dict, Generator, Iterator, Union, Optional, Literal
from pydantic import BaseModel, Field
import os
import base64
import aiohttp
import asyncio
import json

AspectRatioType = Literal[
    "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"
]
OutputFormatType = Literal["jpg", "png"]
SafetyToleranceType = Literal[1, 2, 3, 4, 5, 6]


class Pipe:
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
        )

    async def _process_image(self, url_or_data: str) -> str:
        if url_or_data.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.get(url_or_data, timeout=30) as response:
                    response.raise_for_status()
                    image_data = base64.b64encode(await response.read()).decode("utf-8")
                    content_type = response.headers.get(
                        "Content-Type", f"image/{self.valves.FLUX_OUTPUT_FORMAT}"
                    )
                    return f"![Image](data:{content_type};base64,{image_data})\n`GeneratedImage.{self.valves.FLUX_OUTPUT_FORMAT}`"
        return f"![Image]({url_or_data})\n`GeneratedImage.{self.valves.FLUX_OUTPUT_FORMAT}`"

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
            return "Error: REPLICATE_API_TOKEN is required"

        try:
            prompt = body.get("messages", [{}])[-1].get("content", "")
            if not prompt:
                return "Error: No prompt provided"

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

            input_params = {"prompt": prompt}
            if self.valves.FLUX_RAW_MODE:
                input_params["raw"] = True
            if self.valves.FLUX_SAFETY_TOLERANCE != 2:
                input_params["safety_tolerance"] = self.valves.FLUX_SAFETY_TOLERANCE
            if self.valves.FLUX_SEED is not None:
                input_params["seed"] = self.valves.FLUX_SEED
            if self.valves.FLUX_ASPECT_RATIO != "1:1":
                input_params["aspect_ratio"] = self.valves.FLUX_ASPECT_RATIO
            if self.valves.FLUX_OUTPUT_FORMAT != "jpg":
                input_params["output_format"] = self.valves.FLUX_OUTPUT_FORMAT

            headers = {
                "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.MODEL_URL,
                    headers=headers,
                    json={"input": input_params},
                    timeout=35,
                ) as response:
                    response.raise_for_status()
                    prediction = await response.json()

                result = await self._wait_for_completion(
                    prediction["urls"]["get"], __event_emitter__
                )

                if result.get("status") == "failed":
                    raise Exception(
                        f"Generation failed: {result.get('error', 'Unknown error')}"
                    )
                elif result.get("status") == "canceled":
                    raise Exception("Generation was canceled")
                elif result.get("status") == "processing":
                    return "Image is still processing. Please try again in a moment."

                output = result.get("output")
                if isinstance(output, list) and output:
                    output = output[0]

                if not output:
                    return "Image is still processing. Please try again in a moment."

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Processing generated image...",
                                "done": False,
                            },
                        }
                    )

                result = await self._process_image(output)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Generation complete!",
                                "done": True,
                            },
                        }
                    )

                return result

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )
            return f"Error: {str(e)}"
