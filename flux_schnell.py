"""
title: FLUX Schnell Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 1.0
license: MIT
requirements: pydantic>=2.0.0, requests>=2.31.0
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

from typing import Dict, Generator, Iterator, Union, Optional, Literal, cast
from pydantic import BaseModel, Field
import os
import base64
import requests
import time

AspectRatioType = Literal["1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4", "4:3", "9:16", "9:21"]
OutputFormatType = Literal["webp", "jpg", "png"]
MegapixelsType = Literal["1", "0.25"]

class Pipe:
    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(default="")
        FLUX_SEED: Optional[int] = Field(default=None)
        FLUX_ASPECT_RATIO: AspectRatioType = Field(default="1:1")
        FLUX_OUTPUT_FORMAT: OutputFormatType = Field(default="webp")
        FLUX_GO_FAST: bool = Field(default=True)
        FLUX_MEGAPIXELS: MegapixelsType = Field(default="1")
        FLUX_NUM_OUTPUTS: int = Field(default=1, ge=1, le=4)
        FLUX_OUTPUT_QUALITY: int = Field(default=80, ge=0, le=100)
        FLUX_NUM_INFERENCE_STEPS: int = Field(default=4, ge=1, le=4)
        FLUX_DISABLE_SAFETY_CHECKER: bool = Field(default=False)

    def __init__(self):
        self.type = "pipe"
        self.id = "flux_schnell"
        self.name = "Flux Schnell"
        self.MODEL_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
        self.session = requests.Session()
        self.request_count = 0
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

    def _log_request(self, method: str, url: str):
        if not "replicate.delivery" in url:
            self.request_count += 1
            print(f"Request #{self.request_count}: {method} {url}")

    def _process_image(self, url_or_data: str) -> str:
        if url_or_data.startswith('http'):
            response = self.session.get(url_or_data, timeout=30)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode('utf-8')
            content_type = response.headers.get("Content-Type", f"image/{self.valves.FLUX_OUTPUT_FORMAT}")
            return f"![Image](data:{content_type};base64,{image_data})\n`GeneratedImage.{self.valves.FLUX_OUTPUT_FORMAT}`"
        return f"![Image]({url_or_data})\n`GeneratedImage.{self.valves.FLUX_OUTPUT_FORMAT}`"

    def _wait_for_completion(self, prediction_url: str, max_retries: int = 1) -> Dict:
        headers = {
            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
            "Accept": "application/json",
            "Prefer": "wait=30"
        }
        
        for i in range(max_retries):
            self._log_request("GET", prediction_url)
            response = self.session.get(prediction_url, headers=headers, timeout=35)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") in ["succeeded", "failed", "canceled"]:
                return result
            elif i < max_retries - 1:
                time.sleep(5)
        
        return result

    def pipe(self, body: Dict) -> Union[str, Generator[str, None, None], Iterator[str]]:
        self.request_count = 0
        if not self.valves.REPLICATE_API_TOKEN:
            return "Error: REPLICATE_API_TOKEN is required"

        try:
            prompt = body.get("messages", [{}])[-1].get("content", "")
            if not prompt:
                return "Error: No prompt provided"

            input_params = {"prompt": prompt}
            if self.valves.FLUX_SEED is not None:
                input_params["seed"] = self.valves.FLUX_SEED
            if self.valves.FLUX_ASPECT_RATIO != "1:1":
                input_params["aspect_ratio"] = self.valves.FLUX_ASPECT_RATIO
            if self.valves.FLUX_OUTPUT_FORMAT != "webp":
                input_params["output_format"] = self.valves.FLUX_OUTPUT_FORMAT
            if not self.valves.FLUX_GO_FAST:
                input_params["go_fast"] = False
            if self.valves.FLUX_MEGAPIXELS != "1":
                input_params["megapixels"] = self.valves.FLUX_MEGAPIXELS
            if self.valves.FLUX_NUM_OUTPUTS != 1:
                input_params["num_outputs"] = self.valves.FLUX_NUM_OUTPUTS
            if self.valves.FLUX_OUTPUT_QUALITY != 80:
                input_params["output_quality"] = self.valves.FLUX_OUTPUT_QUALITY
            if self.valves.FLUX_NUM_INFERENCE_STEPS != 4:
                input_params["num_inference_steps"] = self.valves.FLUX_NUM_INFERENCE_STEPS
            if self.valves.FLUX_DISABLE_SAFETY_CHECKER:
                input_params["disable_safety_checker"] = True

            headers = {
                "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Prefer": "wait=30"
            }
            
            self._log_request("POST", self.MODEL_URL)
            response = self.session.post(self.MODEL_URL, headers=headers, json={"input": input_params}, timeout=35)
            response.raise_for_status()
            prediction = response.json()

            result = self._wait_for_completion(prediction["urls"]["get"])
            
            if result.get("status") == "failed":
                raise Exception(f"Generation failed: {result.get('error', 'Unknown error')}")
            elif result.get("status") == "canceled":
                raise Exception("Generation was canceled")
            elif result.get("status") == "processing":
                return "Image is still processing. Please try again in a moment."
            
            output = result.get("output")
            if isinstance(output, list) and output:
                output = output[0]
            
            if not output:
                return "Image is still processing. Please try again in a moment."

            return self._process_image(output)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {str(e)}"
            if e.response.status_code == 422:
                error_msg = f"Error 422: {e.response.json().get('detail', str(e))}"
            return error_msg
        except Exception as e:
            return f"Error: {str(e)}"
