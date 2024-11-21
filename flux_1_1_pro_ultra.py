"""
title: FLUX.1.1 Pro Ultra Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 2.0
license: MIT
requirements: pydantic, requests
environment_variables: REPLICATE_API_TOKEN
supported providers: replicate.com
"""

import base64
import os
from typing import Dict, Generator, Iterator, Union, Optional, Literal, cast
import requests
from requests.adapters import HTTPAdapter
from open_webui.utils.misc import get_last_user_message
from pydantic import BaseModel, Field

AspectRatioType = Literal["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]
OutputFormatType = Literal["jpg", "png"]
SafetyToleranceType = Literal[1, 2, 3, 4, 5, 6]

class Pipe:
    BASE_URL = "https://api.replicate.com/v1"
    MODEL_URL = f"{BASE_URL}/models/black-forest-labs/flux-1.1-pro-ultra/predictions"
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=3))

    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(default="")
        FLUX_RAW_MODE: bool = Field(default=False)
        FLUX_SAFETY_TOLERANCE: SafetyToleranceType = Field(default=2)
        FLUX_SEED: Optional[int] = Field(default=None)
        FLUX_ASPECT_RATIO: AspectRatioType = Field(default="1:1")
        FLUX_OUTPUT_FORMAT: OutputFormatType = Field(default="jpg")

        @property
        def headers(self) -> Dict[str, str]:
            return {"Authorization": f"Bearer {self.REPLICATE_API_TOKEN}", "Content-Type": "application/json", "Prefer": "wait=30"}

    def __init__(self):
        self.type, self.id, self.name = "manifold", "FLUX_1_1_PRO_ULTRA", "FLUX.1.1-pro-ultra"
        self.valves = self.Valves(
            REPLICATE_API_TOKEN=os.getenv("REPLICATE_API_TOKEN", ""),
            FLUX_RAW_MODE=os.getenv("FLUX_RAW_MODE", "false").lower() == "true",
            FLUX_SAFETY_TOLERANCE=cast(SafetyToleranceType, int(os.getenv("FLUX_SAFETY_TOLERANCE", "2"))),
            FLUX_SEED=int(os.getenv("FLUX_SEED")) if os.getenv("FLUX_SEED") else None,
            FLUX_ASPECT_RATIO=cast(AspectRatioType, os.getenv("FLUX_ASPECT_RATIO", "1:1")),
            FLUX_OUTPUT_FORMAT=cast(OutputFormatType, os.getenv("FLUX_OUTPUT_FORMAT", "jpg"))
        )

    def handle_image_response(self, response: requests.Response) -> str:
        if not response.content:
            return "Error: Empty image content"
        try:
            image_base64 = base64.b64encode(response.content).decode("utf-8")
            fmt = self.valves.FLUX_OUTPUT_FORMAT
            return f"![Generated Image](data:image/{fmt};base64,{image_base64})\n`GeneratedImage.{fmt}`"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_image(self, prompt: str) -> str:
        if not self.valves.REPLICATE_API_TOKEN:
            return "Error: REPLICATE_API_TOKEN is required"

        try:
            input_params = {"prompt": prompt}
            if self.valves.FLUX_RAW_MODE:
                input_params["raw_mode"] = True
            if self.valves.FLUX_SAFETY_TOLERANCE != 2:
                input_params["safety_tolerance"] = self.valves.FLUX_SAFETY_TOLERANCE
            if self.valves.FLUX_SEED is not None:
                input_params["seed"] = self.valves.FLUX_SEED
            if self.valves.FLUX_ASPECT_RATIO != "1:1":
                input_params["aspect_ratio"] = self.valves.FLUX_ASPECT_RATIO
            if self.valves.FLUX_OUTPUT_FORMAT != "jpg":
                input_params["output_format"] = self.valves.FLUX_OUTPUT_FORMAT
            
            response = self.session.post(
                self.MODEL_URL,
                headers=self.valves.headers,
                json={"input": input_params}
            )
            response.raise_for_status()
            prediction = response.json()

            if prediction.get("status") == "succeeded" and prediction.get("output"):
                img_response = self.session.get(prediction["output"])
                img_response.raise_for_status()
                return self.handle_image_response(img_response)
            
            prediction_id = prediction.get("id")
            if not prediction_id:
                print(f"Debug - Prediction Response: {prediction}")
                return "Error: Invalid prediction response"

            while True:
                prediction = self.session.get(
                    f"{self.BASE_URL}/predictions/{prediction_id}",
                    headers=self.valves.headers
                ).json()
                
                if prediction["status"] == "succeeded":
                    img_response = self.session.get(prediction["output"], stream=True)
                    img_response.raise_for_status()
                    content = b''.join(chunk for chunk in img_response.iter_content(chunk_size=8192))
                    complete_response = requests.Response()
                    complete_response._content = content
                    complete_response.status_code = img_response.status_code
                    complete_response.headers = img_response.headers
                    return self.handle_image_response(complete_response)
                    
                if prediction["status"] in ["failed", "canceled"]:
                    return f"Error: Generation {prediction['status']} - {prediction.get('error', 'Unknown error')}"
                if prediction["status"] != "processing":
                    return f"Error: Unexpected status {prediction['status']}"

        except requests.exceptions.HTTPError as e:
            return f"Error 422: {e.response.json().get('detail', str(e))}" if e.response.status_code == 422 else f"HTTP Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def pipe(self, body: Dict) -> Union[str, Generator[str, None, None], Iterator[str]]:
        return self.generate_image(get_last_user_message(body["messages"]) or "Error: No prompt provided")

    def pipes(self) -> list:
        return [{"id": "flux_1_1_pro_ultra", "name": "Flux 1.1 Pro Ultra"}]
