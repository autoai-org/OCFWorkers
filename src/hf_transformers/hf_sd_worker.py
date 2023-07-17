from diffusers import DiffusionPipeline
import torch
from io import BytesIO
from PIL import Image
import base64
import argparse
from loguru import logger

from _base import InferenceWorker

class HFSDWorker(InferenceWorker):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
        self.model = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.model.to("cuda")
        self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
        self.format = "JPEG"
        # warmup
        self.model(prompt="test")
    async def handle_requests(self, msg):
        prompt = msg.get('prompt', '')
        images = self.model(prompt=prompt).images
        results = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format=self.format)
            img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
            results.append(img_str)
        output = {
            'output': {
                'images': results,
            },
            'params': {
                'prompt': prompt,
            }
        }
        return output

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="stabilityai/stable-diffusion-xl-base-0.9")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    worker = HFSDWorker(args.model_name)
    worker.start()