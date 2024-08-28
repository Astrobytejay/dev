import os
import torch
from diffusers import FluxPipeline

# Set Hugging Face API token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Specify the cache directory
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16, 
    cache_dir="/workspace"
)
pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, height=1024, width=1024, guidance_scale=3.5, num_inference_steps=50).images[0]
image.save("/workspace/flux-dev.png")
