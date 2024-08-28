from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from diffusers import FluxPipeline
import torch
import os
import uuid

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set Hugging Face API token and cache directory
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
cache_dir = "/workspace/.cache"

# Load model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)
pipe.enable_model_cpu_offload()

@app.get("/")
async def index():
    return FileResponse('index.html')

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get('prompt')

    # Generate image
    image = pipe(prompt, height=1024, width=1024, guidance_scale=3.5, num_inference_steps=50).images[0]

    # Save image with a unique name
    image_name = f"{uuid.uuid4()}.png"
    image_path = os.path.join('static', image_name)
    image.save(image_path)

    return JSONResponse(content={'image_url': f"/static/{image_name}"})

@app.get("/history")
async def history():
    images = os.listdir('static')
    images_list = "".join([f'<img src="/static/{image}" style="max-width:100%;" />' for image in images])
    return HTMLResponse(content=images_list)

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
