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

@app.get("/edit")
async def edit_image(request: Request):
    image_url = request.query_params.get('image_url')
    if not image_url:
        return HTMLResponse(content="Image URL not provided", status_code=400)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Edit Image</title>
        <style>
            body {{
                background-color: #000004;
                color: #fff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                overflow: hidden;
            }}
            #image-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
                height: 80vh;
                overflow: auto;
            }}
            img {{
                max-width: 90%;
                max-height: 80vh;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .controls {{
                display: flex;
                justify-content: center;
                gap: 10px;
            }}
            .controls button,
            .controls select {{
                padding: 10px;
                background-color: #e94560;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}
            .controls button:hover,
            .controls select:hover {{
                background-color: #ff6b81;
            }}
        </style>
    </head>
    <body>
        <div id="image-container">
            <canvas id="canvas"></canvas>
        </div>
        <div class="controls">
            <button onclick="startCrop()">Start Crop</button>
            <button onclick="applyCrop()">Apply Crop</button>
            <select id="resize-options">
                <option value="1:1">1:1</option>
                <option value="4:6">4 x 6 inches</option>
                <option value="5:7">5 x 7 inches</option>
                <option value="8:10">8 x 10 inches</option>
                <option value="8.5:11">8.5 x 11 inches</option>
                <option value="12:18">12 x 18 inches</option>
                <option value="18:24">18 x 24 inches</option>
                <option value="24:36">24 x 36 inches</option>
                <option value="freehand">Freehand</option>
            </select>
            <button onclick="applyResize()">Apply Resize</button>
            <select id="filter-options">
                <option value="none">None</option>
                <option value="grayscale">Grayscale</option>
                <option value="sepia">Sepia</option>
                <option value="invert">Invert</option>
                <option value="blur">Blur</option>
                <option value="brightness">Brightness</option>
                <option value="contrast">Contrast</option>
                <option value="hue-rotate">Hue Rotate</option>
                <option value="saturate">Saturate</option>
                <option value="sharpness">Sharpness</option>
                <option value="vintage">Vintage</option>
                <option value="polaroid">Polaroid</option>
                <option value="kodachrome">Kodachrome</option>
                <option value="technicolor">Technicolor</option>
                <option value="duotone">Duotone</option>
            </select>
            <button onclick="applyFilter()">Apply Filter</button>
            <button onclick="saveImage()">Save</button>
        </div>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            let cropping = false;
            let cropStartX, cropStartY, cropWidth, cropHeight;

            // Load the image onto the canvas
            img.src = new URLSearchParams(window.location.search).get('image_url');
            img.onload = () => {{
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            }};

            // Start Crop
            function startCrop() {{
                cropping = true;
                canvas.addEventListener('mousedown', startSelection);
                canvas.addEventListener('mouseup', endSelection);
            }}

            function startSelection(e) {{
                if (!cropping) return;
                const rect = canvas.getBoundingClientRect();
                cropStartX = e.clientX - rect.left;
                cropStartY = e.clientY - rect.top;
                canvas.addEventListener('mousemove', trackSelection);
            }}

            function trackSelection(e) {{
                if (!cropping) return;
                const rect = canvas.getBoundingClientRect();
                cropWidth = (e.clientX - rect.left) - cropStartX;
                cropHeight = (e.clientY - rect.top) - cropStartY;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                ctx.strokeStyle = 'red';
                ctx.strokeRect(cropStartX, cropStartY, cropWidth, cropHeight);
            }}

            function endSelection() {{
                if (!cropping) return;
                canvas.removeEventListener('mousemove', trackSelection);
                cropping = false;
            }}

            // Apply Crop
            function applyCrop() {{
                if (cropWidth && cropHeight) {{
                    const imageData = ctx.getImageData(cropStartX, cropStartY, cropWidth, cropHeight);
                    canvas.width = cropWidth;
                    canvas.height = cropHeight;
                    ctx.putImageData(imageData, 0, 0);
                }}
            }}

            // Apply Resize
            function applyResize() {{
                const resizeOption = document.getElementById('resize-options').value;
                let resizeWidth, resizeHeight;

                switch (resizeOption) {{
                    case '1:1':
                        resizeWidth = resizeHeight = Math.min(canvas.width, canvas.height);
                        break;
                    case '4:6':
                        resizeWidth = 600;
                        resizeHeight = 400;
                        break;
                    case '5:7':
                        resizeWidth = 700;
                        resizeHeight = 500;
                        break;
                    case '8:10':
                        resizeWidth = 1000;
                        resizeHeight = 800;
                        break;
                    case '8.5:11':
                        resizeWidth = 1100;
                        resizeHeight = 850;
                        break;
                    case '12:18':
                        resizeWidth = 1800;
                        resizeHeight = 1200;
                        break;
                    case '18:24':
                        resizeWidth = 2400;
                        resizeHeight = 1800;
                        break;
                    case '24:36':
                        resizeWidth = 3600;
                        resizeHeight = 2400;
                        break;
                    case 'freehand':
                        resizeWidth = prompt("Enter new width:");
                        resizeHeight = prompt("Enter new height:");
                        break;
                }}

                if (resizeWidth && resizeHeight) {{
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    canvas.width = resizeWidth;
                    canvas.height = resizeHeight;
                    ctx.putImageData(imageData, 0, 0);
                    ctx.drawImage(img, 0, 0, resizeWidth, resizeHeight);
                }}
            }}

            // Apply Filter
            function applyFilter() {{
                const filterOption = document.getElementById('filter-options').value;
                ctx.filter = getFilter(filterOption);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            }}

            function getFilter(filter) {{
                switch (filter) {{
                    case 'grayscale':
                        return 'grayscale(100%)';
                    case 'sepia':
                        return 'sepia(100%)';
                    case 'invert':
                        return 'invert(100%)';
                    case 'blur':
                        return 'blur(5px)';
                    case 'brightness':
                        return 'brightness(150%)';
                    case 'contrast':
                        return 'contrast(200%)';
                    case 'hue-rotate':
                        return 'hue-rotate(90deg)';
                    case 'saturate':
                        return 'saturate(200%)';
                    case 'sharpness':
                        return 'contrast(200%) brightness(110%)';
                    case 'vintage':
                        return 'sepia(100%) contrast(120%) brightness(90%)';
                    case 'polaroid':
                        return 'contrast(150%) saturate(120%) brightness(110%)';
                    case 'kodachrome':
                        return 'contrast(120%) saturate(110%)';
                    case 'technicolor':
                        return 'contrast(130%) saturate(110%) brightness(120%) hue-rotate(50deg)';
                    case 'duotone':
                        return 'sepia(90%) hue-rotate(180deg)';
                    default:
                        return 'none';
                }}
            }}

            // Save Image
            function saveImage() {{
                const link = document.createElement('a');
                link.href = canvas.toDataURL();
                link.download = 'edited-image.png';
                link.click();
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/history")
async def history():
    images = os.listdir('static')
    images_list = "".join([f'<img src="/static/{image}" style="max-width:100%;" />' for image in images])
    return HTMLResponse(content=images_list)

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
