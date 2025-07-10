from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import logging
import os

# Optional: suppress diffusers warnings
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Flask setup
app = Flask(__name__)

# Load pipeline
print("Loading pipeline... (This may take a minute)")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
)
pipe = pipe.to("cpu")  # CPU mode

# Image generation function
def generate_image(prompt_text, output_path="static/generated1.png"):
    print(f"Generating image for: {prompt_text}")
    result = pipe(prompt_text, num_inference_steps=20, guidance_scale=7.5)
    image = result.images[0]
    image.save(output_path)
    return output_path

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    generated_image = None
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            image_path = generate_image(prompt)
            generated_image = image_path
    return render_template("index.html", image=generated_image)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5001)

