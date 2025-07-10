from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import logging

# Optional: suppress diffusers warnings
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Set your prompt
prompt = "a simple floor plan of a 2 bedroom apartment with kitchen and bathroom, blueprint style"

# Load the model pipeline (CPU only)
print("Loading pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
)
pipe = pipe.to("cpu")  # Force CPU mode

# Function to generate image from prompt
def generate_image(prompt_text):
    print("Generating image... please wait.")
    result = pipe(
        prompt_text,
        num_inference_steps=20,
        guidance_scale=7.5
    )
    image = result.images[0]
    return image

# Generate and save image
if __name__ == "__main__":
    img = generate_image(prompt)
    output_path = "generated_blueprint.png"
    img.save(output_path)
    print(f"Image saved to {output_path}")
    img.show()
