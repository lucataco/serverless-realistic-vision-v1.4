# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
from diffusers import StableDiffusionPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_name = os.getenv("MODEL_NAME")
    model = StableDiffusionPipeline.from_pretrained(model_name)
    

if __name__ == "__main__":
    download_model()