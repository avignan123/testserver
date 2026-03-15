#!/usr/bin/env python3
"""Bulk image generation using Google Vertex AI Imagen 3 via the google-genai library."""

import os
import time
from pathlib import Path
from google import genai
from google.genai import types
from google.auth.credentials import Credentials as BaseCredentials


class AccessTokenCredentials(BaseCredentials):
    """Simple credentials wrapper for a raw access token."""

    def __init__(self, token):
        super().__init__()
        self.token = token

    def refresh(self, request):
        pass

    @property
    def valid(self):
        return True


# Configuration
PROJECT_ID = "tth-jam"
LOCATION = "us-central1"
MODEL_ID = "imagen-3.0-generate-002"
OUTPUT_DIR = Path("generated_images")
ACCESS_TOKEN = os.environ.get("GOOGLE_ACCESS_TOKEN", "")

# Define your prompts here - add/remove as needed
PROMPTS = [
    "A serene mountain landscape at sunset with golden light reflecting off a lake",
    "A futuristic city skyline with flying vehicles and neon lights at night",
    "A cozy coffee shop interior with warm lighting and bookshelves",
    "An underwater coral reef teeming with colorful tropical fish",
    "A medieval castle on a cliff overlooking a misty valley at dawn",
]


def generate_images():
    """Generate images for all prompts using Vertex AI Imagen 3."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not ACCESS_TOKEN:
        print("ERROR: Set GOOGLE_ACCESS_TOKEN environment variable.")
        print("  Run: export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)")
        return

    creds = AccessTokenCredentials(ACCESS_TOKEN)
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=creds,
    )

    print(f"Generating {len(PROMPTS)} images with {MODEL_ID}...")
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] Generating: {prompt[:80]}...")

        try:
            response = client.models.generate_images(
                model=MODEL_ID,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="16:9",
                    safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                ),
            )

            if response.generated_images:
                image = response.generated_images[0]
                filename = OUTPUT_DIR / f"image_{i:03d}.png"
                image.image.save(str(filename))
                print(f"  Saved: {filename}")
            else:
                print(f"  WARNING: No image returned (may have been filtered)")

        except Exception as e:
            print(f"  ERROR: {e}")

        # Small delay between requests to avoid rate limiting
        if i < len(PROMPTS):
            time.sleep(2)

    print(f"\nDone! Images saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    generate_images()
