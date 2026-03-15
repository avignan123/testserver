#!/usr/bin/env python3
"""Generate missing shots (008, 019, 021, 064, 070, 079, 084) for Tell-Tale Heart using Gemini."""

import os
import time
from pathlib import Path
from google import genai
from google.genai import types
import httpx

# --- Config ---
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_ID = "gemini-2.5-flash-image"
OUTPUT_DIR = Path("generated_v2")
IMAGES_PER_PROMPT = 3
DELAY_BETWEEN_REQUESTS = 3
MAX_RETRIES = 3
RETRY_DELAY = 15
REQUEST_TIMEOUT = 180  # 3 minutes per API call

# --- Reference images ---
REF_NARRATOR = Path("narrator-ref.png")
REF_OLD_MAN = Path("old-man-ref.png")
REF_EYE = Path("eye-ref.png")
REF_BEDROOM = Path("bedroom-ref.png")

NARRATOR = "narrator"
OLD_MAN = "old_man"
EYE = "eye"
BEDROOM = "bedroom"

# --- Missing shots ---
SHOTS = [
    ("008", "A Victorian Gothic oil painting depicting a wide shot from a doorway threshold looking into a Victorian bedroom. The room is fully visible: dark green-grey peeling wallpaper with large torn patches on the walls, pale wooden plank floor, a wooden bed with ball-top posts and a blue-grey coverlet at the far right, a three-legged wooden stool with a single candle burning on it at the centre, a dark wooden chest of drawers against the left wall, a dark-framed painting on the upper left wall, a door frame visible at the right edge. A single candle flame on the stool is the sole light source, casting warm amber-gold light outward across the room in soft radial warmth, touching everything with golden tones, the corners of the room receding into deep warm shadow. A very elderly frail man with a short close-cropped white beard, deeply hooded heavy-lidded eyes, a grey-green drooping nightcap, cream collarless Victorian nightshirt, papery thin aged skin — seated peacefully in a plain wooden chair at the far right, his expression gentle and entirely at ease. Wide shot, 24mm lens, eye level from doorway threshold. Victorian Gothic oil painting with thick impasto brushstrokes, visible canvas texture, Caravaggio-style chiaroscuro, in the style of Rembrandt and Vermeer, raw sienna and warm burnt umber palette. The image must be in a widescreen 16:9 landscape format.",
     [OLD_MAN, BEDROOM]),

    ("019", "A Victorian Gothic oil painting depicting a medium shot in strict profile of a tall gaunt man with a hollow angular face, dark damp hair falling across his forehead, very deep-set cavernous eyes with heavy purple-black rings beneath them, razor-sharp prominent cheekbones, thin dark stubble, wearing a worn dark charcoal wool jacket with lapels, grey open-collar shirt — standing in a dark stone corridor, facing the right side of the frame, one foot raised in mid-step, his body weight shifting forward. The rough dark stone walls of the corridor press close on both sides, narrowing the frame. Far ahead at the right side of the frame: a door, barely ajar, with a single hairline of warm amber-gold light visible at its very base — the only warmth in the entire composition. Cold darkness behind, warm amber trace ahead. 35mm lens, eye level. Victorian Gothic oil painting with thick impasto brushstrokes, Caravaggio-style chiaroscuro, in the style of Rembrandt, Payne's grey with single amber trace at the door base. The image must be in a widescreen 16:9 landscape format.",
     [NARRATOR]),

    ("021", "A Victorian Gothic oil painting depicting an extreme close-up macro image of a dark Victorian tin lantern. The right thumb rests against the tin hinge mechanism of the panel with absolute precision — the panel fractionally open, barely one millimetre, a single hairline crack of amber-gold light visible at the seam. The thumb has not yet pressed. The gaunt pale hand holding the lantern is visible at the edges of the frame. Near-total darkness throughout, the single amber hairline the only light in the entire composition. Macro lens, looking slightly down. Victorian Gothic oil painting with thick impasto brushstrokes, in the style of Rembrandt, near-monochrome darkness with one single amber trace at the seam. The image must be in a widescreen 16:9 landscape format.",
     [NARRATOR]),

    ("064", "A Victorian Gothic oil painting depicting a close-up of a tall gaunt man with a hollow angular face, dark damp hair falling across his forehead, very deep-set cavernous eyes with heavy purple-black rings beneath them, razor-sharp prominent cheekbones, thin dark stubble, wearing a worn dark charcoal wool jacket with lapels, grey open-collar shirt. A lantern positioned below frame provides the sole light source — warm amber light cast dramatically upward across his face, deepening the eye sockets into near-black caverns, sharpening the cheekbones into skeletal relief, painting shadows upward across his brow. His eyes are wide and absolutely fixed, his jaw clenched, the muscles at every corner of his face contracted — the physical state of controlled fury that has crossed a point of no return, the surface broken through. Low angle camera, 85mm lens. Victorian Gothic oil painting with thick impasto brushstrokes, Caravaggio-style chiaroscuro, warm amber underlight against near-black surroundings, in the style of Rembrandt and Lucian Freud. The image must be in a widescreen 16:9 landscape format.",
     [NARRATOR]),

    ("070", "A Victorian Gothic oil painting depicting a medium shot of a tall gaunt man with a hollow angular face, dark damp hair falling across his forehead, very deep-set cavernous eyes with heavy purple-black rings beneath them, razor-sharp prominent cheekbones, thin dark stubble, wearing a worn dark charcoal wool jacket with lapels, grey open-collar shirt — standing upright in a Victorian bedroom, facing slightly toward the viewer. He holds a single lit candle in his right hand, the flame the primary and sole light source, casting warm amber light outward from his hand and illuminating his face from the right side and below. The bedroom around him: peeling dark green-grey wallpaper visible on the walls, the displaced ball-top bed to one side. His expression is completely blank and composed — the face of a man at total rest after a completed action. Eye level camera, 50mm lens. Victorian Gothic oil painting with thick impasto brushstrokes, Caravaggio-style chiaroscuro, in the style of Rembrandt and Lucian Freud, warm amber candlelight in near-dark room. The image must be in a widescreen 16:9 landscape format.",
     [NARRATOR]),

    ("079", "A Victorian Gothic oil painting depicting a bird's-eye macro view looking straight down at Victorian wooden floorboards from directly above, the entire frame filled with the floor surface. The boards are completely undisturbed — perfectly aligned, the joins between them showing nothing unusual, the grain of the pale warm wood clearly visible. Warm amber candlelight illuminates the surface evenly. The floor looks exactly as it always has. There is nothing here. Nothing to see. Bird's-eye macro, looking straight down, fills entire frame. Victorian Gothic oil painting with thick impasto brushstrokes, in the style of Rembrandt, warm amber palette, the ordinariness of the surface is the entire point. The image must be in a widescreen 16:9 landscape format.",
     []),

    ("084", "A Victorian Gothic oil painting depicting a wide shot from inside a dark Victorian hallway looking directly toward the front door which stands open. In the open doorway: three Victorian police officer silhouettes against cold pale blue pre-dawn light from outside — their dark blue Victorian uniforms and helmets barely distinguishable as shapes against the blue exterior light behind them. The interior hallway behind the viewer is warm amber. The exterior beyond the open door is cold pale blue. The three silhouettes stand in the threshold, the contrast between the warm amber interior and the cold blue exterior at its absolute maximum. 35mm lens, eye level from inside, looking toward the open door. Victorian Gothic oil painting with thick impasto brushstrokes, Caravaggio-style chiaroscuro, in the style of Rembrandt, warm amber interior meeting cold blue exterior. The image must be in a widescreen 16:9 landscape format.",
     []),
]


def generate_shots():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
        return

    # Load reference images
    print("Loading reference images...")
    ref_images = {}
    for name, path in [(NARRATOR, REF_NARRATOR), (OLD_MAN, REF_OLD_MAN),
                       (EYE, REF_EYE), (BEDROOM, REF_BEDROOM)]:
        if path.exists():
            ref_images[name] = path.read_bytes()
            print(f"  Loaded {name}: {len(ref_images[name]):,} bytes")
        else:
            print(f"  WARNING: {path} not found")

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
    )

    total = len(SHOTS)
    print(f"\nGenerating {total} missing shots x {IMAGES_PER_PROMPT} images = {total * IMAGES_PER_PROMPT} total images")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_DIR.resolve()}\n")

    succeeded = 0
    failed = 0

    for idx, (shot_num, prompt, ref_types) in enumerate(SHOTS, 1):
        # Skip already generated shots
        existing = list(OUTPUT_DIR.glob(f"shot_{shot_num}_*.png"))
        if len(existing) >= IMAGES_PER_PROMPT:
            print(f"[{idx}/{total}] Shot {shot_num}: SKIPPED (already generated)")
            succeeded += 1
            continue

        print(f"[{idx}/{total}] Shot {shot_num}: {prompt[:70]}...")

        for attempt in range(MAX_RETRIES):
            try:
                # Build contents: reference images + text prompt
                contents = []
                for ref_type in ref_types:
                    if ref_type in ref_images:
                        contents.append(types.Part.from_bytes(
                            data=ref_images[ref_type],
                            mime_type="image/png",
                        ))

                contents.append(prompt)

                # Generate images one at a time
                saved_count = 0
                for img_idx in range(IMAGES_PER_PROMPT):
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                            image_config=types.ImageConfig(
                                aspect_ratio="16:9",
                                image_size="2K",
                            ),
                            safety_settings=[
                                types.SafetySetting(
                                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                    threshold="BLOCK_ONLY_HIGH",
                                ),
                                types.SafetySetting(
                                    category="HARM_CATEGORY_HARASSMENT",
                                    threshold="BLOCK_ONLY_HIGH",
                                ),
                                types.SafetySetting(
                                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                    threshold="BLOCK_ONLY_HIGH",
                                ),
                                types.SafetySetting(
                                    category="HARM_CATEGORY_HATE_SPEECH",
                                    threshold="BLOCK_ONLY_HIGH",
                                ),
                            ],
                        ),
                    )

                    if response.candidates and response.candidates[0].content:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                                saved_count += 1
                                filename = OUTPUT_DIR / f"shot_{shot_num}_{saved_count}.png"
                                filename.write_bytes(part.inline_data.data)
                                print(f"  Saved: {filename}")

                    if img_idx < IMAGES_PER_PROMPT - 1:
                        time.sleep(1)

                if saved_count > 0:
                    succeeded += 1
                else:
                    print(f"  WARNING: No images returned")
                    failed += 1
                break  # Exit retry loop

            except (httpx.TimeoutException, TimeoutError) as e:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  TIMEOUT (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    print(f"  FAILED after {MAX_RETRIES} retries (timeout): {e}")
                    failed += 1
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == MAX_RETRIES - 1:
                        print(f"  FAILED after {MAX_RETRIES} retries: {e}")
                        failed += 1
                elif "timed out" in err_str.lower() or "timeout" in err_str.lower():
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  TIMEOUT (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == MAX_RETRIES - 1:
                        print(f"  FAILED after {MAX_RETRIES} retries (timeout): {e}")
                        failed += 1
                elif "not found" in err_str.lower() or "not supported" in err_str.lower():
                    print(f"  ERROR (model issue): {e}")
                    failed += 1
                    break
                else:
                    print(f"  ERROR: {e}")
                    failed += 1
                    break

        # Delay between shots
        if idx < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nDone! {succeeded} shots succeeded, {failed} failed.")
    print(f"Images saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    generate_shots()
