#!/usr/bin/env python3
"""Bulk shot generation for Tell-Tale Heart YouTube film using Vertex AI Imagen 3."""

import os
import time
from pathlib import Path
from google import genai
from google.genai import types
from google.auth.credentials import Credentials as BaseCredentials

# --- Auth ---
class AccessTokenCredentials(BaseCredentials):
    def __init__(self, token):
        super().__init__()
        self.token = token
    def refresh(self, request):
        pass
    @property
    def valid(self):
        return True

# --- Config ---
PROJECT_ID = "tth-jam"
LOCATION = "us-central1"
MODEL_GENERATE = "imagen-3.0-generate-002"
MODEL_EDIT = "imagen-3.0-capability-001"
OUTPUT_DIR = Path("generated")
ACCESS_TOKEN = os.environ.get("GOOGLE_ACCESS_TOKEN", "")
IMAGES_PER_PROMPT = 3
DELAY_BETWEEN_REQUESTS = 10
MAX_RETRIES = 3
RETRY_DELAY = 30

# --- Reference image bytes (loaded once) ---
REF_NARRATOR = Path("narrator-ref.png")
REF_OLD_MAN = Path("old-man-ref.png")
REF_EYE = Path("eye-ref.png")
REF_BEDROOM = Path("bedroom-ref.png")

# Reference type constants
NARRATOR = "narrator"
OLD_MAN = "old_man"
EYE = "eye"
BEDROOM = "bedroom"

# --- Shots ---
# Each shot: (shot_number, prompt, [ref_type, ...])
SHOTS = [
    ("001", "WS, 35mm, eye level. Victorian prison exercise yard, high grey stone walls, overcast pale sky. A lone gaunt figure [1] in a dark jacket walks along the far wall, back to camera, pace steady and measured. Stone cobblestone floor, faint moss on lower wall courses. Cold flat grey light. Victorian Gothic oil painting, Payne's grey and raw umber. 16:9 1920x1080.",
     [NARRATOR]),
    ("002", "CU, 85mm, eye level. [1] Gaunt hollow face, dark damp hair across forehead, very deep-set eyes with heavy dark rings, sharp prominent cheekbones, thin dark stubble, dark charcoal jacket, grey open-collar shirt. Eyes looking directly into camera. Expression: controlled, insistent, urgent. Grey stone wall blurred behind. Cold overcast light. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("003", "MCU, 85mm, slight low angle. [1] Gaunt narrator in three-quarter profile facing screen right. Stone wall very close behind his shoulder. Eyes slightly downcast, expression of precise argument. Cold grey light. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("004", "Extreme close-up, macro. [1] The narrator's ear and side of his head in profile, near a rough grey stone wall. Ear centred in frame. Heavy dark hair above. Expression: focused, attentive. Pure cold grey tones. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("005", "WS, 50mm, eye level. [1] Narrator standing still in the Victorian prison yard, head very slightly tilted as if catching a sound. Other prisoners walking in background, blurred. He is the only still figure. Cold grey light. Stone walls. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("006", "CU, 50mm, slightly elevated. [1] Narrator seated on rough wooden bench against stone wall, one hand raised palm open in mid-gesture. Expression: earnest, urgent to be understood. Cold grey light. Prisoners blurred in background. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("007", "MCU, 85mm, eye level. [1] Narrator leaning slightly forward, both hands on knees, eyes direct and calm. A man who has decided to tell the truth. Stone wall behind, cold grey. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("009", "CU, 85mm, high angle 15 degrees. [1] Very elderly man: short white beard, grey-green drooping nightcap, cream collarless Victorian nightshirt. Heavy-lidded eyes, deep wrinkles. Expression: completely peaceful, trusting, gentle. Warm amber candlelight from below-right. Background blurred warm darkness. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("010", "MS, 50mm, eye level. [1] Very elderly man seated in plain wooden chair. Full figure: cream collarless nightshirt, grey nightcap, worn slippers. Hands rest in lap, thin and folded. Candle on stool beside him. He is entirely at peace. Warm amber. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("013", "MCU, 85mm, eye level. [1] Narrator's face showing reaction to something disturbing. Expression: controlled disgust, involuntary narrowing of eyes. Gaze slightly off to the left at something unseen. Cold grey stone wall behind. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("014", "CU, 85mm, slight low angle. [1] Narrator's face in three-quarter profile, screen right. A private decision arriving, resolution in the eyes, corners of mouth completely neutral. Cold grey wall behind. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("015", "MCU, 85mm, low angle. [1] Narrator seated, back straight, hands placed deliberately on his knees. Posture: composed, self-satisfied, the posture of a man describing an achievement. Cold grey light. Prisoners blurred behind. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("016", "Extreme close-up, macro. [1] The narrator's gaunt pale hands, fingers fully interlaced and clasped. Dark charcoal jacket sleeves at the wrists. The grip: deliberate, controlled, ceremonial. Every finger precisely placed. Cold flat grey light. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("018", "Extreme close-up, macro. A worn iron door latch. [1] Gaunt pale fingers closed around it with extreme lightness. Near-total darkness. The faintest amber thread of light at the base of the door. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("020", "Extreme close-up, macro. Both of [1] the narrator's gaunt hands holding a completely closed dark Victorian tin lantern against pure black background. No light escaping from any seam. Hands hold it with surgical care. Victorian Gothic oil painting, near-total darkness. 16:9 1920x1080.",
     [NARRATOR]),
    ("022", "MS, 35mm, low angle. From inside a dark stone corridor looking toward an open bedroom doorway. [1] Narrator as near-silhouette against the warm amber rectangle of the room beyond. Dark jacket, gaunt silhouette, face entirely in shadow, only the sharp edge of cheekbone caught by amber backlight. Completely still. Deep Caravaggio contrast. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("023", "Extreme close-up, slightly elevated. An old brass mantel clock face, hands positioned precisely at midnight. The brass numerals and hands catch warm amber candlelight. The clock face is cracked and aged. Deep warm darkness behind. Victorian Gothic oil painting. 16:9 1920x1080.",
     []),
    ("025", "MCU, 85mm, low angle. [1] Narrator in profile, screen right. His jaw is set. Eyes focused on something internal. Expression: controlled frustration, a plan obstructed. Cold grey prison wall. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("026", "CU, 85mm, slight low angle, direct address. [1] Narrator's face to camera. Expression: making a precise distinction, the controlled precision of a man explaining something carefully considered. Cold grey. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("028", "MS, 50mm, low angle. [1] Narrator mid-rise from a rough wooden bench, both hands pressing down on bench surface, body lifting upward. As he reaches full height his eyes rise from the floor and focus on the middle distance. Cold grey light. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("030", "Extreme close-up, macro, floor level. A dark shoe, one foot slightly raised from a wooden floorboard, the sole barely clearing the surface. The floorboard beneath is pale warm wood. Warm amber from ahead. Victorian Gothic oil painting. 16:9 1920x1080.",
     []),
    ("032", "CU, 85mm, low angle. [1] Narrator's face showing the expression of private triumph. Eyes very slightly wider, a fractional tightening of the facial muscles. Not smiling, something more dangerous than a smile. Cold grey. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("034", "CU, 85mm, high angle 20 degrees. [1] Old man's face in profile on pillow, eyes fully closed. Short white beard, grey nightcap. Completely peaceful. Warm amber candlelight. The eyelid centred in frame. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("035", "CU, 50mm, from inside the room at low angle. [1] The narrator's head visible through a partly open door, gaunt face in profile, dark hair. Both hands visible below holding a closed lantern. He is leaning through the doorway. Near-darkness. Faint amber trace. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("038", "MS, 35mm, from inside bedroom, floor-level low angle. [1] The narrator stands in the doorway, one foot across the threshold, completely frozen. Near-silhouette against the slightly less dark corridor. Right hand concealed behind back. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("040", "CU, 85mm, profile. [1] Old man sitting completely rigid in bed. Profile facing screen left. Eyes open and fixed, not looking at anything, listening. His neck is taut. His hands clenched on the coverlet. Warm amber trace from the right side only. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("041", "Extreme close-up, macro, profile. [1] An elderly man's ear and the side of his head. The ear turned toward something off-frame. Jaw slightly open, held breath. Very aged, deeply wrinkled skin. A trace of amber candlelight on the outer ear cartilage. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("043", "MCU, 85mm, eye level. [1] Old man in bed, one hand pressed flat against his own forehead. His eyes are closed. He is trying to reason himself into calm. Expression: the effort of self-persuasion. Warm amber trace light on the side of his face. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("044", "CU, 85mm, high angle. [1] Old man's face, eyes reopened, the calm attempt has failed. His eyes are wider than before. The fear is back and deeper. Expression: a man who knows something is in the room with him. Warm amber trace. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("046", "CU, 85mm. [1] Old man's face with a large shadow falling diagonally across it, one half warm amber, one half sudden near-black. His expression in the lit half: the realisation arriving. Victorian Gothic oil painting, extreme contrast. 16:9 1920x1080.",
     [OLD_MAN]),
    ("047", "MS, 85mm, profile. [1] Old man sitting upright in bed, both hands pressed flat against his chest over his heart. Profile facing screen left. His breathing is very shallow. Expression: paralysed, the body knowing what the mind will not accept. Warm amber trace. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("048", "MCU, 85mm, high angle. [1] Old man's face, mouth slightly parted, not a scream, barely a breath. An involuntary sound escaping. His eyes are fixed. The expression of a man who knows death is in the room. Warm amber candlelight from below. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("050", "MCU, 85mm, low angle. [1] Narrator's face showing the expression of recognition. He has made this sound before. He knows exactly what it is. Detached precision of a man identifying a known phenomenon. Amber underlight from the lantern below. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("051", "MS, 85mm, eye level. [1] Narrator seated in prison, hands clasped in his lap, head slightly bowed, the posture of memory. Cold grey light. He is entirely inside the memory. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("053", "CU, 85mm, direct address. [1] Narrator's face to camera. The controlled precision has fractionally loosened, something personal in the expression. He is admitting something about himself. Cold grey. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("058", "CU, 85mm, eye level, profile facing screen left. [1] Old man: one hand pressed flat against his chest over his heart. In his neck a visible pulse. His breathing: tiny shallow sips. Expression: paralysed, knowing. Deep warm darkness behind. Victorian Gothic oil painting. 16:9 1920x1080.",
     [OLD_MAN]),
    ("060", "WS, 35mm, from corner. [1] Narrator standing near the centre of the bedroom, lantern in one hand, the beam closed. Completely still. Old man upright in bed at the right. The contrast of the narrator's stillness against the old man's terror. Warm amber. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("061", "MCU, 85mm, slight low angle. [1] Narrator's face, head tilted very slightly, listening. His expression: the question arriving. Cold grey from behind, trace of amber from below. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("063", "Extreme close-up, macro, bird's-eye view. Dark shoes on pale wooden floorboards. One foot pressed firmly down, the other mid-lift. Warm amber candlelight illuminating the grain of the wood beneath. Victorian Gothic oil painting. 16:9 1920x1080.",
     []),
    ("065", "MS, 35mm, from behind and low. [1] The narrator as dark silhouette, a wooden chair raised above his head with both hands, arms fully extended. Against warm amber candlelight behind him. The bedroom around him, bed visible at right. Victorian Gothic oil painting, strong silhouette. 16:9 1920x1080.",
     [NARRATOR]),
    ("068", "MS, 35mm, from behind and slightly elevated. [1] Narrator standing, back to camera, looking down at the floor beside the bed. The bed is empty, coverlet pulled. His posture: completed action. Warm amber. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("071", "Extreme close-up, macro. [1] The narrator's ear and the side of his head, angled downward as if pressed toward the floor. His eye barely visible above the ear, watching sideways. He is listening to something below. The floorboards at the very bottom of the frame. Warm amber. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("072", "Extreme close-up, macro. [1] The narrator's gaunt pale hand pressed flat on a surface, fingers spread, palm down, the hand completely still. Warm amber candlelight from the side. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("073", "CU, 85mm, high angle. [1] Narrator's face looking downward from slightly above. His expression: clinical, confirming, not triumphant. A professional confirming a result. His eyes looking at something below the frame. Amber from the side. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("075", "MCU, 85mm, direct address. [1] Narrator's face to camera. Expression: calm certainty, the challenge issued. He is about to prove something. His eyes are direct and unwavering. Cold grey. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("076", "Extreme close-up, macro, from slightly above. [1] The narrator's gaunt hands gripping the edges of a floorboard, fingers finding the gap, board beginning to lift. The darkness of the space below just barely visible. Warm amber candlelight. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("078", "Extreme close-up, macro, bird's-eye view. [1] The narrator's hands pressing the last floorboard flat, palms down, applying firm even pressure. The board is flush with its neighbours. The floor looks undisturbed. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("080", "CU, 85mm. [1] The narrator's hands held up, palms toward camera, examined, clean. The candlelight behind them making the thin skin glow slightly amber. No stain. His expression: satisfaction in the evidence. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("082", "CU, macro. An old brass mantel clock, hands now at four o'clock. The room around it near-dark, candle burning very low. Through a high window in the background the faintest pre-dawn blue-grey light beginning. Victorian Gothic oil painting. 16:9 1920x1080.",
     []),
    ("083", "WS, 35mm, from inside a dark corridor looking toward the front door. The door heavy dark wood, closed. A fist knock visible, the door vibrating very slightly, dust falling from the knocker impact. Pure darkness inside, faintest pre-dawn blue-grey at the door seams. Victorian Gothic oil painting. 16:9 1920x1080.",
     []),
    ("085", "MS, 50mm, from the side. [1] Narrator at the open front door, dark jacket, gaunt face, gesturing the officers inside with one arm extended. His expression: welcoming, unhurried, the perfect innocent householder. Three officer silhouettes entering from the right. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("087", "WS, 24mm, from corner of bedroom. [1] Narrator standing, gesturing welcomingly. Three blurred officers near the doorway, dark uniforms. Narrator sharply in focus, officers slightly soft. Warm amber candlelight. Undisturbed floorboards. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("088", "WS, 35mm, from inside bedroom looking toward the door. [1] Narrator standing in the doorway, arm extended, presenting the room. The bedroom completely ordinary, bed made, everything in place. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("089", "MS, 50mm. [1] Narrator pulling a wooden chair across the floor, the gesture of a helpful host. His expression: helpful, engaged, entirely at ease. One blurred officer seated in background right. Warm amber. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("090", "MCU, 85mm, slight low angle. [1] Narrator seated, both hands on armrests, expression of deliberate ease and controlled confidence. A man completely at home with the police sitting in his room. Officers barely visible behind. Low amber. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("091", "Extreme close-up, macro, floor level. The narrator's dark shoes on the pale floorboards, planted firmly and deliberately. The boards look completely ordinary. Victorian Gothic oil painting. 16:9 1920x1080.",
     []),
    ("092", "MS, 35mm, from the side. [1] Three officers seated, narrator standing among them, his body language relaxed and conversational. One officer laughing slightly. The narrator performing ease perfectly. Warm amber bedroom. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("094", "CU, 85mm. [1] Narrator's face, still talking, but the expression shows effort now. His jaw is slightly too tight. His eyes are slightly too focused. Officers blurred behind. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("096", "Extreme close-up, macro. [1] The narrator's hand gripping the wooden armrest with enormous force, knuckles white, tendons visible on the back of the hand. Dark jacket sleeve. Warm amber candlelight. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("097", "MCU, 50mm, low angle, camera tilted 15 degrees clockwise Dutch tilt. [1] Narrator forward in chair, dark jacket, dark hair. Expression: the mask failing, something breaking through. Officers blurred behind. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("098", "CU, 85mm. [1] Narrator's face, the expression of a man talking too fast to outrun something. His eyes too bright. The performance has become desperation. Very slight perspiration on the brow. Amber underlight. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("100", "MCU, 50mm, low angle, camera tilted 20 degrees clockwise Dutch tilt, most extreme in film. [1] Narrator standing, slightly forward, the room visibly wrong around him. Officers barely visible as shapes at the edges. His face: breaking. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
    ("101", "Extreme close-up, macro. The narrator's eye, dark grey-green iris, pupil massively dilated nearly consuming the iris. Red-veined whites. Heavy dark circles above and below. Victorian Gothic oil painting, thick textured brushstrokes. 16:9 1920x1080.",
     [EYE]),
    ("103", "MCU, 85mm. [1] Narrator's face showing the confession in his expression. Something released, not relief but something more like collapse. His eyes still wide, his jaw working. The moment of total exposure. Victorian Gothic oil painting. 16:9 1920x1080.",
     [NARRATOR]),
]

# Pre-load reference image bytes
REF_BYTES = {}

def load_refs():
    for name, path in [(NARRATOR, REF_NARRATOR), (OLD_MAN, REF_OLD_MAN),
                       (EYE, REF_EYE), (BEDROOM, REF_BEDROOM)]:
        if path.exists():
            REF_BYTES[name] = path.read_bytes()
            print(f"  Loaded {name}: {len(REF_BYTES[name]):,} bytes")
        else:
            print(f"  WARNING: {path} not found")


def build_reference_images(ref_types):
    """Build reference image list for edit_image API."""
    refs = []
    for ref_type in ref_types:
        if ref_type not in REF_BYTES:
            continue

        ref_image = types.Image(image_bytes=REF_BYTES[ref_type])

        if ref_type == EYE:
            refs.append(types.StyleReferenceImage(
                reference_image=ref_image,
                reference_id=len(refs) + 1,
                config=types.StyleReferenceConfig(
                    style_description="Extreme close-up eye, Victorian Gothic oil painting with thick textured brushstrokes"
                ),
            ))
        elif ref_type in (NARRATOR, OLD_MAN):
            subject_desc = {
                NARRATOR: "Gaunt man with dark hair, deep-set eyes, dark charcoal jacket, grey shirt",
                OLD_MAN: "Very elderly man with short white beard, grey-green nightcap, cream nightshirt",
            }[ref_type]
            refs.append(types.SubjectReferenceImage(
                reference_image=ref_image,
                reference_id=len(refs) + 1,
                config=types.SubjectReferenceConfig(
                    subject_type="SUBJECT_TYPE_PERSON",
                    subject_description=subject_desc,
                ),
            ))
    return refs


def generate_shots():
    """Generate all shots."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not ACCESS_TOKEN:
        print("ERROR: Set GOOGLE_ACCESS_TOKEN environment variable.")
        print("  Run: export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)")
        return

    print("Loading reference images...")
    load_refs()

    creds = AccessTokenCredentials(ACCESS_TOKEN)
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=creds,
    )

    total = len(SHOTS)
    print(f"\nGenerating {total} shots x {IMAGES_PER_PROMPT} images = {total * IMAGES_PER_PROMPT} total images")
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
                ref_images = build_reference_images(ref_types)

                if ref_images:
                    # Use edit_image with capability model for reference images
                    response = client.models.edit_image(
                        model=MODEL_EDIT,
                        prompt=prompt,
                        reference_images=ref_images,
                        config=types.EditImageConfig(
                            number_of_images=IMAGES_PER_PROMPT,
                            aspect_ratio="16:9",
                            safety_filter_level="BLOCK_ONLY_HIGH",
                            person_generation="ALLOW_ALL",
                        ),
                    )
                else:
                    # Use generate_images for shots without reference images
                    response = client.models.generate_images(
                        model=MODEL_GENERATE,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=IMAGES_PER_PROMPT,
                            aspect_ratio="16:9",
                            safety_filter_level="BLOCK_ONLY_HIGH",
                            person_generation="ALLOW_ALL",
                        ),
                    )

                if response.generated_images:
                    for j, img in enumerate(response.generated_images, 1):
                        filename = OUTPUT_DIR / f"shot_{shot_num}_{j}.png"
                        img.image.save(str(filename))
                        print(f"  Saved: {filename}")
                    succeeded += 1
                else:
                    print(f"  WARNING: No images returned (may have been filtered)")
                    failed += 1
                break  # Success or filtered, exit retry loop

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == MAX_RETRIES - 1:
                        print(f"  FAILED after {MAX_RETRIES} retries")
                        failed += 1
                else:
                    print(f"  ERROR: {e}")
                    failed += 1
                    break  # Non-retryable error

        # Delay between requests
        if idx < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nDone! {succeeded} shots succeeded, {failed} failed.")
    print(f"Images saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    generate_shots()
