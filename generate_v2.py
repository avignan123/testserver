#!/usr/bin/env python3
"""Bulk shot generation for Tell-Tale Heart using Gemini image generation."""

import os
import time
import base64
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
PROJECT_ID = "tth-jam-1"
LOCATION = "us-central1"
ACCESS_TOKEN = os.environ.get("GOOGLE_ACCESS_TOKEN", "")
MODEL_ID = "gemini-2.0-flash-preview-image-generation"
OUTPUT_DIR = Path("generated_v2")
IMAGES_PER_PROMPT = 3
DELAY_BETWEEN_REQUESTS = 3
MAX_RETRIES = 3
RETRY_DELAY = 15

# --- Reference images ---
REF_NARRATOR = Path("narrator-ref.png")
REF_OLD_MAN = Path("old-man-ref.png")
REF_EYE = Path("eye-ref.png")
REF_BEDROOM = Path("bedroom-ref.png")

NARRATOR = "narrator"
OLD_MAN = "old_man"
EYE = "eye"
BEDROOM = "bedroom"

# --- All shots ---
SHOTS = [
    # NARRATOR SHOTS
    ("001", "A lone gaunt man in a dark charcoal jacket walks slowly along a high grey stone Victorian prison wall, back to camera. Stone cobblestone floor. Cold flat overcast grey light. Two blurred prisoner silhouettes far in background. Wide shot, 35mm perspective. Victorian Gothic oil painting style, visible expressive brushstrokes, Payne's grey and raw umber palette, Caravaggio chiaroscuro lighting. 16:9 landscape format.", [NARRATOR]),
    ("002", "The man from the reference image looks directly into the camera. Extreme close-up, 85mm portrait. Gaunt hollow face, dark damp hair falling across forehead, very deep-set eyes with heavy dark rings, sharp prominent cheekbones, thin dark stubble, dark charcoal jacket, grey open-collar shirt. Expression: controlled, insistent, urgent. Grey stone wall blurred far behind. Cold overcast grey light. Victorian Gothic oil painting, Payne's grey palette. 16:9 landscape.", [NARRATOR]),
    ("003", "The man from the reference image in three-quarter profile facing right. Medium close-up, 85mm, slight low angle. Stone wall very close behind his shoulder. Eyes slightly downcast, expression of precise argument. Cold flat grey light. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("004", "The man from the reference image — only his ear and the side of his head visible in profile, pressed near a rough grey stone wall. Extreme close-up, macro lens. Heavy dark hair above. Expression: focused, attentive. Pure cold grey tones. Victorian Gothic oil painting, thick textured brushstrokes. 16:9 landscape.", [NARRATOR]),
    ("005", "The man from the reference image standing still in a Victorian prison exercise yard, head very slightly tilted as if catching a distant sound. Wide shot. Other dark-clad prisoners walk past in soft blur. He is the only still figure in the frame. Cold grey light. Stone walls. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("006", "The man from the reference image seated on a rough wooden bench against a stone wall, one hand raised with palm open in mid-gesture. Close-up. Expression: earnest, urgent to be understood. Cold grey light from above. Prisoners blurred in background corridor. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("007", "The man from the reference image leaning slightly forward toward camera, both hands on his knees, eyes direct and calm. Medium close-up, eye level. Stone wall behind, cold grey. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("013", "The man from the reference image, face showing controlled disgust — an involuntary narrowing of the eyes, gaze directed slightly left at something unseen off-camera. Medium close-up, eye level. Cold grey stone wall behind. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("014", "The man from the reference image in three-quarter profile facing right. Close-up, slight low angle. A private decision visibly arriving — resolution in the eyes, corners of mouth completely neutral. Cold grey wall behind. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("015", "The man from the reference image seated on a bench, back straight, hands placed deliberately on his knees. Medium close-up, low angle. Posture: composed, self-satisfied. Cold grey light. Prisoners blurred behind. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("016", "Only the man's gaunt pale hands visible — fingers fully interlaced and clasped. Extreme close-up, macro. Dark charcoal jacket sleeves at the wrists. The grip: deliberate, controlled, ceremonial. Cold flat grey light. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("018", "Only the man's gaunt fingers visible, closed around a worn iron door latch with extreme lightness. Extreme close-up, macro. Near-total darkness. The faintest amber thread of light at the base of the door below. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("020", "Only the man's gaunt hands holding a completely closed dark Victorian tin lantern against pure black background. Extreme close-up, macro. No light escaping from any seam. Held with surgical care. Victorian Gothic oil painting, near-total darkness. 16:9 landscape.", [NARRATOR]),
    ("022", "The man from the reference image as a near-silhouette standing in a dark stone corridor doorway, backlit by warm amber candlelight from the room beyond. Medium shot, 35mm, low angle. Only the sharp edge of his cheekbone is caught by amber backlight. Deep Caravaggio contrast, near-black foreground, warm amber rectangle behind. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("025", "The man from the reference image in profile facing right. Medium close-up, low angle. Jaw set, eyes focused on something internal. Expression: controlled frustration, a plan obstructed. Cold grey prison wall behind. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("026", "The man from the reference image looking directly into camera. Close-up, slight low angle. Expression: making a precise distinction, the controlled precision of a man explaining something carefully. Cold grey. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("028", "The man from the reference image mid-rise from a rough wooden bench, both hands pressing down on bench surface, body lifting upward. Medium shot, low angle. As he reaches full height his eyes rise from the floor and focus on the middle distance. Cold grey light. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("030", "Only the man's dark shoe visible — one foot slightly raised from a pale wooden floorboard, the sole barely clearing the surface. Extreme close-up, macro, floor level. Warm amber light coming from ahead. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("032", "The man from the reference image. Close-up, low angle. Expression: private triumph — eyes very slightly wider, a fractional tightening of the facial muscles. Not smiling, something more dangerous. Cold grey. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("035", "The man from the reference image — only his head visible through a partly open door, gaunt face in profile, dark hair. Close-up from inside the room at low angle. Both hands visible below holding a closed dark lantern. Leaning through the doorway. Near-darkness, faint amber trace from inside the room. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("038", "The man from the reference image standing in a bedroom doorway, one foot across the threshold, completely frozen. Medium shot, 35mm, floor-level low angle from inside the room. Near-silhouette against the slightly less dark corridor behind him. Right hand concealed behind back. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("050", "The man from the reference image, expression of detached recognition — he has heard this sound before and knows exactly what it is. Medium close-up, low angle. Amber underlight from a lantern below casts shadows upward across his face. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("051", "The man from the reference image seated, hands clasped in lap, head slightly bowed in the posture of memory. Medium shot, eye level. Cold grey prison light. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("053", "The man from the reference image looking directly into camera. Close-up, direct address. The controlled precision has fractionally loosened — something personal in the expression, an admission. Cold grey. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("060", "The man from the reference image standing still in a Victorian bedroom, a closed dark lantern in one hand. Wide shot from the corner. Completely still. Warm amber candlelight. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("061", "The man from the reference image, head tilted very slightly, listening. Medium close-up, slight low angle. Cold grey from behind, trace of amber from below. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("063", "Only the man's dark shoes on pale Victorian wooden floorboards — one foot pressed firmly down, the other mid-lift. Extreme close-up, macro, bird's-eye view. Warm amber candlelight illuminating the grain of the wood. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("065", "The man from the reference image as a dark silhouette with a wooden chair raised above his head, both arms extended. Medium shot, 35mm, from behind and low. Against warm amber candlelight behind him. Victorian bedroom around him, bed visible at right. Strong silhouette. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("068", "The man from the reference image standing with his back to camera, looking down at the floor beside an empty dishevelled bed. Medium shot from behind and slightly elevated. His posture: completed action, the stillness of completion. Warm amber. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("071", "Only the man's ear and side of his head visible — but now angled downward as if pressed toward the floor, listening to something below. Extreme close-up, macro. His eye barely visible above the ear, watching sideways. The floorboards at the very bottom of the frame. Warm amber. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("072", "Only the man's gaunt pale hand pressed flat on a surface, fingers spread, palm down, completely still. Extreme close-up, macro. Warm amber candlelight from the side. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("073", "The man from the reference image looking downward, face seen from slightly above. Close-up, high angle. Expression: clinical, confirming, not triumphant. His eyes looking at something below the frame. Amber light from the side. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("075", "The man from the reference image looking directly into camera. Medium close-up, direct address. Expression: calm certainty, the challenge issued. His eyes direct and unwavering. Cold grey. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("076", "Only the man's gaunt hands gripping the edges of a floorboard from above, fingers in the gap, the board beginning to lift. Extreme close-up, macro, from slightly above. The darkness of the space below just barely visible. Warm amber candlelight. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("078", "Only the man's hands pressing the last floorboard flat — palms down, applying firm even pressure. Extreme close-up, macro, bird's-eye view. The board flush with its neighbours. The floor looks undisturbed. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("080", "The man's hands held up, palms facing camera, clean — examined. Close-up. The candlelight behind them makes the thin skin glow slightly amber. Expression: satisfaction. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("085", "The man from the reference image at an open front door, gesturing three police officer silhouettes inside with one arm extended. Medium shot from the side. Expression: welcoming, unhurried, the perfect innocent householder. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("087", "The man from the reference image standing in a Victorian bedroom, gesturing welcomingly toward three blurred dark-uniformed officers near the doorway. Wide shot, 24mm. Narrator sharply in focus, officers slightly soft. Warm amber candlelight. Undisturbed pale wooden floorboards. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("088", "The man from the reference image standing in a bedroom doorway, arm extended presenting the room. Wide shot from inside looking toward the door. The bedroom completely ordinary — bed made, everything in place. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("089", "The man from the reference image pulling a wooden chair across a bedroom floor, the gesture of a helpful host. Medium shot. Expression: helpful, engaged, entirely at ease. One blurred dark-uniformed officer seated in background right. Warm amber. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("090", "The man from the reference image seated with both hands on wooden armrests, expression of deliberate ease and controlled confidence. Medium close-up, slight low angle. Three blurred dark-uniformed officers barely visible behind him. Low amber light. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("091", "Only the man's dark shoes on pale Victorian floorboards, planted firmly and deliberately. Extreme close-up, macro, floor level. The boards look completely ordinary. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("092", "The man from the reference image standing among three seated dark-uniformed police officers, his body language relaxed and conversational. Medium shot from the side. Warm amber Victorian bedroom. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("094", "The man from the reference image still talking but his expression shows effort — his jaw slightly too tight, his eyes slightly too focused. Close-up. Officers blurred behind. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("096", "Only the man's hand gripping a wooden chair armrest with enormous force — knuckles white, tendons visible on the back of the hand. Extreme close-up, macro. Dark jacket sleeve. Warm amber candlelight. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("097", "The man from the reference image leaning forward in a wooden chair, the camera tilted 15 degrees clockwise creating a Dutch tilt. Medium close-up, low angle. Expression: the mask failing, something breaking through. Officers blurred behind. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("098", "The man from the reference image — expression of a man talking too fast to outrun something. Close-up. His eyes too bright, the performance become desperation. Very slight perspiration on the brow. Amber underlight. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("100", "The man from the reference image standing in a Victorian bedroom, the camera tilted 20 degrees clockwise — the most extreme Dutch tilt. Medium close-up, low angle. The room visibly wrong around him. Officers barely visible as shapes at the edges. His face: breaking. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),
    ("101", "Only the man's eye, extreme close-up filling the entire frame. Dark grey-green iris. Pupil massively dilated, nearly consuming the iris. Red-veined whites. Heavy dark circles above and below. Victorian Gothic oil painting, thick textured impasto brushstrokes in the iris. 16:9 landscape.", [NARRATOR]),
    ("103", "The man from the reference image — the confession in his expression, something released, not relief but collapse. Medium close-up. His eyes still wide, his jaw working. The moment of total exposure. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR]),

    # OLD MAN SHOTS
    ("009", "The elderly man from the reference image seated in a plain wooden chair in a warm amber candlelit Victorian bedroom. Close-up, 85mm, high angle 15 degrees. Expression: completely peaceful, trusting, gentle. Warm amber candlelight from below-right illuminating his face in golden tones. Background blurred warm darkness. Victorian Gothic oil painting, raw sienna and deep burnt umber palette. 16:9 landscape.", [OLD_MAN]),
    ("010", "The elderly man from the reference image seated in a plain wooden chair, full figure visible. Medium shot, eye level. Cream collarless nightshirt, grey-green nightcap, worn slippers on feet. Hands folded in lap. A single candle on a three-legged stool beside him. He is entirely at peace. Warm amber. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("034", "The elderly man from the reference image, face in profile on a pillow, eyes fully closed, completely peaceful. Close-up, 85mm, high angle 20 degrees looking down. Short white beard visible. Warm amber candlelight. The closed eyelid centred in frame. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("040", "The elderly man from the reference image sitting completely rigid in bed, profile facing left. Close-up, 85mm, eye level. Eyes open and fixed — not looking at anything, listening. Neck taut. Hands clenched on the blue-grey coverlet. Warm amber trace of light from the right side only. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("041", "Only an elderly man's ear and the side of his head visible — the ear turned toward something off-frame. Extreme close-up, macro, profile. Jaw slightly open, held breath. Very aged, deeply wrinkled skin. A trace of warm amber candlelight on the outer ear cartilage. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("043", "The elderly man from the reference image in bed, one hand pressed flat against his own forehead, eyes closed — trying to reason himself calm. Medium close-up, eye level. Expression: the effort of self-persuasion against rising fear. Warm amber trace light on the side of his face. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("044", "The elderly man from the reference image, eyes reopened — the calm attempt has failed. Close-up, high angle. Eyes wider than before, the fear returned and deeper. Expression: a man who knows something is in the room with him. Warm amber trace. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("046", "The elderly man from the reference image with a large shadow falling diagonally across his face — one half warm amber, one half sudden near-black. Close-up, 85mm. His expression in the lit half: the realisation arriving. Victorian Gothic oil painting, extreme Caravaggio contrast. 16:9 landscape.", [OLD_MAN]),
    ("047", "The elderly man from the reference image sitting upright in bed, both hands pressed flat against his chest over his heart. Medium shot, 85mm, profile facing left. His breathing very shallow. Expression: paralysed, the body knowing what the mind will not accept. Warm amber trace. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("048", "The elderly man from the reference image, mouth very slightly parted — not a scream, barely a breath, an involuntary sound escaping. Medium close-up, high angle. His eyes fixed. The expression of a man who knows death is in the room. Warm amber candlelight from below. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),
    ("058", "The elderly man from the reference image in profile facing left, one hand pressed flat against his chest over his heart. Close-up, eye level. In his neck: a visible pulse. Breathing: tiny shallow sips. Expression: paralysed, knowing. Deep warm darkness behind. Warm amber trace on his face. Victorian Gothic oil painting. 16:9 landscape.", [OLD_MAN]),

    # EYE SHOTS
    ("011", "An elderly man's eye fills the entire 16:9 frame. The iris is a burning deep amber-orange, almost copper-red at the very centre, like the core of an ember. A thin milky-white translucent film partially veils the surface. The pupil is small. Heavy aged eyelid skin surrounding it. Near-total warm darkness around the eye. Victorian Gothic oil painting, thick impasto brushstrokes in the iris, the eye glowing from within. 16:9 landscape.", [EYE]),
    ("012", "An elderly man's eye in side profile. The milky film catches the amber candlelight from the right side, creating an almost iridescent glow. The amber-orange iris visible at the right edge. Heavy aged eyelid skin. Pure warm darkness behind. Victorian Gothic oil painting. 16:9 landscape.", [EYE]),
    ("024", "An elderly man's closed eyelid filling the entire 16:9 frame. Heavy, deeply wrinkled skin. The eye hidden beneath. Warm amber candlelight from below-right creating subtle gold gradients across the folds of skin. The eyelid perfectly, completely still. Thick impasto brushstrokes. Victorian Gothic oil painting. 16:9 landscape.", [EYE]),

    # BEDROOM SHOTS
    ("023", "An old brass mantel clock face, hands positioned precisely at midnight. The brass numerals and hands catch warm amber candlelight. The clock face is cracked and aged. Deep warm darkness behind. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("067", "Victorian wooden floorboards seen from directly above — bird's-eye macro view. Amber-orange light glowing through the seams between boards, pressing upward from beneath. The boards themselves still. The light at the seams pulsing with warmth. Deep darkness at the frame edges. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("069", "A wooden Victorian ball-top bed, slightly pulled away from its original position against the wall. The blue-grey coverlet is disturbed and hanging unevenly. The bed is not where it belongs. Warm amber candlelight. No people visible. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("077", "Looking straight down into an open cavity beneath lifted floorboards — bird's-eye macro view. Darkness between the wooden joists, deep and absolute. Warm amber candlelight from above casts the top surfaces of the joists in warm light but the space between is impenetrable dark. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("079", "Clean Victorian wooden floorboards from directly above — completely undisturbed, boards perfectly aligned. Warm amber candlelight. The wood looks exactly as it always has. Nothing to see. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("081", "A simple Victorian wooden tub on a bedroom floor — round, deep. The amber candlelight catches only the top rim. The interior in complete darkness. The tub sits beside pale wooden floorboards. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("082", "An old brass mantel clock, hands at four o'clock. The room around it near-dark, candle burning very low. Through a high window in the background the very faintest pre-dawn blue-grey light beginning. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("083", "Looking from inside a dark corridor toward a heavy dark wood front door, closed. A fist knock visible — the door very slightly vibrating in its frame, dust falling from the knocker impact. Pure darkness inside. The faintest pre-dawn blue-grey light at the door seams. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("095", "Victorian wooden floorboards bird's-eye macro view. A trace of amber-orange light at the seams between boards — barely there, just beginning. The boards still. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),
    ("099", "Victorian wooden floorboards bird's-eye macro. The amber-orange glow at the seams now clearly visible, pressing upward with intensity. The boards still. The light at the cracks unmistakable. Victorian Gothic oil painting. 16:9 landscape.", [BEDROOM]),

    # NARRATOR + OLD MAN SHOTS
    ("017", "The gaunt narrator man on the left at a simple wooden table, the elderly man on the right seated holding a cup. Medium shot. At the left edge of the frame the narrator's dark jacket sleeve and hand places something on the table gently. Warm amber morning light. Victorian Gothic oil painting. 16:9 landscape.", [NARRATOR, OLD_MAN]),
    ("054", "Split composition — the gaunt narrator man in left half of frame in profile facing right, cold grey stone prison world behind him. The elderly man in right half of frame in profile facing left, warm amber Victorian bedroom behind him. They face each other across the centre divide but are in completely different worlds. Victorian Gothic oil painting, the two palettes meeting at the centre line. 16:9 landscape.", [NARRATOR, OLD_MAN]),
]


def generate_shots():
    """Generate all shots using Gemini API."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not ACCESS_TOKEN:
        print("ERROR: Set GOOGLE_ACCESS_TOKEN environment variable.")
        print("  Run: export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)")
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

    creds = AccessTokenCredentials(ACCESS_TOKEN)
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=creds,
    )

    total = len(SHOTS)
    print(f"\nGenerating {total} shots x {IMAGES_PER_PROMPT} images = {total * IMAGES_PER_PROMPT} total images")
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

                # Generate images
                saved_count = 0
                for img_idx in range(IMAGES_PER_PROMPT):
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
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

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == MAX_RETRIES - 1:
                        print(f"  FAILED after {MAX_RETRIES} retries: {e}")
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
