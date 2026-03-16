#!/usr/bin/env python3
"""Generate priority shots for Tell-Tale Heart using Gemini."""

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

# --- Priority shots ---
SHOTS = [
    ("002", "ENVIRONMENT: A rough grey stone prison wall pressed directly behind the subject, filling the entire background as a flat cold textured plane. The wall surface is visible but slightly out of focus. No furniture, no room — only cold grey stone behind him.\nLIGHT: Flat overcast cold light falling from the upper left. No shadows cast. No warmth anywhere. Purely forensic grey illumination throughout the entire frame.\nCHARACTER — use the reference image exactly: A tall gaunt man. His face is hollow and angular with deeply sunken cheeks. His hair is dark, damp, falling loosely across his forehead. His eyes are very deep-set and cavernous, surrounded by heavy purple-black rings of exhaustion. His cheekbones are razor-sharp and prominent. He has thin dark stubble across his jaw. He is wearing a worn dark charcoal wool jacket with visible lapels. Beneath it a grey open-collar shirt with the top button undone. This is his exact appearance — do not alter the face, do not make him younger, do not make him healthier, do not change his clothing.\nEXPRESSION AND POSE: His face fills the frame from chest upward. He is looking directly and unflinchingly into the lens. His jaw is firmly set. His lips are pressed together. The muscles around his eyes are tight with controlled urgency. He is a man who demands to be believed.\nCAMERA: Close-up portrait. 85mm lens perspective. Eye level. Shallow depth of field — the stone wall behind him is softly blurred.\nSTYLE: Oil painting. Thick impasto brushstrokes throughout. Visible canvas texture. Caravaggio-style chiaroscuro. In the style of Rembrandt and Lucian Freud. Victorian Gothic atmosphere. Payne's grey and raw umber colour palette. 16:9 widescreen landscape format.", [NARRATOR]),

    ("065", "ENVIRONMENT — use the reference bedroom exactly and do not change any of the following: The bedroom walls are covered in dark green-grey peeling wallpaper with large torn and peeling patches. The floor is pale wooden planks running horizontally. Against the RIGHT side of the frame is a wooden bed with ball-shaped finials on the bedposts and a blue-grey coverlet. Against the LEFT wall is a dark wooden chest of drawers. Above the chest of drawers on the upper left wall is a dark-framed painting. A three-legged wooden stool stands near the bed. The door frame is at the far right edge. These positions do not change.\nLIGHT: A single candle is positioned on the stool behind the standing figure, providing warm amber backlighting from behind him. The candle flame is the sole light source. It casts amber light across the wallpaper and floor behind him. The figure himself is backlit — his front face is in shadow, his outline catches amber light.\nCHARACTER — the figure is seen only from behind as a near-silhouette: A tall gaunt man. His back is entirely to the viewer. His worn dark charcoal wool jacket with lapels is visible as a silhouette. Both arms are raised fully above his head. He is gripping a plain wooden chair in both hands, arms fully extended upward, the chair raised at the highest point above his head. His figure and the raised chair form a strong dark silhouette against the warm amber candlelight behind him.\nCAMERA: Medium shot. 35mm lens perspective. From behind and low angle — camera looks up at him from below.\nSTYLE: Oil painting. Thick impasto brushstrokes. Caravaggio-style chiaroscuro. Strong dark silhouette against warm amber backlighting. In the style of Rembrandt. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [NARRATOR, BEDROOM]),

    ("068", "ENVIRONMENT — use the reference bedroom exactly and do not change any of the following: The bedroom walls are covered in dark green-grey peeling wallpaper with large torn and peeling patches. The floor is pale wooden planks. Against the LEFT wall is a dark wooden chest of drawers. Above it on the upper left wall is a dark-framed painting. A three-legged wooden stool stands near the bed. The door frame is at the right edge. The wooden ball-top bed is present but its blue-grey coverlet is pulled off and dishevelled, and the bed has been pushed slightly away from the wall — it is visibly not in its original position.\nLIGHT: A single candle sits on the floor near the bottom edge of the frame. It provides warm low amber light as the sole source, casting light upward at a low angle across the floorboards and the figure.\nCHARACTER — seen from behind, do not show the face: A tall gaunt man. His back is entirely to the viewer. He is wearing his worn dark charcoal wool jacket with visible lapels. He is standing still in the centre-left of the room. He is looking downward at the pale wooden floor and the displaced empty bed. His shoulders and spine carry the physical posture of completion — absolute stillness after a finished act. No other person is present in the room.\nCAMERA: Medium shot. 35mm lens perspective. From behind and slightly elevated — camera looks down at him from slightly above.\nSTYLE: Oil painting. Thick impasto brushstrokes. Caravaggio-style chiaroscuro. In the style of Rembrandt. Victorian Gothic atmosphere. Warm amber low floor-level candlelight. 16:9 widescreen landscape format.", [NARRATOR, BEDROOM]),

    ("084", "ENVIRONMENT: A dark Victorian interior hallway. The walls and floor are barely visible in near-total darkness. The corridor stretches toward the far end. At the far end: a heavy dark wooden front door standing fully open. No bedroom furniture appears in this shot — this is the entrance hallway of the house, entirely separate from the bedroom. Behind the viewer in the interior: traces of warm amber from candlelight inside the house. The exterior beyond the open door: cold pale blue pre-dawn light.\nFIGURES: In the open doorway stand three Victorian police officer silhouettes. Their dark blue Victorian uniforms and round police helmets are barely distinguishable as shapes against the cold blue exterior light behind them. The cold blue light outlines them from behind. They are silhouettes only — no faces visible. No narrator visible in this shot.\nLIGHT: The interior hallway carries warm amber traces. The exterior beyond the open door is cold pale blue pre-dawn. The contrast is maximum — warm amber inside, cold blue outside, dark silhouettes in the threshold between the two worlds.\nCAMERA: Wide shot. 35mm lens perspective. Eye level from inside the hallway looking directly toward the open front door.\nSTYLE: Oil painting. Thick impasto brushstrokes. Caravaggio-style chiaroscuro. In the style of Rembrandt. Warm amber interior meeting cold blue exterior at the open doorway. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", []),

    ("085", "ENVIRONMENT: A dark Victorian front hallway interior. The walls are dark and barely visible. Warm amber light from inside the house illuminates the scene from the left side. Three dark-uniformed Victorian police officers with round helmets are entering from the right side of the frame as slightly soft shapes — their forms are slightly out of focus.\nLIGHT: Warm amber from inside the house illuminating everything. The officers are slightly blurred. The narrator is sharply in focus.\nCHARACTER — use the reference image exactly: A tall gaunt man. His face is hollow and angular. His hair is dark, damp, falling across his forehead. His eyes are very deep-set and cavernous with heavy purple-black rings. His cheekbones are razor-sharp. Thin dark stubble. Wearing his worn dark charcoal wool jacket with lapels and grey open-collar shirt. He is standing at the open front door. One arm is extended outward in a welcoming gesture toward the officers. His body language is open, generous, and unhurried. His expression: the physical quality of complete ease — the face of a man with nothing to conceal. His lips are very slightly lifted — not a full smile, but the facial arrangement of a man performing warmth.\nCAMERA: Medium shot. 50mm lens perspective. From the side — both the narrator and the entering officers are visible simultaneously in the same frame.\nSTYLE: Oil painting. Thick impasto brushstrokes. In the style of Rembrandt. Victorian Gothic atmosphere. Warm amber interior. 16:9 widescreen landscape format.", [NARRATOR]),

    ("097", "ENVIRONMENT — use the reference bedroom exactly but apply the Dutch tilt: The bedroom walls are covered in dark green-grey peeling wallpaper with large torn patches. The floor is pale wooden planks. Against the LEFT wall (in the tilted frame) is a dark wooden chest of drawers. Above it is a dark-framed painting. A three-legged wooden stool stands near the bed. The wooden ball-top bed with blue-grey coverlet is present. The door frame is at the right edge. The ENTIRE ROOM is tilted 15 degrees clockwise — every wall, every piece of furniture, the floor, the ceiling, all lean at this wrong angle simultaneously. Three blurred dark-uniformed Victorian police officer shapes are visible as soft dark presences in the tilted background.\nLIGHT: Warm amber candlelight from below the subject, casting upward shadows. The light quality is slightly chaotic and unstable.\nCHARACTER — use the reference image exactly: A tall gaunt man. Hollow angular face. Dark damp hair across his forehead. Very deep-set cavernous eyes with heavy purple-black rings. Razor-sharp cheekbones. Thin dark stubble. Worn dark charcoal wool jacket with lapels. Grey open-collar shirt. He is seated in a plain wooden chair, leaning slightly forward. His expression shows the performance actively failing — the muscles around his eyes have loosened from their usual control, his jaw is set in a wrong over-tight way, his skin is fractionally paler than normal. Something is rising through from beneath the surface.\nCAMERA: Medium close-up. Low angle — camera looks up at him. The camera itself is physically rotated 15 degrees clockwise, making the entire room tilt at this wrong angle in the frame.\nSTYLE: Oil painting. Thick impasto brushstrokes. Caravaggio-style chiaroscuro. In the style of Rembrandt and Lucian Freud. Dutch tilt 15 degrees making the entire room spatially wrong. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [NARRATOR, BEDROOM]),

    ("100", "ENVIRONMENT — use the reference bedroom exactly but apply the most extreme Dutch tilt: The bedroom walls are covered in dark green-grey peeling wallpaper with large torn patches. The floor is pale wooden planks. The dark wooden chest of drawers is on the LEFT wall. The dark-framed painting is above it. The three-legged stool. The wooden ball-top bed. The door frame at the right edge. The ENTIRE ROOM is tilted 20 degrees clockwise — this is the most extreme tilt in the film, making the room feel physically threatening and wrong. Dark officer shapes are barely visible as indistinct dark masses at the extreme distorted edges of the frame. Warm amber and cold blue light mix chaotically without logical direction.\nLIGHT: Warm amber and cold blue mixing chaotically throughout the frame. No logical single light source — the light itself is breaking down.\nCHARACTER — use the reference image exactly: A tall gaunt man. Hollow angular face. Dark damp hair across his forehead. Very deep-set cavernous eyes with heavy purple-black rings. Razor-sharp cheekbones. Thin dark stubble. Worn dark charcoal wool jacket with lapels. Grey open-collar shirt. He is standing, slightly forward. Every muscle of his face has released entirely from its performance — the eyes are wide and completely uncontrolled, the jaw is loose and working, the brow is unclenched and exposed. The full physical expression of a mind that has broken through its own containment.\nCAMERA: Medium close-up. Low angle — camera looks up at him from below. The camera is physically rotated 20 degrees clockwise, making the entire room lean severely.\nSTYLE: Oil painting. Thick impasto brushstrokes. Caravaggio-style chiaroscuro. In the style of Rembrandt and Lucian Freud. Dutch tilt 20 degrees — most extreme in the film. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [NARRATOR, BEDROOM]),

    ("104", "ENVIRONMENT: A Victorian prison exercise yard. The composition is perfectly and deliberately symmetrical — everything is centred to the mathematical centre of the frame. High grey stone walls rise on both sides of the frame — identical height and texture on both sides. Cold pale grey overcast sky above — no blue, no warmth, only grey. Grey cobblestone floor below, stretching toward the far wall. Centred perfectly in the far wall: an arched iron gate, completely closed. A rusted chain is coiled directly on the cobblestones before the gate. At the LEFT wall: a single iron wall-mounted lantern, unlit, cold and dark. A rusted chain visible at the far right edge of the frame. No people anywhere in the frame. No movement. No warmth anywhere — not a single trace of amber, orange, or warm colour anywhere in the entire image.\nLIGHT: Cold flat pale grey overcast light only. Completely even. No shadows. No warmth.\nCAMERA: Extra-wide shot. 24mm lens perspective. Eye level. Camera perfectly centred and symmetrical.\nSTYLE: Oil painting. Thick impasto brushstrokes. Visible canvas texture. In the style of Rembrandt. Pure Payne's grey throughout every element of the frame. Not a single trace of warmth anywhere. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", []),

    ("076", "ENVIRONMENT: The pale wooden plank floor of the Victorian bedroom — the same pale wooden floorboards from the reference bedroom. The floor fills most of the frame as seen from directly above. The grain of the pale warm wood is clearly visible. Warm amber candlelight from the right side illuminates the floor surface.\nCHARACTER — hands only, use reference for skin and sleeve character: Two gaunt pale hands with prominent knuckles, bony fingers, and thin skin. The fingers are wedged into the narrow gap between two pale wooden floorboards. Both hands grip the edges of one board. The board has begun to lift at one edge — the darkness of the space beneath the floor is just becoming visible as a very thin dark line where the lifted board edge separates from the surrounding floor. A dark charcoal wool jacket sleeve is visible at both wrists. These are the same gaunt hands belonging to the narrator — pale, thin-skinned, prominent tendons.\nCAMERA: Extreme close-up macro. From slightly above, looking down at the hands and the floor. The floor fills the frame.\nSTYLE: Oil painting. Thick impasto brushstrokes. In the style of Rembrandt and Lucian Freud. Warm amber on pale wood grain. The thin dark line of darkness beneath the board is just beginning to appear. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [NARRATOR]),

    ("102", "ENVIRONMENT — use the reference bedroom with ultra-wide distortion applied to everything: The bedroom walls covered in dark green-grey peeling wallpaper with large torn patches — but distorted by the ultra-wide lens so the walls bow outward at the edges. The pale wooden plank floor stretches away in extreme and exaggerated perspective beneath the subject. The dark wooden chest of drawers on the LEFT wall is visible but distorted. The dark-framed painting above it is visible but distorted. The three-legged stool. The wooden ball-top bed with blue-grey coverlet visible but distorted at the edge. These elements are all present but bent outward by the extreme 14mm ultra-wide lens. Dark Victorian police officer shapes are barely visible as indistinct dark masses at the extreme distorted frame edges. Warm amber and cold blue light mixing chaotically.\nLIGHT: Warm amber and cold blue mixing without order. Chaotic. The light source is unclear.\nCHARACTER — use the reference image exactly: A tall gaunt man. Hollow angular face. Dark damp hair across his forehead. Very deep-set cavernous eyes with heavy purple-black rings. Razor-sharp cheekbones. Thin dark stubble. Worn dark charcoal wool jacket with lapels. Grey open-collar shirt. He is standing upright with both arms thrown wide to his sides. His head is tilted back. His body is in the posture of complete and total release — completely open, completely exposed, arms wide. The pale floorboards stretch away in extreme distorted perspective beneath his feet.\nCAMERA: Wide shot. 14mm ultra-wide lens perspective — room edges bow outward visibly. Floor-level low angle looking up at him.\nSTYLE: Oil painting. Thick impasto brushstrokes. Caravaggio-style chiaroscuro. In the style of Rembrandt. Spatially distorted ultra-wide perspective with room edges bowing outward. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [NARRATOR, BEDROOM]),

    ("024", "ENVIRONMENT: Near-total warm darkness surrounding the subject. Deep warm shadow fills the frame. Only the eyelid is lit.\nLIGHT: Warm amber candlelight from below-right only. Barely touching the eyelid surface with the softest possible golden light. Everything else in the frame is in deep warm shadow.\nSUBJECT — use the reference images for skin character: A single closed elderly eyelid fills the entire frame from edge to edge. This is the eye of the very elderly man from the old man reference image — the same aged papery skin, same deep wrinkles. The eyelid surface carries heavy, deeply carved wrinkles across it. The skin is ancient and layered — built up in geological folds of aged flesh. The eyelid is perfectly and completely still. Not trembling. Not moving. Absolutely closed. The burning amber-orange iris from the eye reference image is completely hidden beneath it. The stillness is the entire subject of this image. No other body part is visible — only the eyelid fills the frame completely.\nCAMERA: Extreme close-up macro. Fills the entire frame. Direct frontal view.\nSTYLE: Oil painting. Thick impasto brushstrokes. Visible canvas texture. In the style of Rembrandt and Lucian Freud. Layered raw sienna, burnt umber, and yellow ochre in the skin tones. Warm amber trace from below-right only. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [OLD_MAN, EYE]),

    ("038", "ENVIRONMENT — split environment, two worlds visible simultaneously: FOREGROUND — the inside of the reference bedroom: pale wooden plank floor, dark green-grey peeling wallpaper with torn patches, warm amber candlelight filling the room. The dark wooden chest of drawers visible on the left. The dark-framed painting above it. The three-legged stool. The wooden ball-top bed with blue-grey coverlet to one side. BACKGROUND — through the open doorway behind the figure: a dark stone corridor, cold and grey, slightly less dark than black. The doorway is the dividing threshold between these two worlds.\nLIGHT: Warm amber from the bedroom in the foreground. Cold grey darkness from the corridor behind the figure. The figure himself is mostly in shadow — the sharpest edge of his cheekbone catches a single sliver of amber light.\nCHARACTER — use the reference image exactly: A tall gaunt man. His face is in three-quarter shadow — only the razor-sharp edge of his cheekbone catches a sliver of amber light from the room. Dark damp hair. Worn dark charcoal wool jacket with lapels. His right hand is concealed behind his back. He is standing in the open doorway with one foot just across the bedroom threshold. His entire body is frozen in position — a statue. He is not moving. He will not move.\nCAMERA: Medium shot. 35mm lens perspective. From inside the bedroom looking toward him in the doorway. Camera is at floor level, low angle looking up at him.\nSTYLE: Oil painting. Thick impasto brushstrokes. Caravaggio-style chiaroscuro. In the style of Rembrandt. Warm amber bedroom darkness in the foreground meeting cold corridor darkness behind the figure at the doorway threshold. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [NARRATOR, BEDROOM]),

    ("036", "ENVIRONMENT: Pure near-total darkness. Only the lantern and the hand are visible. No room, no background — pure black surrounding.\nLIGHT: The thick shaft of escaping amber-gold light from the lantern's accidentally opened panel is the only illumination in the entire frame. Everything else is near-black. This single shaft of amber light is too wide, too bright — it is clearly an accident.\nSUBJECT — use the narrator reference for hand character: A dark Victorian tin lantern. Its panel has accidentally opened wider than intended. A thick shaft of warm amber-gold light is escaping forcefully through the gap — visibly too wide and too bright, unmistakably the quality of an accident. A thumb is pressed against the tin panel at a slightly wrong angle — it has slipped from its intended position. The thumb is pale, gaunt, bony — the same gaunt hand belonging to the narrator from the reference image. The thick shaft of amber-gold light is the only light in the frame.\nCAMERA: Extreme close-up macro. Looking slightly down at the lantern and the hand.\nSTYLE: Oil painting. Thick impasto brushstrokes. In the style of Rembrandt. Near-total darkness with one thick accidentally escaping amber shaft of light. Victorian Gothic atmosphere. 16:9 widescreen landscape format.", [NARRATOR]),
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
    print(f"\nGenerating {total} priority shots x {IMAGES_PER_PROMPT} images = {total * IMAGES_PER_PROMPT} total images")
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

        print(f"[{idx}/{total}] Shot {shot_num}: generating...")

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
