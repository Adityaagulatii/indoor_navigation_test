import os
import json
import base64
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_PATH  = "C:\\Users\\adity\\Downloads\\tester\\nodemap_tester\\venv\\input_images\\simple-stylish-1024x991.png"   # change to your image filename
OUTPUT_PATH = "annotated_floorplan.jpg"
MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 4000
TEMPERATURE = 0                 # always 0 for consistent results

# ── Pricing (Sonnet) ─────────────────────────────────────────────────────────
COST_PER_INPUT_TOKEN  = 0.000003
COST_PER_OUTPUT_TOKEN = 0.000015

# ── Prompt: Nodes Only ────────────────────────────────────────────────────────
PROMPT = """Analyze this floorplan image.

Find every room, hallway, staircase, elevator,
entrance and exit you can see.

For each space output:
- Short label (max 3 words)
- Bounding box pixel coordinates (x1, y1, x2, y2)
- Center point (cx, cy)
- Type: room | hallway | stairs | elevator | entrance | exit
- Floor number

Respond ONLY with valid JSON, no markdown, no explanation:
{
  "nodes": [
    {
      "id": "N1",
      "label": "Room 125",
      "type": "room",
      "floor": 1,
      "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
      "center": {"cx": 50, "cy": 50}
    }
  ]
}

Rules:
- Keep labels short, max 3 words
- Only add nodes you can clearly see
- bbox and center must be pixel coordinates
- center point must be inside the bbox"""


# ── Step 1: Load API Key ──────────────────────────────────────────────────────
def load_api_key():
    load_dotenv('test.env')
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("❌ No API key found. Check your .env file")
        exit()
    print("✅ API key loaded")
    return key


# ── Step 2: Load and Encode Image ─────────────────────────────────────────────
def load_image(path):
    if not os.path.exists(path):
        print(f"❌ Image not found: {path}")
        exit()

    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = path.split(".")[-1].lower()
    media_types = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "gif":  "image/gif",
        "webp": "image/webp"
    }
    media_type = media_types.get(ext, "image/jpeg")

    img    = Image.open(path)
    w, h   = img.size

    print(f"✅ Image loaded : {path}")
    print(f"✅ Image type   : {media_type}")
    print(f"✅ Image size   : {w} x {h} pixels")

    return image_data, media_type, w, h


# ── Step 3: Call Claude ───────────────────────────────────────────────────────
def call_claude(client, image_data, media_type):
    print("\n⏳ Sending image to Claude...")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": media_type,
                            "data":       image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ]
    )

    print("✅ Response received")
    return response


# ── Step 4: Parse JSON ────────────────────────────────────────────────────────
def parse_response(response):
    raw = response.content[0].text

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    try:
        data  = json.loads(raw.strip())
        nodes = data.get("nodes", [])
        print(f"✅ Valid JSON parsed")
        print(f"   Nodes found : {len(nodes)}")
        return nodes
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print("\n── Raw response ──────────────────────────")
        print(raw)
        return None


# ── Step 5: Draw Annotations ──────────────────────────────────────────────────
def draw_annotations(image_path, nodes, output_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Could not open image: {image_path}")
        return

    for node in nodes:
        bbox   = node.get("bbox",   {})
        center = node.get("center", {})
        label  = node.get("label",  node.get("id", "?"))

        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        cx = int(center.get("cx", (x1 + x2) // 2))
        cy = int(center.get("cy", (y1 + y2) // 2))

        # Semi-transparent blue fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 150, 0), -1)
        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

        # Blue bounding box border
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 0), 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img,
                      (x1, y1),
                      (x1 + tw + 6, y1 + th + 6),
                      (0, 0, 0), -1)

        # Room label
        cv2.putText(img, label,
                    (x1 + 3, y1 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        # Red dot at center
        cv2.circle(img, (cx, cy), 8,  (0, 0, 255), -1)
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), 1)

    cv2.imwrite(output_path, img)
    print(f"\n✅ Annotated image saved : {output_path}")

    cv2.imshow("Node Map — Press any key to close", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Step 6: Save JSON + Print Cost ────────────────────────────────────────────
def save_and_summarise(nodes, response):
    # Save nodes to JSON file
    with open("nodes.json", "w") as f:
        json.dump({"nodes": nodes}, f, indent=2)
    print("✅ Nodes saved     : nodes.json")

    # Cost
    in_tok  = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    cost    = (in_tok  * COST_PER_INPUT_TOKEN) + \
              (out_tok * COST_PER_OUTPUT_TOKEN)

    print("\n── Token Usage ────────────────────────────────")
    print(f"   Input tokens  : {in_tok}")
    print(f"   Output tokens : {out_tok}")
    print(f"   Estimated cost: ${cost:.6f}")
    print("───────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🗺️  Claude Node Extractor — Call 1 (Nodes Only)")
    print("=" * 50)

    # Step 1 — API key
    api_key = load_api_key()
    client  = anthropic.Anthropic(api_key=api_key)
    print("✅ Claude client ready")

    # Step 2 — Load image
    image_data, media_type, w, h = load_image(IMAGE_PATH)

    # Step 3 — Call Claude
    response = call_claude(client, image_data, media_type)

    # Step 4 — Parse JSON
    nodes = parse_response(response)

    if nodes:
        # Step 5 — Draw annotations
        draw_annotations(IMAGE_PATH, nodes, OUTPUT_PATH)

        # Step 6 — Save + cost
        save_and_summarise(nodes, response)


if __name__ == "__main__":
    main()