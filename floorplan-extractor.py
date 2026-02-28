import os
import json
import base64
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import anthropic
import time

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_PATH  = "IMG_5840.jpeg"
OUTPUT_DIR  = "output"
MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 8000
TEMPERATURE = 0

# Tiling config
TILE_OVERLAP = 0.25
TILE_ROWS    = 3
TILE_COLS    = 2

# Image preprocessing
MAX_IMAGE_DIM    = 2048
CONTRAST_ALPHA   = 1.3
BRIGHTNESS_BETA  = 10

# Dedup
IOU_THRESHOLD = 0.35

# Retry
MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2

# Pricing (Sonnet)
COST_PER_INPUT_TOKEN  = 0.000003
COST_PER_OUTPUT_TOKEN = 0.000015

# Max base64 image size for Claude API
MAX_IMAGE_BYTES = 5_000_000


# ── Prompts ─────────────────────────────────────────────────────────────────

NODE_PROMPT = """You are a floor plan analysis system with perfect attention to detail.

This image shows {tile_desc}a floor plan that is {width}x{height} pixels.

TASK: Find EVERY distinct labeled space in this image. Scan the ENTIRE image
systematically — top to bottom, left to right. Do NOT skip any area.

Common space types on floor plans:
- Rooms with numbers (e.g. "123A", "CN-115", "035", "072")
- Open spaces (e.g. "CN-129 OPEN SPACE")
- Hallways, corridors, lobbies
- Bathrooms (look for restroom icons — male/female symbols)
- Staircases, stairwells (e.g. "STAIRWELL TO BASEMENT")
- POD areas (phone booths or small meeting pods)
- Service areas, entrances, exits
- Labeled desks or areas (e.g. "TC", "HS", "DP", "NM", "SA", "SD")
- Archives, reading rooms, collections areas, loading docks

For each space provide:
- id: unique identifier (N1, N2, N3, ...)
- label: the EXACT text written on the plan. If multiple lines, join with space.
- type: one of [room, open_space, hallway, corridor, bathroom, kitchen,
        stairs, elevator, entrance, exit, pod, service_area, utility,
        archives, reading_room, collections, other]
- bbox: pixel coordinates {{x1, y1, x2, y2}} — tight bounding box around the space
        (x1,y1) = top-left, (x2,y2) = bottom-right
- center: {{cx, cy}} — center point inside the space

RULES:
1. Read labels EXACTLY as printed — do not guess or invent.
2. EVERY labeled area must be its own node — do not merge adjacent spaces.
3. Include ALL POD areas as separate nodes even if they look similar.
4. Bathrooms may only have icons (male/female symbols) — still include them.
5. Hallways and corridors between rooms are nodes too.
6. All coordinates must be within 0..{width} (x) and 0..{height} (y).
7. Double-check: scan every quadrant — rooms are often missed at edges.
8. If you see an ENTRANCE or EXIT label, include it.

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "nodes": [
    {{
      "id": "N1",
      "label": "Room 101",
      "type": "room",
      "bbox": {{"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
      "center": {{"cx": 50, "cy": 50}}
    }}
  ]
}}"""


VERIFY_PROMPT = """You are a floor plan verification system.

This floor plan is {width}x{height} pixels. A previous analysis found these nodes:

{nodes_json}

TASK: Look at the floor plan image carefully and find any MISSING spaces that
are NOT in the list above. Check especially:
- The bottom half of the image
- Small rooms or pods
- Hallways and corridors
- Bathrooms (restroom icon areas)
- Stairwells and entrances/exits
- Any labeled area not in the list

If you find missing spaces, return them as new nodes with IDs continuing from N{next_id}.
If nothing is missing, return an empty nodes array.

Respond ONLY with valid JSON:
{{
  "nodes": []
}}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_api_key():
    load_dotenv("test.env")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY found in test.env")
    print("✅ API key loaded")
    return key


def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")

    h, w = img.shape[:2]
    print(f"   Original size: {w}x{h}")

    scale = 1.0
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        print(f"   Resized to:    {w}x{h} (scale={scale:.3f})")

    enhanced = cv2.convertScaleAbs(img, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    preprocessed_path = os.path.join(OUTPUT_DIR, "preprocessed.png")
    cv2.imwrite(preprocessed_path, enhanced)
    print(f"✅ Preprocessed image saved: {preprocessed_path}")

    return enhanced, w, h, scale


def encode_image(img_array):
    """Encode image to base64. Uses PNG if small enough, else JPEG with compression."""
    # Try PNG first
    _, buffer = cv2.imencode(".png", img_array)
    if len(buffer) <= MAX_IMAGE_BYTES:
        b64 = base64.standard_b64encode(buffer).decode("utf-8")
        return b64, "image/png"

    # PNG too large — try JPEG at decreasing quality
    for quality in [92, 85, 75, 60, 45]:
        _, buffer = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if len(buffer) <= MAX_IMAGE_BYTES:
            b64 = base64.standard_b64encode(buffer).decode("utf-8")
            print(f"   ℹ️  Compressed to JPEG quality={quality} ({len(buffer)/1e6:.1f}MB)")
            return b64, "image/jpeg"

    # Last resort: resize down and compress
    h, w = img_array.shape[:2]
    img_small = cv2.resize(img_array, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.standard_b64encode(buffer).decode("utf-8")
    print(f"   ⚠️  Resized to {w//2}x{h//2} + JPEG to fit 5MB limit")
    return b64, "image/jpeg"


def call_claude(client, image_b64, media_type, prompt, pass_name=""):
    """Send image + prompt to Claude with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"   ⏳ {pass_name} — attempt {attempt}/{MAX_RETRIES}...")
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            print(f"   ✅ {pass_name} — response received")
            return response

        except anthropic.BadRequestError as e:
            print(f"   ❌ Bad request (not retryable): {e}")
            raise
        except anthropic.APIError as e:
            print(f"   ⚠️  API error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                raise


def parse_json_response(response):
    raw = response.content[0].text.strip()

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    print("❌ Could not parse JSON from response:")
    print(raw[:500])
    return None


# ── Tiling ──────────────────────────────────────────────────────────────────

def create_tiles(img, rows, cols, overlap):
    h, w = img.shape[:2]
    tiles = []

    tile_h = h // rows
    tile_w = w // cols
    overlap_h = int(tile_h * overlap)
    overlap_w = int(tile_w * overlap)

    for r in range(rows):
        for c in range(cols):
            y1 = max(0, r * tile_h - overlap_h)
            y2 = min(h, (r + 1) * tile_h + overlap_h)
            x1 = max(0, c * tile_w - overlap_w)
            x2 = min(w, (c + 1) * tile_w + overlap_w)

            tile = img[y1:y2, x1:x2]
            tiles.append((tile, x1, y1))
            print(f"   Tile [{r},{c}]: offset=({x1},{y1}), size={x2-x1}x{y2-y1}")

    return tiles


def offset_nodes(nodes, x_off, y_off):
    for node in nodes:
        bbox = node["bbox"]
        bbox["x1"] += x_off
        bbox["y1"] += y_off
        bbox["x2"] += x_off
        bbox["y2"] += y_off
        node["center"]["cx"] += x_off
        node["center"]["cy"] += y_off
    return nodes


def compute_iou(box_a, box_b):
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def deduplicate_nodes(all_nodes, iou_threshold=IOU_THRESHOLD):
    if not all_nodes:
        return []

    all_nodes.sort(
        key=lambda n: (n["bbox"]["x2"] - n["bbox"]["x1"]) *
                      (n["bbox"]["y2"] - n["bbox"]["y1"]),
        reverse=True,
    )

    kept = []
    for node in all_nodes:
        is_dup = False
        for existing in kept:
            iou = compute_iou(node["bbox"], existing["bbox"])
            same_label = (node["label"].strip().lower() == existing["label"].strip().lower())
            if iou > iou_threshold or (same_label and iou > 0.15):
                is_dup = True
                break
        if not is_dup:
            kept.append(node)

    print(f"   ✅ Deduplication: {len(all_nodes)} → {len(kept)} unique nodes")
    return kept


# ── Validation ──────────────────────────────────────────────────────────────

def validate_nodes(nodes, img_w, img_h):
    valid = []
    for node in nodes:
        bbox = node.get("bbox", {})

        x1 = max(0, min(int(bbox.get("x1", 0)), img_w - 1))
        y1 = max(0, min(int(bbox.get("y1", 0)), img_h - 1))
        x2 = max(x1 + 1, min(int(bbox.get("x2", img_w)), img_w))
        y2 = max(y1 + 1, min(int(bbox.get("y2", img_h)), img_h))

        node["bbox"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        node["center"] = {"cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2}

        area = (x2 - x1) * (y2 - y1)
        if area < 100:
            print(f"   ⚠️  Skipping {node.get('id')} — too small ({area}px²)")
            continue

        valid.append(node)

    print(f"   ✅ {len(valid)}/{len(nodes)} nodes passed validation")
    return valid


def reassign_ids(nodes):
    for i, node in enumerate(nodes, 1):
        node["id"] = f"N{i}"
    return nodes


# ── Visualization ───────────────────────────────────────────────────────────

TYPE_COLORS = {
    "room":         (255, 150, 0),
    "open_space":   (255, 200, 50),
    "hallway":      (0, 200, 200),
    "corridor":     (0, 200, 200),
    "bathroom":     (200, 100, 255),
    "kitchen":      (0, 180, 0),
    "stairs":       (0, 0, 255),
    "elevator":     (0, 0, 200),
    "entrance":     (0, 255, 0),
    "exit":         (0, 255, 0),
    "pod":          (255, 100, 100),
    "service_area": (100, 200, 255),
    "lobby":        (255, 200, 0),
    "office":       (255, 150, 0),
    "utility":      (150, 150, 150),
    "archives":     (200, 180, 100),
    "reading_room": (180, 200, 255),
    "collections":  (160, 160, 200),
    "other":        (180, 180, 180),
}


def draw_nodes(image_path, nodes, output_path, scale=1.0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot open image: {image_path}")
        return

    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    for node in nodes:
        bbox  = node["bbox"]
        cx, cy = node["center"]["cx"], node["center"]["cy"]
        label = node.get("label", node["id"])
        ntype = node.get("type", "other")
        color = TYPE_COLORS.get(ntype, (180, 180, 180))

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        # Semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

        # Border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label with background
        display = f"{node['id']}: {label}"
        font_scale = 0.4
        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        label_y = y1 - 4 if y1 > th + 10 else y1 + th + 6
        label_bg_y1 = label_y - th - 4 if y1 > th + 10 else y1
        label_bg_y2 = label_y + 2 if y1 > th + 10 else y1 + th + 10

        cv2.rectangle(img, (x1, label_bg_y1), (x1 + tw + 6, label_bg_y2), (0, 0, 0), -1)
        cv2.putText(img, display, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Center dot
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1)

    cv2.imwrite(output_path, img)
    print(f"\n✅ Annotated image saved: {output_path}")


# ── Cost Tracking ───────────────────────────────────────────────────────────

class CostTracker:
    def __init__(self):
        self.total_input  = 0
        self.total_output = 0
        self.calls        = 0

    def add(self, response):
        self.total_input  += response.usage.input_tokens
        self.total_output += response.usage.output_tokens
        self.calls        += 1

    def report(self):
        cost = (self.total_input  * COST_PER_INPUT_TOKEN +
                self.total_output * COST_PER_OUTPUT_TOKEN)
        print(f"\n── Cost Summary ({self.calls} API calls) ──────────────")
        print(f"   Input tokens  : {self.total_input}")
        print(f"   Output tokens : {self.total_output}")
        print(f"   Estimated cost: ${cost:.6f}")
        print("───────────────────────────────────────────────")


# ── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    print("\n🗺️  Floor Plan Node Extractor v2 (Tiled + Verification)")
    print("=" * 58)

    ensure_output_dir()
    api_key = load_api_key()
    client  = anthropic.Anthropic(api_key=api_key)
    tracker = CostTracker()

    # ── Preprocess ──
    print("\n📷 Step 1: Preprocessing image...")
    enhanced, img_w, img_h, scale = preprocess_image(IMAGE_PATH)

    # ── Tiled Node Extraction ──
    print(f"\n🔍 Step 2: Tiled node extraction ({TILE_ROWS}x{TILE_COLS} tiles)...")
    tiles = create_tiles(enhanced, TILE_ROWS, TILE_COLS, TILE_OVERLAP)

    all_nodes = []
    for idx, (tile_img, x_off, y_off) in enumerate(tiles):
        tile_h, tile_w = tile_img.shape[:2]
        tile_b64, tile_media = encode_image(tile_img)

        prompt = NODE_PROMPT.format(
            width=tile_w, height=tile_h,
            tile_desc="a section of " if len(tiles) > 1 else ""
        )

        resp = call_claude(client, tile_b64, tile_media, prompt, f"Tile {idx+1}/{len(tiles)}")
        tracker.add(resp)

        data = parse_json_response(resp)
        if data and "nodes" in data:
            tile_nodes = data["nodes"]
            tile_nodes = offset_nodes(tile_nodes, x_off, y_off)
            tile_nodes = validate_nodes(tile_nodes, img_w, img_h)
            print(f"   Tile {idx+1}: {len(tile_nodes)} nodes found")
            all_nodes.extend(tile_nodes)
        else:
            print(f"   ⚠️  Tile {idx+1}: no nodes extracted")

    print(f"\n   Total raw nodes from all tiles: {len(all_nodes)}")

    # ── Deduplicate overlapping detections ──
    print("\n🧹 Step 3: Deduplicating overlapping detections...")
    nodes = deduplicate_nodes(all_nodes)

    # ── Verification pass on full image ──
    print("\n🔎 Step 4: Verification pass (checking for missed rooms)...")
    full_b64, full_media = encode_image(enhanced)
    nodes_summary = json.dumps(
        [{"id": n["id"], "label": n["label"], "type": n["type"],
          "center": n["center"]} for n in nodes],
        indent=2,
    )
    verify_prompt = VERIFY_PROMPT.format(
        width=img_w, height=img_h,
        nodes_json=nodes_summary,
        next_id=len(nodes) + 1,
    )
    verify_resp = call_claude(client, full_b64, full_media, verify_prompt, "Verification")
    tracker.add(verify_resp)

    verify_data = parse_json_response(verify_resp)
    if verify_data and verify_data.get("nodes"):
        new_nodes = validate_nodes(verify_data["nodes"], img_w, img_h)
        for nn in new_nodes:
            is_dup = False
            for existing in nodes:
                iou = compute_iou(nn["bbox"], existing["bbox"])
                if iou > IOU_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                nodes.append(nn)
                print(f"   ✅ Verification found missed node: {nn['label']}")

        if not new_nodes:
            print("   ✅ Verification: no missed rooms")
    else:
        print("   ✅ Verification: no missed rooms")

    # ── Reassign clean IDs ──
    nodes = reassign_ids(nodes)

    # ── Print results ──
    print(f"\n{'─'*58}")
    print(f"📍 Final node count: {len(nodes)}")
    print(f"{'─'*58}")
    for n in nodes:
        bbox = n["bbox"]
        size = f"{bbox['x2']-bbox['x1']}x{bbox['y2']-bbox['y1']}"
        print(f"  {n['id']:>4s}  {n['type']:<14s} {n['label']:<28s} [{size}]")

    # ── Save JSON ──
    graph = {
        "metadata": {
            "image": IMAGE_PATH,
            "width": img_w,
            "height": img_h,
            "scale": scale,
            "model": MODEL,
            "tile_config": f"{TILE_ROWS}x{TILE_COLS}",
        },
        "nodes": nodes,
    }

    graph_path = os.path.join(OUTPUT_DIR, "floorplan_nodes.json")
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"\n✅ Nodes saved: {graph_path}")

    # ── Visualize ──
    output_img = os.path.join(OUTPUT_DIR, "annotated_floorplan.png")
    draw_nodes(IMAGE_PATH, nodes, output_img, scale)

    # ── Cost ──
    tracker.report()

    print(f"\n🎉 Done! Check:")
    print(f"   • {graph_path}")
    print(f"   • {output_img}\n")


if __name__ == "__main__":
    main()