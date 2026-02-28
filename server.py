#!/usr/bin/env python3
"""
splitit — local receipt splitter (Gemini)
- Serves index.html
- /parse      : single image/pdf
- /parse_multi: multiple images

Setup:
  pip install -r requirements.txt

Run:
  Mac/Linux:  GEMINI_API_KEY=your-key python3 server.py
  Windows:    set GEMINI_API_KEY=your-key && python server.py

Open:
  http://localhost:8765
"""

import os
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from google import genai
    from google.genai import types
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
PORT = 8765
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# guard against giant payloads
MAX_REQUEST_BYTES = 16 * 1024 * 1024  # 16MB

PROMPT = (
    "You are a receipt parser. Look at this receipt and extract every purchased item.\n"
    "Return ONLY a JSON object matching the provided schema.\n\n"
    "Rules:\n"
    "- Human-readable item names (Green Grapes not GRN GRPS)\n"
    "- price = final price paid for that item line (after per-item discounts)\n"
    "- discount = total savings/instant savings shown on receipt (informational)\n"
    "- Do NOT include subtotal/total/tax/tip rows as items\n"
    "- Round all numbers to 2 decimals\n"
    "- Use 0 for unknown numeric values\n"
)

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "store": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                },
                "required": ["name", "price"],
                "additionalProperties": False,
            },
        },
        "tax": {"type": "number"},
        "discount": {"type": "number"},
        "tip": {"type": "number"},
        "total": {"type": "number"},
    },
    "required": ["store", "items", "tax", "discount", "tip", "total"],
    "additionalProperties": False,
}


def clamp_money(x):
    try:
        v = float(x)
        if v != v or v < 0:
            return 0.0
        return round(v + 1e-9, 2)
    except Exception:
        return 0.0


def normalize_receipt(d):
    out = {
        "store": str(d.get("store") or "").strip() or "Receipt",
        "items": [],
        "tax": clamp_money(d.get("tax", 0)),
        "discount": clamp_money(d.get("discount", 0)),
        "tip": clamp_money(d.get("tip", 0)),
        "total": clamp_money(d.get("total", 0)),
    }

    items = d.get("items") or []
    if not isinstance(items, list):
        items = []

    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        price = clamp_money(it.get("price", 0))
        if not name:
            continue
        upper = name.upper()
        if any(k in upper for k in ["SUBTOTAL", "TOTAL", "TAX", "TIP", "CHANGE", "BALANCE DUE"]):
            continue
        out["items"].append({"name": name, "price": price})

    return out


def parse_with_gemini(img_bytes: bytes, media_type: str):
    client = genai.Client(api_key=API_KEY)
    part = types.Part.from_bytes(data=img_bytes, mime_type=media_type)

    resp = client.models.generate_content(
        model=MODEL,
        contents=[part, PROMPT],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": RECEIPT_SCHEMA,
            "temperature": 0.2,
        },
    )

    txt = (resp.text or "").strip()
    if not txt:
        raise RuntimeError("Empty response from model")

    parsed = json.loads(txt)
    return normalize_receipt(parsed)


def merge_receipts(receipts):
    """
    Merge multiple parsed receipts (multi-image).
    Strategy:
      - store: first non-empty
      - items: concatenate
      - tax/discount/tip/total: take LAST non-zero (often last page has totals)
    """
    merged = {"store": "Receipt", "items": [], "tax": 0.0, "discount": 0.0, "tip": 0.0, "total": 0.0}
    for r in receipts:
        if r.get("store") and merged["store"] == "Receipt":
            merged["store"] = r["store"]
        merged["items"].extend(r.get("items", []))

        # last non-zero wins
        for k in ("tax", "discount", "tip", "total"):
            v = clamp_money(r.get(k, 0))
            if v > 0:
                merged[k] = v

    # normalize numbers again
    merged["tax"] = clamp_money(merged["tax"])
    merged["discount"] = clamp_money(merged["discount"])
    merged["tip"] = clamp_money(merged["tip"])
    merged["total"] = clamp_money(merged["total"])
    return merged


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {fmt % args}")

    def send_json(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, mime):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            self.send_json(404, {"error": f"Missing file: {path}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_file("index.html", "text/html; charset=utf-8")
        elif self.path == "/health":
            self.send_json(200, {"ok": True, "has_sdk": HAS_SDK, "has_key": bool(API_KEY), "model": MODEL})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/parse", "/parse_multi"):
            self.send_json(404, {"error": "not found"})
            return

        if not HAS_SDK:
            self.send_json(400, {"error": "google-genai not installed. Run: pip install -r requirements.txt"})
            return
        if not API_KEY:
            self.send_json(400, {"error": "GEMINI_API_KEY not set. Get free key: https://aistudio.google.com/apikey"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0

        if length <= 0:
            self.send_json(400, {"error": "Empty request body"})
            return
        if length > MAX_REQUEST_BYTES:
            self.send_json(413, {"error": f"Request too large (> {MAX_REQUEST_BYTES} bytes). Use smaller images."})
            return

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_json(400, {"error": "Invalid JSON body"})
            return

        try:
            if self.path == "/parse":
                result = self.handle_parse(body)
            else:
                result = self.handle_parse_multi(body)

            self.send_json(200, {"ok": True, "data": result})
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if "429" in msg or "quota" in low or "resource_exhausted" in low:
                self.send_json(400, {
                    "error": (
                        "Quota exceeded.\n\n"
                        "Fix:\n"
                        "1) Go to https://aistudio.google.com/apikey\n"
                        "2) Create API key in a NEW project\n"
                        "3) Restart with the new key"
                    )
                })
            else:
                self.send_json(400, {"error": msg})

    def handle_parse(self, body):
        image_b64 = body.get("image_b64", "")
        media_type = body.get("media_type", "image/jpeg") or "image/jpeg"
        if not isinstance(image_b64, str) or not image_b64:
            raise ValueError("Missing image_b64")
        img_bytes = base64.b64decode(image_b64, validate=True)
        return parse_with_gemini(img_bytes, media_type)

    def handle_parse_multi(self, body):
        images = body.get("images", [])
        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("Missing images[]")

        receipts = []
        for idx, item in enumerate(images):
            if not isinstance(item, dict):
                continue
            image_b64 = item.get("image_b64", "")
            media_type = item.get("media_type", "image/jpeg") or "image/jpeg"
            if not image_b64:
                continue
            img_bytes = base64.b64decode(image_b64, validate=True)
            receipts.append(parse_with_gemini(img_bytes, media_type))

        if not receipts:
            raise ValueError("No valid images in images[]")

        return merge_receipts(receipts)


if __name__ == "__main__":
    print("\n" + "─" * 56)
    print("  splitit — local receipt splitter (Gemini)")
    print("─" * 56)

    if not HAS_SDK:
        print("\n  ⚠️  google-genai not installed!")
        print("  Run:  pip install -r requirements.txt\n")
    elif not API_KEY:
        print("\n  ⚠️  No API key!")
        print("  Get free key: https://aistudio.google.com/apikey")
        print("  Mac/Linux:  GEMINI_API_KEY=your-key python3 server.py")
        print("  Windows:    set GEMINI_API_KEY=your-key && python server.py\n")
    else:
        print(f"\n  ✓ SDK: google-genai")
        print(f"  ✓ Key: {API_KEY[:8]}...")
        print(f"  ✓ Model: {MODEL}\n")

    print(f"  🚀 http://localhost:{PORT}")
    print("  Ctrl+C to stop\n")
    HTTPServer(("localhost", PORT), Handler).serve_forever()