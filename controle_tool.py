import base64
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import split_documents as sd

HTML_PATH = Path(__file__).resolve().parent / "controle_tool.html"


def split_bytes(data: bytes, filename: str, current_count: int):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "kon afbeelding niet lezen"

    boxes, _ = sd.split_image_boxes(img)
    note = f"{len(boxes)} documenten gevonden"
    if len(boxes) < current_count:
        boxes, _ = sd.split_image_boxes(img, expected=current_count + 1)
        note = f"{len(boxes)} documenten gevonden (geforceerde splitsing)"

    stem = sd.output_stem(Path(filename))
    crops = []
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        crop = img[y:y + h, x:x + w]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        crops.append({
            "name": f"{stem}_doc{idx:02d}.jpg",
            "dataUrl": f"data:image/jpeg;base64,{b64}",
        })
    return crops, note


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/api/ping":
            self._send_json({"server": True})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/split":
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        filename = self.headers.get("X-Filename", "scan.jpg")
        try:
            current = int(self.headers.get("X-Expected", "0"))
        except ValueError:
            current = 0
        try:
            crops, note = split_bytes(data, filename, current)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            return
        if crops is None:
            self._send_json({"error": note}, 400)
            return
        self._send_json({"crops": crops, "note": note})


def free_port(start=8765):
    port = start
    while port < start + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("geen vrije poort gevonden")


def main():
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Controle tool beschikbaar op http://127.0.0.1:{port}")
    print("(alleen lokaal bereikbaar; stoppen met Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
