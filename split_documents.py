import cv2
import numpy as np
from pathlib import Path

SCALE = 0.25
MIN_BLOB_AREA_FRAC = 0.002
MIN_PART_AREA_FRAC = 0.001
MIN_SIDE_FRAC = 0.12
MAX_ASPECT = 4.0
ERODE_SIZES = (3, 7, 11, 15, 21, 27, 35)


def build_mask(small):
    b, g, r = (small[:, :, i].astype(int) for i in range(3))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = (((r - b) < 15) & (gray < 170)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def rect_score(w, h):
    return max(w, h) / max(1, min(w, h))


def background_fraction(small, x, y, w, h, side, strip=4):
    sh, sw = small.shape[:2]
    b, g, r = (small[:, :, i].astype(int) for i in range(3))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bgish = (((r - b) > 25) & (gray > 150)).astype(np.uint8)
    if side == "links":
        return bgish[y:y + h, x:x + strip].mean()
    if side == "rechts":
        return bgish[y:y + h, x + w - strip:x + w].mean()
    if side == "boven":
        return bgish[y:y + strip, x:x + w].mean()
    return bgish[y + h - strip:y + h, x:x + w].mean()


def refine_box(small, x, y, w, h):
    """Vergroot de box licht en knip daarna beige achtergrond aan elke kant weg."""
    sh, sw = small.shape[:2]
    mx = max(2, int(0.02 * w))
    my = max(2, int(0.02 * h))
    x = max(0, x - mx); y = max(0, y - my)
    w = min(sw, x + w + 2 * mx) - x; h = min(sh, y + h + 2 * my) - y
    strip = 4
    for _ in range(20):
        changed = False
        while w > 2 * strip and background_fraction(small, x, y, w, h, "links", strip) > 0.2:
            x += strip // 2; w -= strip // 2; changed = True
        while w > 2 * strip and background_fraction(small, x, y, w, h, "rechts", strip) > 0.2:
            w -= strip // 2; changed = True
        while h > 2 * strip and background_fraction(small, x, y, w, h, "boven", strip) > 0.2:
            y += strip // 2; h -= strip // 2; changed = True
        while h > 2 * strip and background_fraction(small, x, y, w, h, "onder", strip) > 0.2:
            h -= strip // 2; changed = True
        if not changed:
            break
    return x, y, w, h


def split_blob(blob_mask):
    """Splits een blob via erosie; herstel ieder deel door terug-dilateren."""
    area_min = MIN_PART_AREA_FRAC * blob_mask.size
    for k in ERODE_SIZES:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        eroded = cv2.erode(blob_mask, kernel)
        n, lab, st, _ = cv2.connectedComponentsWithStats(eroded, 8)
        parts = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= area_min]
        if len(parts) >= 2:
            restored = []
            for p in parts:
                comp = (lab == p).astype(np.uint8) * 255
                comp = cv2.dilate(comp, kernel)
                comp = cv2.bitwise_and(comp, blob_mask)
                restored.append(comp)
            return restored
    return [blob_mask]


def find_document_boxes(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    part_masks = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_AREA_FRAC * mask.size:
            continue
        blob = (labels == i).astype(np.uint8) * 255
        part_masks.extend(split_blob(blob))

    boxes = []
    for pm in part_masks:
        ys, xs = np.where(pm > 0)
        x0, y0 = int(xs.min()), int(ys.min())
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        area = len(xs)
        if area < MIN_BLOB_AREA_FRAC * mask.size:
            continue
        if area / (w * h) < 0.5:
            continue
        boxes.append((x0, y0, w, h))
    return boxes


def split_image(path: Path, output_dir: Path, debug_dir: Path):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  kon {path.name} niet lezen")
        return

    small = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    mask = build_mask(small)
    regions = find_document_boxes(mask)

    inv = 1.0 / SCALE
    h_img, w_img = img.shape[:2]
    boxes = []
    for x, y, w, h in regions:
        rx, ry, rw, rh = refine_box(small, x, y, w, h)
        bw, bh = int(rw * inv), int(rh * inv)
        if bw < MIN_SIDE_FRAC * w_img and bh < MIN_SIDE_FRAC * h_img:
            continue
        if rect_score(bw, bh) > MAX_ASPECT:
            continue
        boxes.append((int(rx * inv), int(ry * inv), bw, bh))
    boxes.sort(key=lambda bx: (bx[1] // 400, bx[0]))
    print(f"  {len(boxes)} documenten gevonden")

    stem = path.stem
    if "-" in stem:
        stem = stem.split("-", 1)[1]
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        crop = img[y:y + h, x:x + w]
        out = output_dir / f"{stem}_doc{idx:02d}.jpg"
        cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"    {out.name}  ({w}x{h} @ {x},{y})")

    vis = small.copy()
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        sx, sy, sw, sh = [int(v * SCALE) for v in (x, y, w, h)]
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)
        cv2.putText(vis, str(idx), (sx + 8, sy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.imwrite(str(debug_dir / f"{stem}_boxes.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 80])


def ask_input_dir() -> Path:
    while True:
        antwoord = input("Map met te splitsen scans (input_dir): ").strip().strip('"').strip("'")
        pad = Path(antwoord).expanduser()
        if pad.is_dir():
            return pad
        print(f"  '{pad}' bestaat niet of is geen map, probeer opnieuw.")


def ask_output_dir() -> Path:
    while True:
        antwoord = input("Map voor de uitsneden (output_dir): ").strip().strip('"').strip("'")
        pad = Path(antwoord).expanduser()
        try:
            pad.mkdir(parents=True, exist_ok=True)
            return pad
        except OSError as e:
            print(f"  '{pad}' kon niet worden aangemaakt ({e}), probeer opnieuw.")


def main():
    input_dir = ask_input_dir()
    output_dir = ask_output_dir()
    debug_dir = output_dir / "_debug"
    debug_dir.mkdir(exist_ok=True)

    scans = sorted(input_dir.glob("*.jpg"))
    if not scans:
        print(f"Geen .jpg-bestanden gevonden in {input_dir}")
        return
    for f in scans:
        print(f.name)
        split_image(f, output_dir, debug_dir)
    print(f"\nKlaar. Uitsneden staan in {output_dir}")


if __name__ == "__main__":
    main()
