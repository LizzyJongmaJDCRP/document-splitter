import sys
import cv2
import numpy as np
from pathlib import Path

SCALE = 0.25
MIN_BLOB_AREA_FRAC = 0.002
MIN_PART_AREA_FRAC = 0.001
MIN_SIDE_FRAC = 0.12
MAX_ASPECT = 4.0
ERODE_SIZES = (3, 7, 11, 15, 21, 27, 35)
SEAM_MIN_FRAC = 0.35
SEAM_MARGIN_FRAC = 0.08
SEAM_GRAD_THRESH = 60
OVERSIZE_FACTOR = 1.2
MAX_DEPTH = 3


def build_mask(small):
    b, g, r = (small[:, :, i].astype(int) for i in range(3))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = (((r - b) < 15) & (gray < 170)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, gray


def rect_score(w, h):
    return max(w, h) / max(1, min(w, h))


def mask_area(m):
    return int((m > 0).sum())


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


def erode_split(blob_mask):
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
                restored.append(cv2.bitwise_and(comp, blob_mask))
            return restored
    return None


def find_seams(gcrop, region, axis):
    h, w = region.shape
    if axis == "v":
        grad = np.abs(cv2.Sobel(gcrop, cv2.CV_32F, 1, 0, ksize=3))
        klen = max(4, int(SEAM_MIN_FRAC * h))
        if klen > h:
            return []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
        base = h
    else:
        grad = np.abs(cv2.Sobel(gcrop, cv2.CV_32F, 0, 1, ksize=3))
        klen = max(4, int(SEAM_MIN_FRAC * w))
        if klen > w:
            return []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        base = w
    strong = ((grad > SEAM_GRAD_THRESH) & (region > 0)).astype(np.uint8) * 255
    opened = cv2.morphologyEx(strong, cv2.MORPH_OPEN, kernel)
    counts = opened.sum(axis=0 if axis == "v" else 1) / 255.0
    thresh = SEAM_MIN_FRAC * base
    idx = np.where(counts >= thresh)[0]
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1][0] <= 3:
            groups[-1].append((int(i), float(counts[i])))
        else:
            groups.append([(int(i), float(counts[i]))])
    return [max(g, key=lambda t: t[1]) for g in groups]


def seam_split(blob_mask, gray):
    ys, xs = np.where(blob_mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    region = blob_mask[y0:y1, x0:x1]
    gcrop = gray[y0:y1, x0:x1]
    h, w = region.shape
    blob_area = mask_area(blob_mask)
    best = None
    for axis in ("v", "h"):
        dim = w if axis == "v" else h
        other = h if axis == "v" else w
        margin = int(SEAM_MARGIN_FRAC * dim)
        for pos, sup in find_seams(gcrop, region, axis):
            if pos < margin or pos > dim - 1 - margin:
                continue
            a = region.copy()
            b = region.copy()
            if axis == "v":
                a[:, pos:] = 0
                b[:, :pos] = 0
            else:
                a[pos:, :] = 0
                b[:pos, :] = 0
            ok = True
            for p in (a, b):
                if mask_area(p) < 0.12 * blob_area:
                    ok = False
                    break
                plen = p.shape[0] if axis == "v" else p.shape[1]
                if plen < 0.6 * other:
                    ok = False
                    break
            if not ok:
                continue
            rel = sup / dim
            if best is None or rel > best[0]:
                best = (rel, axis, pos)
    if best is None:
        return None
    _, axis, pos = best
    left = region.copy()
    right = region.copy()
    if axis == "v":
        left[:, pos:] = 0
        right[:, :pos] = 0
    else:
        left[pos:, :] = 0
        right[:pos, :] = 0
    parts = []
    for p in (left, right):
        full = np.zeros_like(blob_mask)
        full[y0:y1, x0:x1] = p
        parts.append(full)
    return parts


def try_split(part_mask, gray):
    sub = erode_split(part_mask)
    if sub:
        return sub
    return seam_split(part_mask, gray)


def split_oversized(parts, gray, unbreakable, gate=True):
    """Split herhaaldelijk delen die duidelijk groter zijn dan de rest."""
    for _ in range(MAX_DEPTH * max(1, len(parts)) + 2):
        areas = [mask_area(p) for p in parts]
        med = float(np.median(areas))
        cand = [i for i, p in enumerate(parts)
                if i not in unbreakable
                and (not gate or mask_area(p) > OVERSIZE_FACTOR * med)]
        if not cand:
            return
        i = max(cand, key=lambda j: areas[j])
        sub = try_split(parts[i], gray)
        if sub and len(sub) >= 2:
            parts[i:i + 1] = sub
        else:
            unbreakable.add(i)


def find_document_parts(mask, gray, expected=None):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_AREA_FRAC * mask.size:
            continue
        blobs.append((labels == i).astype(np.uint8) * 255)

    parts = []
    unsplit = []
    for blob in blobs:
        eroded_parts = erode_split(blob)
        if eroded_parts:
            parts.extend(eroded_parts)
        else:
            unsplit.append(blob)

    areas = [mask_area(p) for p in parts] + [mask_area(b) for b in unsplit]
    med = float(np.median(areas)) if areas else 0.0
    unbreakable = set()
    for blob in unsplit:
        if len(blobs) == 1 or mask_area(blob) > OVERSIZE_FACTOR * med:
            sub = try_split(blob, gray)
            parts.extend(sub if sub else [blob])
        else:
            parts.append(blob)

    split_oversized(parts, gray, unbreakable, gate=True)

    if expected is not None and len(parts) < expected:
        split_oversized(parts, gray, unbreakable, gate=False)
    return parts


def find_document_boxes(mask, gray, expected=None):
    parts = find_document_parts(mask, gray, expected)
    boxes = []
    for pm in parts:
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


def split_image_boxes(img, expected=None):
    small = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    mask, gray = build_mask(small)
    regions = find_document_boxes(mask, gray, expected)

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
    return boxes, small


def output_stem(path):
    stem = path.stem
    if "-" in stem:
        stem = stem.split("-", 1)[1]
    return stem


def split_image(path: Path, output_dir: Path, debug_dir: Path, expected=None):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  kon {path.name} niet lezen")
        return
    boxes, small = split_image_boxes(img, expected)
    print(f"  {len(boxes)} documenten gevonden")
    if expected is not None and len(boxes) < expected:
        print(f"  let op: verwacht {expected}, geen verdere splitsing mogelijk")

    stem = output_stem(path)
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
    verwacht = input(
        "Verwacht aantal documenten per scan (Enter = automatisch): ").strip()
    expected = int(verwacht) if verwacht.isdigit() else None
    debug_dir = output_dir / "_debug"
    debug_dir.mkdir(exist_ok=True)

    scans = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    if not scans:
        print(f"Geen .jpg/.jpeg-bestanden gevonden in {input_dir}")
        return
    for f in scans:
        print(f.name)
        split_image(f, output_dir, debug_dir, expected)
    print(f"\nKlaar. Uitsneden staan in {output_dir}")


if __name__ == "__main__":
    main()
