"""Bird detection (YOLO11, tiled) + species ID (BioCLIP 2, zero-shot over a regional list)."""
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms
from ultralytics import YOLO

HERE = Path(__file__).parent
DATA = HERE / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BIRD_CLASS = 14  # COCO
# BioCLIP second opinion on each detector box: is this a bird at all?
BIRD_PROMPTS = ["a photo of a bird.", "a photo of Animalia Chordata Aves."]
NOT_BIRD_PROMPTS = ["a photo of a plant.", "a photo of grass seed heads.", "a photo of a leaf.", "a photo of a tree branch.",
                    "a photo of reeds.", "a photo of a rock.", "a photo of mud.", "a photo of water ripples.",
                    "a photo of the sky.", "a photo of a building.", "a photo of Plantae."]
Image.MAX_IMAGE_PIXELS = None


# ---------- metadata ----------

def exif_time(img: Image.Image):
    raw = img.getexif().get_ifd(0x8769).get(36867) or img.getexif().get(306)
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def exif_gps(img: Image.Image):
    gps = img.getexif().get_ifd(0x8825)
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def deg(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W") else d

    return deg(gps[2], gps.get(1, "N")), deg(gps[4], gps.get(3, "E"))


def load(path):
    img = ImageOps.exif_transpose(Image.open(path))
    return img.convert("RGB"), exif_time(img), exif_gps(img)


# ---------- regional species list ----------

def taxonomy():
    path = DATA / "ebird_taxonomy.csv"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        r = httpx.get("https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&cat=species", timeout=60)
        path.write_bytes(r.content)
    with open(path) as f:
        return {row["SCIENTIFIC_NAME"]: row for row in csv.DictReader(f)}


def regional_species(lat, lng, month=None, radius_km=30):
    """Birds iNaturalist has research-grade records of near (lat, lng), with counts as a prior."""
    key = f"inat_{lat:.2f}_{lng:.2f}_{radius_km}_{month}.json"
    cache = DATA / key
    if cache.exists():
        return json.loads(cache.read_text())
    params = dict(lat=lat, lng=lng, radius=radius_km, iconic_taxa="Aves",
                  quality_grade="research", per_page=500)
    if month:
        params["month"] = ",".join(str((month + d - 1) % 12 + 1) for d in (-1, 0, 1))
    r = httpx.get("https://api.inaturalist.org/v1/observations/species_counts", params=params, timeout=60)
    tax = taxonomy()
    out = []
    for res in r.json()["results"]:
        row = tax.get(res["taxon"]["name"])
        if row:
            out.append(dict(sci=row["SCIENTIFIC_NAME"], common=row["COMMON_NAME"],
                            code=row["SPECIES_CODE"], order=float(row["TAXON_ORDER"]),
                            count=res["count"]))
    cache.write_text(json.dumps(out))
    return out


# ---------- models ----------

class Models:
    def __init__(self):
        self.yolo = YOLO(str(HERE / "yolo11l.pt"))
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        self.clip = self.clip.to(DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
        self._text = {}
        with torch.no_grad():
            t = self.tokenizer(BIRD_PROMPTS + NOT_BIRD_PROMPTS).to(DEVICE)
            self._birdness = torch.nn.functional.normalize(self.clip.encode_text(t), dim=-1)

    # Tiled detection so small, distant birds in 40MP frames are still found.
    def detect(self, img: Image.Image, conf=0.15, work_size=3840, tile=1280, overlap=0.25):
        W, H = img.size
        scale = min(1.0, work_size / max(W, H))
        small = img.resize((round(W * scale), round(H * scale))) if scale < 1 else img
        sw, sh = small.size
        boxes, scores = [], []

        def run(im, ox, oy, inner=()):
            res = self.yolo.predict(im, imgsz=tile, conf=conf, classes=[BIRD_CLASS], device=DEVICE, verbose=False)[0]
            iw, ih = im.size
            for b, s in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                # a box cut by a tile edge that isn't the photo edge is a fragment; a neighbouring tile has it whole
                if ("l" in inner and b[0] < 3) or ("t" in inner and b[1] < 3) or \
                        ("r" in inner and b[2] > iw - 3) or ("b" in inner and b[3] > ih - 3):
                    continue
                boxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy])
                scores.append(s)

        # whole frame (big birds) …
        full = small.resize((tile, round(tile * sh / sw))) if sw >= sh else small.resize((round(tile * sw / sh), tile))
        fx = sw / full.size[0]
        n0 = len(boxes)
        run(full, 0, 0)
        for i in range(n0, len(boxes)):
            boxes[i] = [v * fx for v in boxes[i]]
        # … plus overlapping tiles (small birds)
        step = int(tile * (1 - overlap))
        for y in range(0, max(sh - tile, 0) + step, step):
            for x in range(0, max(sw - tile, 0) + step, step):
                x0, y0 = min(x, max(sw - tile, 0)), min(y, max(sh - tile, 0))
                inner = {e for e, cut in (("l", x0 > 0), ("t", y0 > 0), ("r", x0 + tile < sw), ("b", y0 + tile < sh)) if cut}
                run(small.crop((x0, y0, x0 + tile, y0 + tile)), x0, y0, inner)
        if not boxes:
            return []
        b, s = torch.tensor(boxes), torch.tensor(scores)
        keep = nms(b, s, 0.4)
        out = []
        for i in keep.tolist():
            x1, y1, x2, y2 = (b[i] / scale).tolist()
            # drop boxes mostly contained in a bigger kept box (tile-edge fragments)
            out.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, det=float(s[i])))
        out = _drop_contained(out)
        return [dict(x=o["x1"] / W, y=o["y1"] / H, w=(o["x2"] - o["x1"]) / W, h=(o["y2"] - o["y1"]) / H, det=o["det"],
                     px=round(max(o["x2"] - o["x1"], o["y2"] - o["y1"])))
                for o in out]

    @staticmethod
    def mark_reflections(img, dets):
        """Flag a box as a reflection when, flipped upside down, its pixels match the box above better than unflipped."""
        W, H = img.size

        def patch(d, flip=False):
            c = img.crop((d["x"] * W, d["y"] * H, (d["x"] + d["w"]) * W, (d["y"] + d["h"]) * H)).convert("L").resize((48, 48))
            a = np.asarray(c.transpose(Image.FLIP_TOP_BOTTOM) if flip else c, dtype=float)
            return (a - a.mean()) / (a.std() + 1e-6)

        for i, j in _mirror_candidates(dets):
            upper = patch(dets[i])
            mirrored, as_is = float((patch(dets[j], True) * upper).mean()), float((patch(dets[j]) * upper).mean())
            if mirrored > 0.3 and mirrored > as_is + 0.05:
                dets[j]["reflection"] = True

    def frame_embedding(self, img: Image.Image):
        """Whole-photo BioCLIP embedding, used to tell whether two photos show the same view."""
        with torch.no_grad():
            f = self.clip.encode_image(self.preprocess(img).unsqueeze(0).to(DEVICE))
            return torch.nn.functional.normalize(f, dim=-1)[0].cpu().numpy()

    def text_features(self, species):
        key = tuple(sp["code"] for sp in species)
        if key not in self._text:
            prompts = [f"a photo of {sp['sci']} with common name {sp['common']}." for sp in species]
            feats = []
            with torch.no_grad():
                for i in range(0, len(prompts), 128):
                    t = self.tokenizer(prompts[i:i + 128]).to(DEVICE)
                    feats.append(torch.nn.functional.normalize(self.clip.encode_text(t), dim=-1))
            self._text[key] = torch.cat(feats)
        return self._text[key]

    def classify(self, img: Image.Image, dets, species, prior_weight=0.3, topk=3):
        if not dets:
            return dets
        W, H = img.size
        crops = []
        for d in dets:
            cx, cy = (d["x"] + d["w"] / 2) * W, (d["y"] + d["h"] / 2) * H
            side = max(d["w"] * W, d["h"] * H) * 1.25
            crops.append(self.preprocess(img.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))))
        text = self.text_features(species)
        total = sum(sp["count"] for sp in species)
        log_prior = torch.tensor([math.log(sp["count"] / total) for sp in species], device=DEVICE)
        with torch.no_grad():
            feats = []
            for i in range(0, len(crops), 32):
                f = self.clip.encode_image(torch.stack(crops[i:i + 32]).to(DEVICE))
                feats.append(torch.nn.functional.normalize(f, dim=-1))
            feats = torch.cat(feats)
            logits = self.clip.logit_scale.exp() * feats @ text.T
            probs = (logits + prior_weight * log_prior).softmax(-1)
            bird = (self.clip.logit_scale.exp() * feats @ self._birdness.T).softmax(-1)[:, :len(BIRD_PROMPTS)].sum(-1)
        top = probs.topk(min(topk, len(species)), dim=-1)
        for d, b in zip(dets, bird.tolist()):
            d["bird"] = round(b, 3)
        self.mark_reflections(img, dets)
        for d, ps, ix in zip(dets, top.values.tolist(), top.indices.tolist()):
            d["guesses"] = [dict(code=species[i]["code"], common=species[i]["common"], p=round(p, 3))
                            for p, i in zip(ps, ix)]
            d["code"] = d["guesses"][0]["code"]
        return dets


def stop_breaks(times, embs, burst_s=5, chain_s=120, min_sim=0.82):
    """For photos in order, whether each one starts a new stop. Bursts always stay together; a photo taken soon after
    the last one that shows a similar view continues the stop. Anything else, or missing data, starts a new one."""
    out = [True]
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        dt = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds() if t0 and t1 else None
        sim = float(np.dot(embs[i - 1], embs[i])) if embs[i - 1] is not None and embs[i] is not None else 0
        out.append(not (dt is not None and (0 <= dt <= burst_s or (0 <= dt <= chain_s and sim >= min_sim))))
    return out


def _mirror_candidates(dets):
    """(upper, lower) index pairs where lower sits just under upper, aligned and similarly sized, as a reflection would."""
    pairs = []
    for i, a in enumerate(dets):
        for j, b in enumerate(dets):
            if i == j or b["y"] <= a["y"]:
                continue
            overlap = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
            gap = b["y"] - (a["y"] + a["h"])
            if (overlap > 0.6 * min(a["w"], b["w"]) and 0.6 < b["w"] / a["w"] < 1.6 and 0.4 < b["h"] / a["h"] < 1.6
                    and -0.5 * a["h"] < gap < 0.6 * a["h"]):
                pairs.append((i, j))
    return pairs


def _drop_contained(boxes, thresh=0.4):
    boxes = sorted(boxes, key=lambda o: -(o["x2"] - o["x1"]) * (o["y2"] - o["y1"]))
    kept = []
    for o in boxes:
        area = (o["x2"] - o["x1"]) * (o["y2"] - o["y1"])
        if any(_inter(o, k) / area > thresh for k in kept):
            continue
        kept.append(o)
    return kept


def _inter(a, b):
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    return max(w, 0) * max(h, 0)
