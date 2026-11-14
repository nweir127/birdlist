"""Birdlist POC: folder of bird photos -> reviewed eBird checklist + media sorted for upload."""
import csv
import io
import json
import re
import shutil
import threading
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

import pipeline as P

HERE = Path(__file__).parent
OUTINGS = HERE / "outings"
CAMERAS = P.DATA / "cameras.json"
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}
OUTINGS.mkdir(exist_ok=True)

app = FastAPI()
models = None
models_ready = threading.Event()
lock = threading.Lock()
TAX = P.taxonomy()
TAX_BY_CODE = {r["SPECIES_CODE"]: r for r in TAX.values()}
# How counts may combine across stops (see the count rules in index.html):
# 'once': big or mobile birds that patrol a site, likely the same bird seen again; never summed unless the user says so.
# 'territory': small birds holding fixed territories along a path; summing separate stops is reasonable.
# everything else ('flock') can be summed only when the user confirms the flocks were separate.
ONCE_FAMILIES = {"Accipitridae", "Falconidae", "Pandionidae", "Cathartidae", "Ardeidae", "Laridae", "Pelecanidae",
                 "Phalacrocoracidae", "Threskiornithidae", "Sulidae", "Gaviidae", "Corvidae", "Hirundinidae",
                 "Apodidae", "Tytonidae", "Strigidae", "Ciconiidae", "Gruidae"}
TERRITORY_FAMILIES = {"Passerellidae", "Troglodytidae", "Parulidae", "Tyrannidae", "Mimidae", "Regulidae",
                      "Cisticolidae", "Paridae", "Aegithalidae", "Sittidae", "Certhiidae", "Polioptilidae",
                      "Picidae", "Trochilidae", "Paradoxornithidae", "Cardinalidae", "Vireonidae", "Turdidae"}


def _load_models():
    global models
    models = P.Models()
    models_ready.set()


threading.Thread(target=_load_models, daemon=True).start()


# ---------- persistence ----------

def odir(oid):
    d = OUTINGS / oid
    if not d.exists():
        raise HTTPException(404)
    return d


def read_state(oid):
    return json.loads((odir(oid) / "state.json").read_text())


def write_state(oid, state):
    with lock:
        tmp = odir(oid) / "state.json.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(odir(oid) / "state.json")


def cameras():
    return json.loads(CAMERAS.read_text()) if CAMERAS.exists() else {}


# ---------- location ----------

def past_place(q):
    """A location from an earlier outing whose name starts the same way, e.g. "Bedwell" -> Bedwell Bayfront Park.
    Earlier outings' locations have usually been checked, which a geocoder lookup of one word hasn't."""
    word = q.split()[0].lower() if q.split() else ""
    for d in sorted(OUTINGS.iterdir(), reverse=True):
        f = d / "state.json"
        if not word or not f.exists():
            continue
        s = json.loads(f.read_text())
        m = s["meta"]
        if m.get("lat") is not None and word in (s["name"].split()[0].lower(), m["location"].split()[0].lower()):
            return dict(name=m["location"], lat=m["lat"], lng=m["lng"], state=m["state"], country=m["country"])
    return None


def guess_place(name):
    q = re.sub(r"\b\d{1,2}[-_./]\d{1,2}[-_./]\d{2,4}\b", "", name).strip(" -_")
    if hit := past_place(q):
        return hit
    try:
        r = httpx.get("https://nominatim.openstreetmap.org/search",
                      params=dict(q=q, format="json", limit=1, addressdetails=1),
                      headers={"User-Agent": "birdlist-poc"}, timeout=15)
        hit = r.json()[0]
    except Exception:
        return dict(name=q, lat=None, lng=None, state="", country="")
    iso = hit.get("address", {}).get("ISO3166-2-lvl4", "")
    return dict(name=q, lat=float(hit["lat"]), lng=float(hit["lon"]),
                state=iso.split("-")[-1] if iso else "",
                country=hit.get("address", {}).get("country_code", "").upper())


# ---------- processing ----------

def process(oid):
    d = odir(oid)
    state = read_state(oid)
    models_ready.wait()
    meta = state["meta"]
    month = None
    for ph in state["photos"]:
        if ph.get("time"):
            month = int(ph["time"][5:7])
            break
    if meta["lat"] is None:
        meta["lat"], meta["lng"] = 37.77, -122.42  # fallback so we still get a list
    species = P.regional_species(meta["lat"], meta["lng"], month=month)
    state["species"] = {s["code"]: s for s in species}
    state["species_at"] = [meta["lat"], meta["lng"]]
    write_state(oid, state)
    frames = [None] * len(state["photos"])
    for i, ph in enumerate(state["photos"]):
        src = Path(ph["src"])
        try:
            img, _, _ = P.load(src)
            prev = img.copy()
            prev.thumbnail((2400, 2400))
            prev.save(d / "preview" / f"{ph['id']}.jpg", quality=85)
            prev.thumbnail((320, 320))
            prev.save(d / "thumb" / f"{ph['id']}.jpg", quality=75)
            frames[i] = models.frame_embedding(prev)
            dets = models.classify(img, models.detect(img), species)
            for k, det in enumerate(dets):
                det["id"] = f"{ph['id']}-{k}"
                det["user"] = False
            ph["dets"] = dets
            ph["done"] = True
        except Exception as e:  # keep going; show the error on that photo
            ph["error"] = str(e)
            ph["done"] = True
        cur = read_state(oid)  # merge: user may have edited earlier photos meanwhile
        cur["photos"][i].update({k: ph[k] for k in ("dets", "done", "error") if k in ph})
        if i == len(state["photos"]) - 1:
            cur["processing"] = False
            cur["auto_breaks"] = P.stop_breaks([p["time"] for p in cur["photos"]], frames)
        write_state(oid, cur)


def load_photo(oid, ph):
    """The original if it's still where it was imported from, else the app's 2400px copy (fine for species ID)."""
    src = Path(ph["src"])
    return P.load(src if src.exists() else odir(oid) / "preview" / f"{ph['id']}.jpg")


def new_outing(name, files):
    oid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = OUTINGS / oid
    for sub in ("preview", "thumb"):
        (d / sub).mkdir(parents=True)
    photos, model = [], None
    for f in sorted(files, key=lambda p: p.name):
        img = Image.open(f)
        t = P.exif_time(img)
        gps = P.exif_gps(img)
        model = model or img.getexif().get(272)
        photos.append(dict(id=f"p{len(photos):04d}", name=f.name, src=str(f),
                           time=t.isoformat() if t else None, gps=gps, dets=[], done=False))
    place = guess_place(name)
    gps = next((p["gps"] for p in photos if p["gps"]), None)
    if gps:
        place["lat"], place["lng"] = gps
    times = [p["time"] for p in photos if p["time"]]
    cam = (model or "unknown").strip()
    offset = cameras().get(cam, 0)
    meta = dict(location=place["name"], lat=place["lat"], lng=place["lng"], state=place["state"],
                country=place["country"], camera=cam, clock_offset_min=offset,
                first=min(times) if times else None, last=max(times) if times else None,
                protocol="Traveling", observers=1, distance_mi="", complete=False, comments="")
    state = dict(id=oid, name=name, meta=meta, photos=photos, species={}, counts={},
                 extra=[], stars={}, breaks={}, summable={}, via={}, auto_breaks=None,
                 processing=True, created=time.time())
    (d / "state.json").write_text(json.dumps(state))
    threading.Thread(target=process, args=(oid,), daemon=True).start()
    return oid


# ---------- API ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text()


@app.get("/api/status")
def status():
    return dict(models=models_ready.is_set())


@app.get("/api/outings")
def list_outings():
    out = []
    for d in sorted(OUTINGS.iterdir(), reverse=True):
        if (d / "state.json").exists():
            s = json.loads((d / "state.json").read_text())
            out.append(dict(id=s["id"], name=s["name"], photos=len(s["photos"]),
                            done=sum(p["done"] for p in s["photos"])))
    return out


# ---------- Google Photos shared albums ----------

imports = {}  # job id -> progress, shown on the home page while originals download


def album_items(url):
    """(title, [(download url, filename hint)]) for a public Google Photos share link. Reads the item list the
    album page embeds; '=d' on a photo URL serves the original file with its EXIF."""
    page = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
    m = re.search(r"AF_initDataCallback\(\{key: 'ds:1'.*?data:(.*?), sideChannel", page, re.S)
    if not m:
        raise ValueError("That link didn't open a shared album. In Google Photos, use Share → Create link.")
    data = json.loads(m.group(1))
    items = [(it[1][0] + "=d", it[0]) for it in data[1]]
    if not items:
        raise ValueError("That album has no photos.")
    return (data[3][1] if len(data) > 3 else None) or "Google Photos album", items


def import_album(job, url):
    try:
        title, items = album_items(url)
        imports[job].update(total=len(items), name=title)
        dest = OUTINGS / "_uploads" / job
        dest.mkdir(parents=True)
        saved = []

        def fetch(item):
            src, key = item
            with httpx.stream("GET", src, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                fname = re.search(r'filename="([^"]+)"', r.headers.get("content-disposition", ""))
                target = dest / Path(fname.group(1) if fname else f"{key}.jpg").name
                if target.exists():  # two photos with one name: keep both
                    target = target.with_stem(f"{target.stem}-{key[-6:]}")
                with open(target, "wb") as out:
                    for chunk in r.iter_bytes(1 << 20):
                        out.write(chunk)
            imports[job]["done"] += 1
            return target

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as pool:
            saved = [t for t in pool.map(fetch, items) if t.suffix.lower() in EXTS]
        if not saved:
            raise ValueError("No photos in that album (videos are skipped).")
        imports[job]["id"] = new_outing(title, saved)
    except Exception as e:
        imports[job]["error"] = str(e)


@app.post("/api/outings/from-album")
def from_album(url: str = Form(...)):
    url = url.strip()
    if not re.match(r"https://(photos\.app\.goo\.gl|photos\.google\.com/share)/", url):
        raise HTTPException(400, "Paste a Google Photos shared album link (photos.app.goo.gl/…)")
    job = uuid.uuid4().hex[:8]
    imports[job] = dict(done=0, total=None, name=None, id=None, error=None)
    threading.Thread(target=import_album, args=(job, url), daemon=True).start()
    return dict(job=job)


@app.get("/api/imports/{job}")
def import_status(job: str):
    if job not in imports:
        raise HTTPException(404)
    return imports[job]


@app.post("/api/outings/from-path")
def from_path(path: str = Form(...)):
    p = Path(path.strip().strip("'\"")).expanduser()
    if not p.is_dir():
        raise HTTPException(400, f"Not a folder: {p}")
    files = [f for f in p.iterdir() if f.suffix.lower() in EXTS and not f.name.startswith(".")]
    if not files:
        raise HTTPException(400, "No photos in that folder")
    return dict(id=new_outing(p.name, files))


@app.post("/api/outings/upload")
async def upload(name: str = Form(...), files: list[UploadFile] = File(...)):
    dest = OUTINGS / "_uploads" / uuid.uuid4().hex[:8]
    dest.mkdir(parents=True)
    saved = []
    for uf in files:
        fname = Path(uf.filename).name
        target = dest / fname
        with open(target, "wb") as out:
            shutil.copyfileobj(uf.file, out)
        if target.suffix.lower() == ".zip":
            with zipfile.ZipFile(target) as z:
                for m in z.infolist():
                    mp = Path(m.filename)
                    if mp.suffix.lower() in EXTS and not mp.name.startswith(".") and "__MACOSX" not in mp.parts:
                        (dest / mp.name).write_bytes(z.read(m))
                        saved.append(dest / mp.name)
            target.unlink()
            if name.lower().endswith(".zip"):
                name = name[:-4]
        elif target.suffix.lower() in EXTS:
            saved.append(target)
    if not saved:
        raise HTTPException(400, "No photos found")
    return dict(id=new_outing(name, saved))


def name_info(code):
    r = TAX_BY_CODE.get(code)
    if not r:
        return None
    band = (r["BANDING_CODES"] or r["COM_NAME_CODES"]).split()
    fam = r["FAMILY_SCI_NAME"]
    return dict(common=r["COMMON_NAME"], sci=r["SCIENTIFIC_NAME"], band=band[0] if band else "",
                order=float(r["TAXON_ORDER"]), family=r["FAMILY_COM_NAME"],
                cls="once" if fam in ONCE_FAMILIES else "territory" if fam in TERRITORY_FAMILIES else "flock")


@app.get("/api/outings/{oid}")
def get_outing(oid: str):
    s = read_state(oid)
    codes = set(s["species"]) | set(s["extra"]) | set(s["counts"])
    for p in s["photos"]:
        for d in p["dets"]:
            codes.add(d.get("code"))
            codes.update(g["code"] for g in d.get("guesses", []))
    s["names"] = {c: name_info(c) for c in codes if c and name_info(c)}
    return s


@app.put("/api/outings/{oid}")
def put_outing(oid: str, body: dict):
    cur = read_state(oid)
    for k in ("meta", "counts", "extra", "stars", "breaks", "summable", "via", "final"):
        if k in body:
            cur[k] = body[k]
    if "photos" in body:  # only user-editable fields; never clobber processing results in flight
        by_id = {p["id"]: p for p in body["photos"]}
        for p in cur["photos"]:
            if p["id"] in by_id and p["done"]:
                p["dets"] = by_id[p["id"]]["dets"]
                p["flocks"] = by_id[p["id"]].get("flocks", [])
                p["ignored"] = by_id[p["id"]].get("ignored", False)
    if "meta" in body:
        cams = cameras()
        cams[cur["meta"]["camera"]] = cur["meta"]["clock_offset_min"]
        CAMERAS.write_text(json.dumps(cams))
    write_state(oid, cur)
    return dict(ok=True)


@app.get("/api/outings/{oid}/img/{kind}/{pid}.jpg")
def image(oid: str, kind: str, pid: str):
    if kind not in ("preview", "thumb") or not re.fullmatch(r"p\d{4}", pid):
        raise HTTPException(404)
    f = odir(oid) / kind / f"{pid}.jpg"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "max-age=86400"})


def relabel(oid):
    """Re-run species ID on every box for the outing's current location. Detection is kept; so is any box the
    user confirmed or corrected, apart from refreshed guesses."""
    models_ready.wait()
    state = read_state(oid)
    meta = state["meta"]
    month = next((int(p["time"][5:7]) for p in state["photos"] if p.get("time")), None)
    species = P.regional_species(meta["lat"], meta["lng"], month=month)
    state["species"] = {s["code"]: s for s in species}
    state["species_at"] = [meta["lat"], meta["lng"]]
    write_state(oid, state)
    todo = [ph for ph in state["photos"] if ph["dets"] and not ph.get("error")]
    try:
        _relabel_photos(oid, todo, species)
    finally:  # never leave the outing marked busy
        cur = read_state(oid)
        cur["relabel"] = None
        write_state(oid, cur)


def _relabel_photos(oid, todo, species):
    for i, ph in enumerate(todo):
        img, _, _ = load_photo(oid, ph)
        dets = models.classify(img, [dict(d) for d in ph["dets"]], species)
        fresh = {d["id"]: d for d in dets}
        cur = read_state(oid)  # merge into the latest state so edits made meanwhile survive
        cp = next(p for p in cur["photos"] if p["id"] == ph["id"])
        for d in cp["dets"]:
            f = fresh.get(d["id"])
            if f:
                d["guesses"] = f["guesses"]
                if not d.get("user"):
                    d["code"] = f["code"]
        cur["relabel"] = dict(done=i + 1, total=len(todo))
        write_state(oid, cur)


@app.post("/api/outings/{oid}/relabel")
def start_relabel(oid: str):
    s = read_state(oid)
    if s.get("processing") or s.get("relabel"):
        raise HTTPException(409, "This outing is still being processed")
    if s["meta"]["lat"] is None:
        raise HTTPException(400, "Set a latitude and longitude first")
    s["relabel"] = dict(done=0, total=sum(bool(p["dets"]) for p in s["photos"]))
    write_state(oid, s)
    threading.Thread(target=relabel, args=(oid,), daemon=True).start()
    return dict(ok=True)


@app.post("/api/outings/{oid}/classify")
def classify_box(oid: str, body: dict):
    models_ready.wait()
    s = read_state(oid)
    ph = next(p for p in s["photos"] if p["id"] == body["pid"])
    img, _, _ = load_photo(oid, ph)
    species = list(s["species"].values())
    box = body["box"]
    det = models.classify(img, [dict(box, det=1.0, px=round(max(box["w"] * img.width, box["h"] * img.height)))], species)[0]
    return dict(guesses=det["guesses"], names={g["code"]: name_info(g["code"]) for g in det["guesses"]})


@app.get("/api/species")
def species_search(q: str, oid: str | None = None, limit: int = 12):
    q = q.lower().strip()
    regional = read_state(oid)["species"] if oid else {}
    hits = []
    for r in TAX.values():
        com, sci, codes = r["COMMON_NAME"].lower(), r["SCIENTIFIC_NAME"].lower(), r["BANDING_CODES"].lower()
        if q in com or q in sci or (q and q == codes) or q == r["SPECIES_CODE"]:
            rank = (0 if r["SPECIES_CODE"] in regional else 1, 0 if com.startswith(q) or q == codes else 1, com)
            hits.append((rank, r))
    hits.sort(key=lambda h: h[0])
    return [dict(code=r["SPECIES_CODE"], local=r["SPECIES_CODE"] in regional, **name_info(r["SPECIES_CODE"]))
            for _, r in hits[:limit]]


# ---------- export ----------

def checklist(state, min_det):
    """Fallback for an export the page didn't prepare: max per photo of confident boxes, no flocks or stops."""
    per_photo = defaultdict(Counter)
    for ph in state["photos"]:
        for d in [] if ph.get("ignored") else ph["dets"]:
            if d.get("code") and (d["user"] or (d["det"] >= max(min_det, 0.4) and d["guesses"][0]["p"] >= 0.6
                                                and d.get("px", 999) >= 120 and d.get("bird", 1) >= 0.5
                                                and not d.get("reflection"))):
                per_photo[ph["id"]][d["code"]] += 1
    auto = Counter()
    for c in per_photo.values():
        for code, n in c.items():
            auto[code] = max(auto[code], n)
    for code in state["extra"]:
        auto.setdefault(code, 1)
    counts = {code: state["counts"].get(code, n) for code, n in auto.items()}
    return {c: n for c, n in counts.items() if n}


def export_counts(state, min_det):
    final = state.get("final") or {c: dict(n=n, comment="") for c, n in checklist(state, min_det).items()}
    return final, {c: f["n"] for c, f in final.items() if f["n"]}


def safe_name(state):
    return re.sub(r"[^\w -]", " ", state["name"]).strip() or "outing"


@app.get("/api/outings/{oid}/export.csv")
def export_csv(oid: str, min_det: float = 0.3):
    """The checklist alone, in eBird Record Format (Extended), for eBird's Import data page."""
    state = read_state(oid)
    m = state["meta"]
    if m["protocol"] == "Traveling" and not float(m.get("distance_mi") or 0) > 0:
        raise HTTPException(400, "eBird needs a distance for a Traveling checklist. Add it on the Send to eBird page.")
    final, counts = export_counts(state, min_det)
    start = datetime.fromisoformat(m["first"]) + timedelta(minutes=m["clock_offset_min"]) if m["first"] else None
    dur = round((datetime.fromisoformat(m["last"]) - datetime.fromisoformat(m["first"])).total_seconds() / 60) if m["first"] else ""
    rows = []
    for code, n in sorted(counts.items(), key=lambda kv: float(TAX_BY_CODE[kv[0]]["TAXON_ORDER"])):
        r = TAX_BY_CODE[code]
        genus, _, sp = r["SCIENTIFIC_NAME"].partition(" ")
        rows.append([r["COMMON_NAME"], genus, sp, n, "", m["location"], m["lat"] or "", m["lng"] or "",
                     start.strftime("%m/%d/%Y") if start else "", start.strftime("%H:%M") if start else "",
                     m["state"], m["country"], m["protocol"], m["observers"], m.get("duration", dur),
                     "Y" if m["complete"] else "N",
                     m["distance_mi"] if m["protocol"] == "Traveling" else "", "", m["comments"]])
    s = io.StringIO()
    csv.writer(s).writerows(rows)
    return Response(s.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{safe_name(state)} ebird.csv"'})


@app.get("/api/outings/{oid}/photos.zip")
def export_photos(oid: str, min_det: float = 0.3):
    """Photos to attach after importing, one folder per species. Kept out of the CSV download: eBird's importer
    takes only the checklist file."""
    state = read_state(oid)
    _, counts = export_counts(state, min_det)
    out = odir(oid) / "photos.zip"
    with zipfile.ZipFile(out, "w") as z:  # stored, not deflated: JPEGs don't compress
        src = {p["id"]: p["src"] for p in state["photos"]}
        best = {}  # fallback when the user hasn't starred any photo of a species
        for ph in state["photos"]:
            for d in [] if ph.get("ignored") else ph["dets"]:
                if d.get("code") in counts:
                    score = d["w"] * d["h"] * (1 if d["user"] else d["guesses"][0]["p"] if d.get("guesses") else 0)
                    if score > best.get(d["code"], (-1, None))[0]:
                        best[d["code"]] = (score, ph["id"])
        for code in counts:
            pids = state.get("final", {}).get(code, {}).get("photos") or state["stars"].get(code, [best[code][1]] if code in best else [])
            name = re.sub(r"[^\w' -]", "", TAX_BY_CODE[code]["COMMON_NAME"])
            for pid in pids:
                orig = Path(src[pid])  # originals moved since import: the 2400px copy still works for eBird
                z.write(orig if orig.exists() else odir(oid) / "preview" / f"{pid}.jpg", f"media/{name}/{orig.name}")
    return FileResponse(out, media_type="application/zip", filename=f"{safe_name(state)} photos.zip")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
