# Birdlist

Turn one birding outing's photos into an eBird checklist with photos to attach.

Drop a folder, a zip, or paste a shared Google Photos album link. Every bird is found and identified on your
machine, you review the IDs and counts, then download a CSV for eBird's importer and a zip of photos sorted by species.

- **Detection:** YOLO11 (COCO "bird"), tiled so small distant birds in large photos are found.
- **Species ID:** [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2), zero-shot over the birds iNaturalist has
  research-grade records of within 30 km of the outing's location that season, weighted by how common they are.
- **Cleanup:** a second BioCLIP check drops boxes that aren't birds, and reflections in water are flagged.
- **Counts:** a species' count is the most of it in any one photo. Flock estimates, stops (photos of the same view
  close in time) and an optional "add up separate stops" suggestion handle the cases that rule gets wrong.

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Uses Apple Silicon's GPU when available, else the CPU.

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
./run.sh            # opens http://localhost:8765
```

The first start downloads the model weights (about 1.8 GB) and the eBird taxonomy.

## Getting the checklist into eBird

1. On **Send to eBird**, download the checklist (.csv) and upload it at
   [eBird's Import data page](https://ebird.org/import/upload.form) as **eBird Record Format (Extended)**.
2. Download the photos (.zip) and drag each species' folder onto that species under **Edit species**.

Outings are stored in `outings/`, which is not part of the repo.
