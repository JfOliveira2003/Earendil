"""Merge RDD2022 and UAV-PDD2023 into a single YOLO pothole-detection dataset.

Both source datasets are read straight from their distribution zips; only the
files that end up in the merged dataset are extracted. The output is a
single-class ("Pothole") detection dataset:

    <OUT>/images/{train,val,test}/*.jpg
    <OUT>/labels/{train,val,test}/*.txt
    <OUT>/data.yml
    <OUT>/manifest.csv      one row per exported image
    <OUT>/stats.json        counts per source / split

Design decisions (see the TCC write-up):

* Class mapping. RDD2022 annotates potholes as ``D40``; UAV-PDD2023 as
  ``Pothole``. Every other damage class (cracks, repairs, ...) is dropped, so
  images that only contain those become *negatives* rather than being relabelled.
* Negatives. Images with no pothole are kept at ``NEG_RATIO`` per positive so the
  detector sees intact road surface and does not over-predict.
* Group-aware split. UAV-PDD2023 ships augmented copies of each source photo
  (``lr_`` / ``tb_`` / ``r180_`` prefixes = flips and rotation, ``_top_left`` ...
  suffixes = quadrant crops). All files derived from one source photo form a
  *group* and always land in the same split; otherwise near-duplicates straddle
  train and val and inflate the validation score. The official
  ``ImageSets/Main`` lists are NOT used because they leak: 184 of their 185
  validation groups also occur in training.
* Geometry. Box normalisation uses the true JPEG dimensions read from the image
  itself, not the ``<size>`` recorded in the XML, which is unreliable in RDD2022.

* Capture platform. ``--drone-only`` restricts the merge to nadir drone imagery:
  all of UAV-PDD2023 plus ``China_Drone``, the one RDD2022 country flown by UAV.
  The other six RDD2022 countries are dashcam or motorbike mounted and are
  dropped at indexing time, so their archives are never opened.

Usage:  python src/merge_datasets.py [--out dataset/merged] [--drone-only] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent

RDD_ZIP_DIR = ROOT / "dataset" / "RD2022" / "RDD2022"
RDD_COUNTRIES = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]
# RDD2022's only UAV-captured country; the rest are dashcam or motorbike mounted.
RDD_DRONE_COUNTRIES = {"China_Drone"}
UAV_ZIP = ROOT / "__root__" / "UAV-PDD2023.zip"

# Source label -> merged class id. Everything absent from this map is ignored.
POTHOLE_LABELS = {"D40": 0, "Pothole": 0}
CLASS_NAMES = {0: "Pothole"}

NEG_RATIO = 0.33      # negative images kept per positive image, within each source
SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))
SEED = 42
MIN_BOX_PX = 2        # drop boxes thinner than this many pixels on either side

# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    source: str          # "rdd:<country>" or "uav"
    stem: str            # output file stem, unique across the merged dataset
    group: str           # split-assignment unit; augmented copies share a group
    img_member: str      # path of the .jpg inside the zip
    boxes: list = field(default_factory=list)   # VOC (xmin, ymin, xmax, ymax)

    @property
    def positive(self) -> bool:
        return bool(self.boxes)


_UAV_AUG_PREFIX = re.compile(r"^(lr|tb|r180)_")


def uav_group(stem: str) -> str:
    """`lr_00418_top_left` -> `00418`: all crops/flips of one source photo."""
    return _UAV_AUG_PREFIX.sub("", stem).split("_")[0]


# --------------------------------------------------------------------------- #
# Reading the source archives
# --------------------------------------------------------------------------- #


def parse_voc(blob: bytes) -> list:
    """Return the pothole boxes of one VOC XML as (xmin, ymin, xmax, ymax)."""
    boxes = []
    for obj in ET.fromstring(blob).findall("object"):
        if obj.findtext("name") not in POTHOLE_LABELS:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            boxes.append(tuple(float(bb.findtext(k)) for k in
                               ("xmin", "ymin", "xmax", "ymax")))
        except (TypeError, ValueError):
            continue          # malformed box: skip it rather than guess
    return boxes


def read_source(zip_path: Path, xml_dir: str, img_dir: str, source: str,
                prefix: str, group_of) -> list:
    """Index one archive: parse every annotation, pair it with its image."""
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        images = {Path(n).stem: n for n in names
                  if n.startswith(img_dir) and n.lower().endswith(".jpg")}
        xmls = sorted(n for n in names
                      if n.startswith(xml_dir) and n.lower().endswith(".xml"))
        for member in xmls:
            stem = Path(member).stem
            img_member = images.get(stem)
            if img_member is None:
                continue      # annotation without an image: unusable
            samples.append(Sample(
                source=source,
                stem=f"{prefix}{stem}",
                group=f"{source}/{group_of(stem)}",
                img_member=img_member,
                boxes=parse_voc(zf.read(member)),
            ))
    return samples


def collect(drone_only: bool = False) -> dict:
    """Index every archive. Returns {source: [Sample, ...]}.

    With `drone_only`, the ground-level RDD2022 countries are skipped outright, so
    their zips are never opened (Norway alone is 10.6 GB). UAV-PDD2023 is drone
    imagery throughout and is always read.
    """
    countries = RDD_COUNTRIES
    if drone_only:
        countries = [c for c in RDD_COUNTRIES if c in RDD_DRONE_COUNTRIES]
        skipped = [c for c in RDD_COUNTRIES if c not in RDD_DRONE_COUNTRIES]
        print(f"  --drone-only: skipping ground-level {', '.join(skipped)}")

    per_source = {}
    for country in countries:
        zip_path = RDD_ZIP_DIR / f"{country}.zip"
        if not zip_path.exists():
            print(f"  ! missing {zip_path}, skipping", file=sys.stderr)
            continue
        source = f"rdd:{country}"
        # Only <country>/train/ carries annotations; <country>/test/ is unlabelled.
        per_source[source] = read_source(
            zip_path,
            xml_dir=f"{country}/train/annotations/xmls/",
            img_dir=f"{country}/train/images/",
            source=source,
            prefix="rdd_",
            group_of=lambda stem: stem,      # RDD images are independent
        )
        print(f"  {source:<26} {len(per_source[source]):>6} annotated images")

    if UAV_ZIP.exists():
        per_source["uav"] = read_source(
            UAV_ZIP,
            xml_dir="Annotations/",
            img_dir="JPEGImages/",
            source="uav",
            prefix="uav_",
            group_of=uav_group,
        )
        print(f"  {'uav':<26} {len(per_source['uav']):>6} annotated images")
    else:
        print(f"  ! missing {UAV_ZIP}, skipping", file=sys.stderr)
    return per_source


# --------------------------------------------------------------------------- #
# Selection and splitting
# --------------------------------------------------------------------------- #


def select(per_source: dict, rng: random.Random, neg_ratio: float) -> list:
    """Keep every positive, plus `neg_ratio` negatives per positive, per source."""
    kept = []
    for source in sorted(per_source):
        samples = per_source[source]
        pos = [s for s in samples if s.positive]
        neg = [s for s in samples if not s.positive]
        n_neg = min(len(neg), round(neg_ratio * len(pos)))
        chosen_neg = rng.sample(sorted(neg, key=lambda s: s.stem), n_neg)
        kept.extend(pos + chosen_neg)
        print(f"  {source:<26} {len(pos):>5} positive + {n_neg:>5} negative "
              f"(of {len(neg)} available)")
    return kept


def assign_splits(samples: list, rng: random.Random) -> dict:
    """Split by group, stratified on source and on whether the group has potholes.

    Splitting on groups rather than files keeps augmented copies of one photo
    together. Stratifying keeps every country represented in val and test.
    """
    groups = defaultdict(list)
    for s in samples:
        groups[s.group].append(s)

    strata = defaultdict(list)
    for group, members in groups.items():
        has_pos = any(m.positive for m in members)
        strata[(members[0].source, has_pos)].append(group)

    split_of = {}
    for key in sorted(strata):
        names = sorted(strata[key])
        rng.shuffle(names)
        n = len(names)
        # Largest-remainder allocation so no split silently rounds down to zero.
        cuts, start = [], 0
        for i, (split, frac) in enumerate(SPLITS):
            end = n if i == len(SPLITS) - 1 else start + int(round(frac * n))
            cuts.append((split, start, min(end, n)))
            start = min(end, n)
        for split, a, b in cuts:
            for g in names[a:b]:
                split_of[g] = split
    return split_of


# --------------------------------------------------------------------------- #
# Writing the merged dataset
# --------------------------------------------------------------------------- #


def to_yolo(box, width: int, height: int):
    """VOC corners -> YOLO (cx, cy, w, h), normalised and clipped to the image."""
    xmin, ymin, xmax, ymax = box
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    xmin, xmax = max(0.0, xmin), min(float(width), xmax)
    ymin, ymax = max(0.0, ymin), min(float(height), ymax)
    if xmax - xmin < MIN_BOX_PX or ymax - ymin < MIN_BOX_PX:
        return None
    return ((xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height,
            (xmax - xmin) / width, (ymax - ymin) / height)


def export(samples: list, split_of: dict, out: Path) -> dict:
    """Extract the selected images and write their YOLO labels."""
    for split, _ in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    by_zip = defaultdict(list)
    for s in samples:
        zip_path = (UAV_ZIP if s.source == "uav"
                    else RDD_ZIP_DIR / f"{s.source.split(':', 1)[1]}.zip")
        by_zip[zip_path].append(s)

    stats = defaultdict(Counter)
    rows, dropped, mismatched = [], 0, 0

    for zip_path in sorted(by_zip):
        with zipfile.ZipFile(zip_path) as zf:
            for s in sorted(by_zip[zip_path], key=lambda x: x.stem):
                split = split_of[s.group]
                blob = zf.read(s.img_member)
                try:
                    with Image.open(BytesIO(blob)) as im:
                        width, height = im.size
                except Exception as exc:               # truncated / not a JPEG
                    print(f"  ! unreadable {s.stem}: {exc}", file=sys.stderr)
                    continue
                if not width or not height:
                    continue

                lines = []
                for box in s.boxes:
                    yolo = to_yolo(box, width, height)
                    if yolo is None:
                        dropped += 1
                        continue
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in yolo))

                (out / "images" / split / f"{s.stem}.jpg").write_bytes(blob)
                # A negative gets an empty .txt: YOLO reads it as "no objects".
                (out / "labels" / split / f"{s.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))

                stats[split]["images"] += 1
                stats[split]["boxes"] += len(lines)
                stats[split]["positives"] += bool(lines)
                stats[f"{split}/{s.source}"]["images"] += 1
                stats[f"{split}/{s.source}"]["boxes"] += len(lines)
                rows.append({"file": f"{s.stem}.jpg", "split": split,
                             "source": s.source, "group": s.group,
                             "width": width, "height": height,
                             "boxes": len(lines)})

    with (out / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                                ["file", "split", "source", "group",
                                 "width", "height", "boxes"])
        writer.writeheader()
        writer.writerows(rows)

    if dropped:
        print(f"  dropped {dropped} degenerate boxes (< {MIN_BOX_PX}px a side)")
    return {k: dict(v) for k, v in stats.items()}


def write_data_yml(out: Path, sources: str) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in sorted(CLASS_NAMES.items()))
    (out / "data.yml").write_text(
        f"# Generated by src/merge_datasets.py -- {sources}\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names}\n"
    )


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "merged")
    ap.add_argument("--neg-ratio", type=float, default=NEG_RATIO)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--drone-only", action="store_true",
                    help="keep only drone-captured sources: UAV-PDD2023 and "
                         f"RDD2022/{'/'.join(sorted(RDD_DRONE_COUNTRIES))}")
    ap.add_argument("--dry-run", action="store_true",
                    help="index and split, but do not write any file")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Indexing archives")
    per_source = collect(args.drone_only)
    if not per_source:
        print("no source archives found", file=sys.stderr)
        return 1

    print("\nSelecting images")
    samples = select(per_source, rng, args.neg_ratio)

    print("\nAssigning splits (group-aware, stratified by source)")
    split_of = assign_splits(samples, rng)
    for split, _ in SPLITS:
        n = sum(1 for s in samples if split_of[s.group] == split)
        print(f"  {split:<6} {n:>6} images")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    print(f"\nExporting to {args.out}")
    stats = export(samples, split_of, args.out)
    sources = ("RDD2022 ({}) + UAV-PDD2023".format("/".join(sorted(RDD_DRONE_COUNTRIES)))
               if args.drone_only else "RDD2022 + UAV-PDD2023")
    write_data_yml(args.out, sources)
    (args.out / "stats.json").write_text(json.dumps(
        {"seed": args.seed, "neg_ratio": args.neg_ratio,
         "drone_only": args.drone_only, "splits": stats}, indent=2))

    print("\nMerged dataset")
    for split, _ in SPLITS:
        s = stats.get(split, {})
        print(f"  {split:<6} images={s.get('images', 0):>6} "
              f"boxes={s.get('boxes', 0):>6} "
              f"positives={s.get('positives', 0):>6}")
    print(f"\ndata.yml -> {args.out / 'data.yml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
