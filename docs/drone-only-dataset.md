# Restricting the dataset to drone imagery

This document records why the pothole dataset was narrowed to UAV imagery, what code was written
to make that possible, and what the resulting dataset contains — including the limitations the
restriction imposes on the results that can be claimed from it.

## 1. Motivation

The detector is intended for UAV-based road inspection, but the dataset it was originally trained
on did not represent that viewpoint. The merged dataset built from RDD2022 and UAV-PDD2023 held
5,062 images, of which only 261 — about **5%** — were captured from a drone. The remaining 95%
came from cameras mounted on cars and motorbikes.

The two viewpoints are not minor variations of one another:

| | Ground-level capture | Drone capture |
|---|---|---|
| Camera attitude | Oblique, horizon in frame | Nadir (straight down) |
| Road coverage | Lower third of the image | Entire image |
| Scene content | Sky, buildings, traffic, vegetation | Pavement texture only |
| Pothole appearance | Small, foreshortened, perspective-distorted | Near-planar, consistent scale |
| Illumination | Mixed, with strong cast shadows | Comparatively uniform |

A detector trained predominantly on oblique dashcam frames learns a perspective geometry and a
scene context that never occur in nadir drone footage. Keeping that majority in the training set
optimises the model for a deployment condition that does not apply. The dataset was therefore
restricted to drone imagery only.

## 2. Identifying the drone sources

**UAV-PDD2023** is drone-captured in its entirety, so all 2,440 of its annotated images qualify.

**RDD2022** is organised by country, and the countries differ in capture platform. Six of the
seven — India, Japan, Czech, Norway, United States and China_MotorBike — are vehicle-mounted
(dashcam or motorbike). One, **`China_Drone`**, is UAV-captured, as its name records and as
inspection of its images confirms: 512×512 frames of pavement viewed from directly above, with no
horizon and no roadside context, visually consistent with the UAV-PDD2023 material. A minor
artifact worth noting for preprocessing: 8 of the 85 retained `China_Drone` images carry a black
band along one edge, a padding residue from the original capture.

So the restriction is not "drop RDD2022". It is: keep RDD2022's drone subset, drop its six
ground-level subsets, keep UAV-PDD2023 whole.

## 3. The code

### 3.1 Where the filter belongs

`src/merge_datasets.py` generates the dataset from the distribution zips. It is the single source
of truth: in one pass it writes the images, the YOLO label files, `data.yml`, `manifest.csv` and
`stats.json`, and the whole result is reproducible from a fixed `--seed`.

Two implementations were possible:

1. **Prune the existing output** — delete the ground-level images and labels from
   `dataset/merged/` after the fact.
2. **Filter at the source** — teach the generator to skip non-drone sources and build a fresh
   dataset.

The second was chosen. Pruning would leave `manifest.csv` and `stats.json` describing a dataset
that no longer existed, would silently invalidate the recorded split proportions, and would be
irreversible — the deleted files could only be recovered by re-running the generator anyway.
Filtering at the source keeps all five outputs mutually consistent by construction, keeps the
result reproducible, and leaves the original dataset intact for comparison.

The filter is applied at **indexing** time rather than at export time. This matters practically:
the six excluded country archives are never opened at all, and `Norway.zip` alone is 10.6 GB. The
drone-only build completes in seconds rather than minutes.

### 3.2 The changes

A single declarative constant records which RDD2022 countries are drone-captured:

```python
# RDD2022's only UAV-captured country; the rest are dashcam or motorbike mounted.
RDD_DRONE_COUNTRIES = {"China_Drone"}
```

`collect()` consults it when the flag is set, and reports what it skipped so a run is
self-explanatory from its own log:

```python
def collect(drone_only: bool = False) -> dict:
    countries = RDD_COUNTRIES
    if drone_only:
        countries = [c for c in RDD_COUNTRIES if c in RDD_DRONE_COUNTRIES]
        skipped = [c for c in RDD_COUNTRIES if c not in RDD_DRONE_COUNTRIES]
        print(f"  --drone-only: skipping ground-level {', '.join(skipped)}")
```

UAV-PDD2023 needs no condition, being drone imagery throughout.

Two smaller changes keep the output self-describing. `write_data_yml()` now receives the
provenance string instead of hardcoding it, so a drone-only build is labelled
`RDD2022 (China_Drone) + UAV-PDD2023` in its own header; and `stats.json` records
`"drone_only": true` alongside the existing `seed` and `neg_ratio`, so any dataset directory
states the conditions that produced it.

The flag defaults to off, so the original mixed dataset remains reproducible from the same script.
This was verified: running the generator with no flag still reports 4,048 / 501 / 513 images,
matching the recorded baseline exactly.

### 3.3 What was deliberately not changed

Everything downstream of source selection is platform-agnostic and was reused untouched:

- `select()` — retains every pothole-positive image, plus negatives at `NEG_RATIO` per positive,
  computed *within each source*, so the balance holds per source rather than only in aggregate.
- `assign_splits()` — the group-aware, stratified split. This remains essential here: UAV-PDD2023
  ships augmented copies of each source photograph (`lr_`/`tb_`/`r180_` prefixes for flips and
  rotation, `_top_left`-style suffixes for quadrant crops), and all copies of one photograph must
  land in the same split or near-duplicates straddle train and validation and inflate the score.
  Because stratification is per `(source, has_pothole)`, both remaining sources are guaranteed
  representation in validation and test.
- `to_yolo()` and `export()` — box normalisation against the true decoded JPEG dimensions, and
  the writing of empty `.txt` files for negatives.

The class mapping is also unchanged: RDD2022's `D40` and UAV-PDD2023's `Pothole` both map to class
`0 Pothole`, and all other distress types (cracks, repairs) are discarded rather than relabelled,
so an image containing only cracks becomes a negative.

## 4. Result

Written to `dataset/drone/`, leaving `dataset/merged/` in place.

### Composition

| | Baseline (mixed) | Drone-only |
|---|---|---|
| Images | 5,062 | **261** |
| Positive images | 3,674 | **196** |
| Pothole boxes | 6,544 | **281** |
| Sources | 8 | **2** |

Every drone-captured pothole box available in either archive is present — 86 from `China_Drone`
and 195 from UAV-PDD2023, totalling 281. Nothing was lost to the filter beyond the ground-level
material it was meant to remove.

### By source

| Source | Images | Positive | Negative | Boxes | Resolution |
|---|---|---|---|---|---|
| `uav` (UAV-PDD2023) | 176 | 132 | 44 | 195 | 2592×1944 |
| `rdd:China_Drone` | 85 | 64 | 21 | 86 | 512×512 |

### By split

| Split | Images | Positive | Boxes | Groups | Positive groups |
|---|---|---|---|---|---|
| train | 210 | 159 | 228 | 120 | 76 |
| val | 28 | 21 | 28 | 14 | 9 |
| test | 23 | 16 | 25 | 16 | 10 |

## 5. Limitations

Two consequences of the restriction should be stated alongside any result obtained from this
dataset.

**The dataset is small, and its effective diversity is smaller than the image count suggests.**
The 132 positive UAV-PDD2023 images derive from only **31 distinct source photographs**; the
remainder are flips, rotations and quadrant crops of those 31. Added to `China_Drone`'s 64
independent frames, the dataset contains roughly **95 genuinely independent pothole scenes**. This
is a ceiling imposed by the source archives, not by the filter: UAV-PDD2023 contains 195 pothole
boxes in total against 5,398 transverse cracks, and `China_Drone` yields 86. Every one of them is
already included.

**Validation and test sets are thin.** The 80/10/10 split leaves 21 positive validation images
drawn from 9 independent scenes, and 16 positive test images from 10. At that size a single image
moves mAP by several points, so validation mAP will fluctuate markedly between epochs and the
`patience=50` early-stopping criterion rests on a weak signal. Test-set figures should be reported
with that caveat and not compared to the mixed-dataset baseline as if the two were measured at
equivalent precision.

Both are properties of the available drone data. Mitigations, should they become necessary, would
be to raise the negative ratio (roughly 4,645 pothole-free drone images remain unused across the
two archives), to adopt a wider split such as 70/15/15, or to evaluate with cross-validation over
groups rather than a single held-out split.

## 6. Reproduction

```bash
# Drone-only dataset (what src/main.py now trains on)
python src/merge_datasets.py --drone-only --out dataset/drone

# Original mixed dataset, unchanged and still reproducible
python src/merge_datasets.py --out dataset/merged
```

Add `--dry-run` to index and split without writing files. `--neg-ratio` and `--seed` behave as
before.

Verification applied to the generated dataset:

```bash
# only drone sources present
awk -F, 'NR>1{print $3}' dataset/drone/manifest.csv | sort -u

# every drone pothole box retained (expect 281)
cat dataset/drone/labels/*/*.txt | wc -l

# no augmentation group straddles two splits (expect no output)
awk -F, 'NR>1{print $4,$2}' dataset/drone/manifest.csv | sort -u | awk '{print $1}' | uniq -d

# labels well-formed: class 0 only, coordinates in (0,1] (expect no output)
cat dataset/drone/labels/*/*.txt \
  | awk '$1!=0 || $2<=0 || $2>1 || $3<=0 || $3>1 || $4<=0 || $4>1 || $5<=0 || $5>1'
```

All checks pass on the generated dataset. Note that `dataset/` is git-ignored: the archives and
the generated datasets are not tracked, so the generator and this document are the reproducible
record.
