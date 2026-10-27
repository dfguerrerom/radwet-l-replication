# RadWet-L replication

An independent, open re-implementation of **RadWet-L**, the ALOS-2 PALSAR-2
L-band radar classifier that maps **inundated forest** and **open water** in
the Amazon Basin (Oakes et al., 2024).

It runs the authors' pre-trained XGBoost models. On the published Zenodo maps
it reproduces flood-forest area to within ±3 percentage points on 7 of 8 test
cycles (2019–2023), with a mean error of about 2 pp.

> **Not the authors' code.** RadWet-L was developed by Gregory Oakes, Andy Hardy,
> Pete Bunting (Aberystwyth University) and Ake Rosenqvist (soloEO). The
> original implementation is
> [gro5-AberUni/ALOS-LBand](https://github.com/gro5-AberUni/ALOS-LBand). This
> repository rebuilds that pipeline from the paper and the original scripts,
> and documents what it takes to match the published products. See
> [Credits and citation](#credits-and-citation).

## Results at a glance

Loreto test area, Peru (`data/aoi/loreto_test_area.geojson`), flood-forest share:

| Cycle | Dates | Season | This repo | Published (Zenodo) | Δ pp |
|-------|-------|--------|-----------|--------------------|------|
| C119 | Jan–Feb 2019 | rising | 33.2 % | 33.4 % | −0.2 |
| C122 | Mar 2019 | peak | 43.5 % | 40.7 % | +2.9 |
| C130 | Jul 2019 | receding | 21.2 % | 23.1 % | −1.8 |
| C145 | Jan–Feb 2020 | rising | 26.4 % | 25.3 % | +1.2 |
| C148 | Mar 2020 | peak | 27.7 % | 26.7 % | +1.0 |
| C174 | Mar 2021 | peak, single pass | 30.4 % | 24.6 % | +5.8 |
| C200 | Mar 2022 | peak | 23.7 % | 23.5 % | +0.2 |
| C226 | Mar 2023 | peak | 22.2 % | 24.9 % | −2.7 |

Pixel-by-pixel agreement is 81–84 %. Most of the remaining gap is the
forest/non-forest mask: the authors' pre-built mask file is not public, and
the reconstruction used here marks fewer pixels as non-forest. With the mask
taken from the published products, agreement on a second test area rises from
85 % to 97 % over all 43 published cycles.

[`docs/findings.md`](docs/findings.md) has the full validation, what had to
be gotten right (working resolution, multi-pass mosaicking, resampling), and
the remaining differences.

## How it works

```
ALOS-2 ScanSAR L2.2 (HH, HV, incidence angle) ─┐
Slope (HydroSHEDS) · HAND (MERIT Hydro) ────────┼─► resample to 50 m grid
Forest mask (Copernicus LC + Hansen GFC) ──────┘
        │
        ▼
 1. DN → dB  (10·log10(DN²) − 83),  NDPI = (HH − HV)/(HH + HV)
 2. 3×3 Lee speckle filter; stack HH, HV, NDPI, incidence angle
 3. Mask non-forest and slope > 10°
 4. Shepherd segmentation in two stages (low-backscatter areas, then the
    whole image), merged
 5. Per-segment statistics → XGBoost water model (8 features)
                           → XGBoost flood model (4 features)
 6. Combine, 3×3 mode filter, sieve objects < 50 px
 7. Stamp masked pixels as classes 5 and 6
        │
        ▼
 classified.tif
```

| Value | Class | Colour |
|-------|-------|--------|
| 0 | No data | `#000000` |
| 1 | Inundated forest | `#6CABDD` |
| 2 | Open water | `#000080` |
| 4 | Other land cover (forest, not flooded) | `#004225` |
| 5 | Non-forest (masked) | `#a7a7a7` |
| 6 | Slope > 10° (masked) | `#d21255` |

Products are made per **orbit cycle**. ALOS-2 repeats its orbit every 14
days, and cycles are numbered from August 2014 (`data/cycle_dates.csv`,
cycles 1–279, up to March 2025). When several passes cover an area within
one cycle, they are combined into an *angle-optimal* mosaic: each pixel comes
from the pass whose incidence angle is closest to the nominal 37°.

## Setup

The geospatial stack (`rsgislib`, `rios`) is only available from conda-forge,
so use conda, mamba or micromamba:

```bash
git clone https://github.com/dfguerrerom/radwet-l-replication.git
cd radwet-l-replication
micromamba env create -f environment.yml     # or: conda env create -f environment.yml
micromamba activate radwet-l
python -m pytest                              # unit tests, a few seconds
```

Downloading inputs uses [Google Earth Engine](https://earthengine.google.com/)
(free for research) with a Cloud project. Exports are written to Google
Drive and then downloaded, so authenticate with Drive scope:

```bash
earthengine authenticate --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.full_control
earthengine set_project YOUR-PROJECT   # or pass --project / set EE_PROJECT
```

## Usage

Run all commands from the repository root.

**1. Download inputs** for an area and one or more cycles. The area can be a
GeoJSON file, a shapefile, a zipped shapefile, or `--bbox W S E N`:

```bash
python -m radwet_l.fetch --aoi data/aoi/loreto_test_area.geojson \
    --cycle 226 --out inputs/loreto
```

This writes the slope, HAND and forest-mask layers to `inputs/loreto/aux/`
once. For each cycle it writes the angle-optimal HH/HV/LIN mosaic to
`inputs/loreto/C226/`. It then prints the classification command. Files
already present are skipped, so the command can be re-run.

**2. Classify:**

```bash
python -m radwet_l.pipeline \
    --hh  inputs/loreto/C226/C226_angleopt_HH.tif \
    --hv  inputs/loreto/C226/C226_angleopt_HV.tif \
    --inc inputs/loreto/C226/C226_angleopt_LIN.tif \
    --slope inputs/loreto/aux/Slope_HydroSHEDS_100m.tif \
    --hand  inputs/loreto/aux/HAND_MERITHydro_90m.tif \
    --lcc   inputs/loreto/aux/ForestMask_Copernicus_Hansen.tif \
    --level L22 --outdir runs/C226_loreto
```

The result is `runs/C226_loreto/classified.tif`; intermediate files are in
`work/` (they can be deleted). A 2,600 km² area takes about a minute on a
laptop.

**3. Compare with the published product** (Zenodo covers cycles 119–240):

```bash
python -m radwet_l.zenodo download --out zenodo --cycle 226    # ~125 MB per cycle
python -m radwet_l.zenodo compare --ours runs/C226_loreto/classified.tif \
    --cycle 226 --zenodo-dir zenodo
```

### Using your own imagery

The pipeline works with any HH/HV rasters in EPSG:4326, not just those from
`fetch`.

- `--level L22`: Level 2.2 γ⁰ stored as DN (as on Earth Engine).
- `--level L11`: Level 1.1 DN; `--inc` is then required.
- Incidence angle may be in degrees or degrees × 100.
- Without `--inc`, an L2.2 run uses a constant 38.5°, which is less accurate
  across the swath.
- If a pass is split into numbered scenes (`*_HH_0.tif`, `*_HH_1.tif`), pass
  the first one. The siblings are averaged in automatically.
- The forest mask must use **0 = forest, 1 = non-forest**.

## Repository layout

```
radwet_l/            the package
  pipeline.py        classification CLI (steps above)
  preprocess.py      calibration, NDPI, Lee filter, resampling, masks
  segment.py         two-stage Shepherd segmentation
  classify.py        per-segment statistics + XGBoost prediction
  postprocess.py     class combination, mode and sieve filters
  fetch.py           Earth Engine input download (angle-optimal mosaic, ancillary layers)
  zenodo.py          download / compare against the published products
  cycles.py          orbit-cycle calendar
  gee.py             Earth Engine export + Drive download helpers
  config.py          thresholds, features, class codes
models/              pre-trained models from the original repository (see models/README.md)
data/                cycle calendar, example area of interest
docs/findings.md     validation and replication notes
tests/               unit tests
```

## Credits and citation

**Method and models.** If you use this code, cite the original paper:

> Oakes, G., Hardy, A., Bunting, P., & Rosenqvist, A. (2024). RadWet-L: A Novel
> Approach for Mapping of Inundation Dynamics of Forested Wetlands Using ALOS-2
> PALSAR-2 L-Band Radar Imagery. *Remote Sensing*, 16(12), 2078.
> https://doi.org/10.3390/rs16122078

```bibtex
@article{oakes2024radwetl,
  title   = {RadWet-L: A Novel Approach for Mapping of Inundation Dynamics of Forested
             Wetlands Using ALOS-2 PALSAR-2 L-Band Radar Imagery},
  author  = {Oakes, Gregory and Hardy, Andy and Bunting, Pete and Rosenqvist, Ake},
  journal = {Remote Sensing},
  volume  = {16},
  number  = {12},
  pages   = {2078},
  year    = {2024},
  doi     = {10.3390/rs16122078}
}
```

To cite this replication as well, use [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository").

**Original code.** [gro5-AberUni/ALOS-LBand](https://github.com/gro5-AberUni/ALOS-LBand)
(GPL-3.0). The models in `models/` and `data/cycle_dates.csv` come from that
repository.

**Published products.** Oakes, G. (2024). *RadWet-L ALOS-2 PALSAR-2 ScanSAR
Orbit Cycle Classified Products 2019-2023* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.10566417 (CC BY 4.0). They are used for
validation only and are not redistributed here.

**Input data.** None of these datasets is included in the repository; `fetch`
reads them from Earth Engine. Each has its own terms; check them before
redistributing derived products.

| Dataset | Used for | Cite | Terms |
|---------|----------|------|-------|
| ALOS-2 PALSAR-2 ScanSAR L2.2 © JAXA | HH, HV, incidence angle | Acknowledge JAXA EORC as data provider | [Catalog](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR-2_Level2_2_ScanSAR) |
| WWF HydroSHEDS conditioned DEM | Slope | Lehner, Verdin & Jarvis (2008), *Eos* 89(10) | [Catalog](https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_03CONDEM) |
| MERIT Hydro v1.0.1 | HAND | Yamazaki et al. (2019), *Water Resources Research* 55 | [Catalog](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1) |
| Copernicus Global Land Cover 100 m (Collection 3) | Forest mask | Buchhorn et al. (2020), Copernicus Global Land Service | [Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global) |
| Hansen Global Forest Change v1.12 | Forest mask (loss after 2016) | Hansen et al. (2013), *Science* 342 | [Catalog](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12) |

## License

[GPL-3.0](LICENSE), the same license as the original RadWet-L code. This is a
derived work. The pipeline follows the logic of the original
`applyXGBoostClassificationImgs_Bins.py`, and the models are converted copies
of the original files.

Changes from the original, made in 2026:

- the pipeline was rewritten as a Python package;
- the models were converted from the legacy XGBoost format to UBJ/JSON;
- Earth Engine input download was added, with angle-optimal multi-pass
  mosaicking;
- tools to compare against the published products were added.
