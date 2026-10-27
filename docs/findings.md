# Replication findings

What it took to reproduce the published RadWet-L products, how close the
replication gets, and where the remaining differences come from.

Reference products: Oakes et al. (2024), Zenodo record
[10566417](https://zenodo.org/records/10566417), 43 Amazon-wide
classifications for orbit cycles 119–240 (2019–2023).

All percentages below are shares of classified pixels inside the test area.
"Flood forest" is class 1 (inundated forest).

## 1. Validation results

### Loreto test area (Peru, ~2.5° × 2.2°)

Test area: `data/aoi/loreto_test_area.geojson`
(lon −76.47 to −73.99, lat −5.90 to −3.75), lowland Amazon floodplain.

| Cycle | Dates | Season | Passes | Flood forest, ours | Flood forest, Zenodo | Δ (pp) |
|-------|-------|--------|--------|------|--------|-----|
| C119 | 2019-01-28 – 02-10 | rising | multi | 33.2 % | 33.4 % | −0.2 |
| C122 | 2019-03-11 – 03-24 | peak | multi | 43.5 % | 40.7 % | +2.9 |
| C130 | 2019-07-01 – 07-14 | receding | multi | 21.2 % | 23.1 % | −1.8 |
| C145 | 2020-01-27 – 02-09 | rising | multi | 26.4 % | 25.3 % | +1.2 |
| C148 | 2020-03-09 – 03-22 | peak | multi | 27.7 % | 26.7 % | +1.0 |
| C174 | 2021-03-08 – 03-21 | peak | single | 30.4 % | 24.6 % | +5.8 |
| C200 | 2022-03-07 – 03-20 | peak | multi | 23.7 % | 23.5 % | +0.2 |
| C226 | 2023-03-06 – 03-19 | peak | multi | 22.2 % | 24.9 % | −2.7 |

Seven of eight cycles are within ±3 pp; mean absolute error is about 2 pp.
C174 is the outlier: only one pass covers the area in that cycle, so the
angle-optimal mosaic (section 2.2) cannot help.

Open water is within 0.7 pp on every cycle, and pixel-by-pixel agreement is
81–84 %. Non-forest is 6.15 pp lower than Zenodo on every cycle; that is the
forest mask (section 3.1). The table can be regenerated with
`python -m radwet_l.zenodo compare`.

### Second area: a 2,645 km² catchment in Loreto, all 43 cycles

| | Mean pixel agreement | Mean flood-forest bias |
|--|--|--|
| Reconstructed forest mask (as shipped here) | 85.2 % | +2.0 pp |
| Forest mask taken from the Zenodo products | 96.8 % | +1.0 pp |

Most of the disagreement comes from the forest mask, not the classifier
(section 3.1). With the paper's own mask, the remaining gap is about 3 %.

The worst single cycle is C203 (late April 2022): flood forest is +13 pp
over Zenodo (+9 pp with the paper's mask). This is a classifier difference
the mask does not explain.

## 2. What matters for a faithful replication

Each item below was found by changing one thing and measuring the effect.

### 2.1 Use the 50 m working grid

The models were trained on 50 m data. Running inference on the native 25 m
L2.2 imagery produces smaller segments with different statistics, and
flood forest roughly doubles:

| C174 run | Flood forest |
|----------|--------------|
| 25 m (native) | 59.1 % |
| 50 m | 29.9 % |
| Zenodo | 24.6 % |

The pipeline resamples SAR inputs to 50 m by default (`--working-grid 50m`).

### 2.2 Mosaic multiple passes by incidence angle, not by averaging

In most 14-day cycles, several ScanSAR passes cover an area. The incidence
angle varies by roughly ±7° across each swath around the nominal 37°, and
backscatter changes with it.

- **Averaging the passes** mixes geometries: C119 gives 27.3 % flood forest.
- **A single pass** leaves extreme angles at its swath edges. It gave a
  left/right bias of +6 / −12 pp, and 30.3 % overall.
- **Angle-optimal mosaicking** gives 33.2 %, against 33.4 % for Zenodo. Each
  pixel takes the pass whose local incidence angle is closest to 37°.

`radwet_l.fetch` builds this mosaic on Earth Engine
(`qualityMosaic` on −|LIN/100 − 37|).

JAXA later advised (June 2026) that its per-cycle 14-day ScanSAR mosaics
are deprecated. Brightness differs between overlapping paths and beams,
which is the same effect. JAXA recommends the re-processed Level 2.2 scene
data, which is what this repository uses.

### 2.3 Reproject SAR with nearest neighbour

Earth Engine sometimes exports L2.2 scenes in UTM. Reprojecting them to
EPSG:4326 with `average` resampling smooths the high-HH double-bounce peaks
that signal flooded forest. NDPI standard deviation drops by 24 %, and C119
flood forest falls from 30.3 % to 26.5 %. Use `near`. `radwet_l.fetch`
exports directly in EPSG:4326, which Earth Engine resamples with nearest
neighbour by default.

### 2.4 Fill scene gaps within a pass

One C174 pass was delivered as two scenes, and the first had a diagonal
no-data gap covering 7.9 % of the area. Classifying it alone left a strip
artifact. The pipeline averages numbered sibling scenes (`*_HH_0.tif`,
`*_HH_1.tif`, …) automatically (`--scene-mosaic auto`). They agree to within
0.06 % in the overlap.

### 2.5 Use imagery from the right cycle

The JAXA K&C *annual* mosaic for 2021 is a dry-season composite
(April–December). Using it for the March 2021 cycle gave 37.2 % flood forest.
A product must be built from that cycle's own passes.

### 2.6 Ancillary sources

Using the paper's sources (paper Table 1) rather than substitutes made little
difference to flood forest. On C174, swapping AW3D30 slope for HydroSHEDS
changed nothing, and adding Copernicus to a Hansen-only mask changed it by
0.3 pp. They are used here for fidelity:

| Layer | Source | Earth Engine asset |
|-------|--------|--------------------|
| Slope | WWF HydroSHEDS conditioned DEM, 100 m | `WWF/HydroSHEDS/03CONDEM` |
| HAND | MERIT Hydro v1.0.1, 90 m | `MERIT/Hydro/v1_0_1` (`hnd`) |
| Forest mask | Copernicus Global Land Cover (classes 30/40/50/60) + Hansen GFC loss after 2016 | `COPERNICUS/Landcover/100m/Proba-V-C3/Global`, `UMD/hansen/global_forest_change_2024_v1_12` |

## 3. Remaining differences

### 3.1 The forest mask

The original pipeline used a pre-built mask
(`Combined_LCC-GFC_LY-16_Binary_COG_Blank_COG.tif`) that is not published.
The reconstruction from the same two sources marks far fewer pixels as
non-forest: 1.9 % vs 7.9 % in the Loreto area, and about 3 % vs 14 % in the
catchment.

Class 5 (non-forest) is not predicted by the classifier. It is stamped from
the mask after classification, so this gap shows up directly as non-forest
pixels that we label as other classes.

The Zenodo non-forest class is identical in every cycle (14.05 % of the
catchment in all 43 products). That makes it a static mask that can be
extracted from any product. Using it lifts agreement from 85 % to 97 %
(section 1).

### 3.2 Segment-boundary pixels

After the mask, a gap of about 3–4 % remains. Pixels we classify as flood
forest while Zenodo does not, and the reverse, have nearly the same
backscatter (HH −6.12 vs −6.16 dB, NDPI −0.45 vs −0.42). They sit on the
XGBoost decision boundary. Classification is per segment, so small input
differences change the Shepherd segments and move these pixels to either
side.

### 3.3 Single-pass dry-season cycles

In 3 of 104 cycles tried on the catchment, segmentation failed with
*"Not enough unique cluster centres"*. All three were single-pass,
dry-season cycles with too few distinct backscatter values for the
250-cluster k-means.

## 4. Notes on the original code

- **Feature counts.** `applyXGBoostClassificationImgs_Bins.py` builds 21
  (water) and 9 (flood) features, but the shipped models expect 8 and 4.
  This replication uses the lists from the training scripts (see
  `models/README.md`).
- **Calibration.** Level 2.2 γ⁰ is stored as DN, so the same
  `dB = 10·log10(DN²) − 83` conversion applies to L1.1 and L2.2. DN = 1
  (−83 dB) is treated as no data, as in the original.
- **Masking order.** The forest and slope masks are applied before
  segmentation and again after the mode and sieve filters (as classes 5 and
  6), matching the original.
- **Temporal in-filling** (`infillData.py`) only fills background pixels
  (about 0.2 %), so it is not implemented.
- **Cadence.** "14 days" is the orbit-cycle length. The "42 days" in the
  paper is how often a given area is revisited (every third cycle). Each
  product is one cycle's passes, not a 42-day composite.
- **Other issues found in the original scripts** (not carried over):

| File | Issue |
|------|-------|
| `trainXGBoostGPU_Water.py:29` | calls an undefined `crash()` |
| `applyXGBoostClassificationImgs_Bins.py:236` | indentation error in `extractTraining()` |
| `applyXGBoostClassificationImgs_Bins.py:359` | argparse help says "Sentinel 1" |
| `applyXGBoostClassificationImgs_Bins.py:660, 675, 843` | bare `except:` clauses hide errors |
| `radwet.yml:9` | hard-coded conda prefix |
