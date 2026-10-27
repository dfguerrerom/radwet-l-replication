"""RadWet-L classification pipeline.

Usage (from the repository root):

    # ALOS-2 Level 2.2 (gamma-naught DN, e.g. from Google Earth Engine):
    python -m radwet_l.pipeline \
        --hh C226_HH.tif --hv C226_HV.tif --inc C226_LIN.tif \
        --slope Slope.tif --hand HAND.tif --lcc ForestMask.tif \
        --level L22 --outdir runs/C226

    # ALOS-2 Level 1.1 (raw DN, incidence angle required):
    python -m radwet_l.pipeline \
        --hh tile_HH.tif --hv tile_HV.tif --inc tile_linci.tif \
        --slope Slope.tif --hand HAND.tif --lcc ForestMask.tif \
        --level L11 --outdir runs/L11

SAR inputs are resampled to the 50 m working grid the models were trained on
before segmentation and classification. ``--working-grid native`` keeps the
input resolution and is only meant for diagnostics.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import rios.rat
import rsgislib
import rsgislib.imageutils
import rsgislib.rastergis
from osgeo import gdal

from .classify import classify_flood, classify_water, extract_segment_stats
from .config import (
    CLASS_NAMES, CLASS_NON_FOREST, CLASS_STEEP_SLOPE, COLOR_TABLE,
    DEFAULT_WORKING_GRID, FALLBACK_INCIDENCE_DEG, MODEL_DIR, SLOPE_MASK_THRESH,
    TARGET_PIXEL_SIZE_50M_DEGREES, WORKING_GRID_RESAMPLE_ALG,
)
from .postprocess import (
    apply_mode_filter, apply_sieve_filter, combine_classifications,
)
from .preprocess import (
    apply_lee_filter, compute_ndpi, dn_to_db, find_scene_group_paths,
    gen_valid_mask, get_pixel_size, mask_image, mask_image_with_class,
    mosaic_scene_group, resample_to_match, resample_to_working_grid,
    save_geotiff, stack_bands,
)
from .segment import create_low_backscatter_image, segment_tile

OUTPUT_NAME = "classified.tif"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m radwet_l.pipeline",
        description="RadWet-L wetland classification pipeline (replication)",
    )
    parser.add_argument("--hh", required=True, help="HH band GeoTIFF")
    parser.add_argument("--hv", required=True, help="HV band GeoTIFF")
    parser.add_argument("--inc", help="Incidence-angle GeoTIFF (degrees or degrees x 100; "
                                      "required for L11)")
    parser.add_argument("--slope", required=True, help="Slope raster (degrees)")
    parser.add_argument("--hand", required=True, help="HAND raster (m)")
    parser.add_argument("--lcc", required=True,
                        help="Forest mask (0=forest, 1=non-forest)")
    parser.add_argument("--level", choices=["L11", "L22"], required=True,
                        help="Data level: L11 (raw DN) or L22 (gamma-naught DN)")
    parser.add_argument(
        "--working-grid",
        choices=["50m", "native"],
        default=DEFAULT_WORKING_GRID,
        help="Working grid used for segmentation/classification. Default: 50m",
    )
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--models", default=str(MODEL_DIR),
                        help="Directory with XGBoost models and scalers. Default: models/")
    parser.add_argument(
        "--scene-mosaic",
        choices=["auto", "off"],
        default="auto",
        help="Detect numbered sibling scenes (e.g. *_HH_0.tif, *_HH_1.tif) and "
             "average them into one input before preprocessing. Default: auto",
    )
    parser.add_argument("--reuse-segments", metavar="DIR",
                        help="Reuse clumps from a previous run's work/ dir "
                             "(skips preprocessing and segmentation, re-runs "
                             "stats extraction and classification with current inputs)")
    return parser.parse_args(argv)


def load_band(path):
    """Load single-band raster as numpy array."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    del ds
    return arr


def load_incidence(path):
    """Load incidence angle in degrees, accepting GEE's degrees x 100 encoding."""
    inc_arr = load_band(path)
    if np.nanmax(inc_arr) > 360:
        inc_arr = inc_arr / 100.0
        print("  Auto-scaled incidence angle from deg*100 to degrees")
    return inc_arr


def prepare_scene_inputs(args, workdir):
    """Resolve and optionally mosaic sibling scene rasters."""
    resolved = {"hh": args.hh, "hv": args.hv, "inc": args.inc}
    groups = {"hh": [args.hh], "hv": [args.hv], "inc": [args.inc] if args.inc else []}

    if args.scene_mosaic == "off":
        return resolved["hh"], resolved["hv"], resolved["inc"], groups

    for label, source_path in (("hh", args.hh), ("hv", args.hv), ("inc", args.inc)):
        if not source_path:
            continue
        group = find_scene_group_paths(source_path)
        groups[label] = group
        if len(group) > 1:
            mosaic_path = os.path.join(workdir, f"{label}_scene_mosaic.tif")
            resolved[label] = mosaic_scene_group(group, mosaic_path)
        else:
            resolved[label] = group[0]

    return resolved["hh"], resolved["hv"], resolved["inc"], groups


def prepare_working_inputs(hh_path, hv_path, inc_path, working_grid, workdir):
    """Prepare SAR inputs on the requested working grid."""
    if working_grid == "native":
        return hh_path, hv_path, inc_path

    def to_grid(path, name):
        return resample_to_working_grid(
            path, os.path.join(workdir, name), TARGET_PIXEL_SIZE_50M_DEGREES,
            resample_alg=WORKING_GRID_RESAMPLE_ALG,
        )

    hh_path = to_grid(hh_path, "hh_working_50m.tif")
    hv_path = to_grid(hv_path, "hv_working_50m.tif")
    inc_path = to_grid(inc_path, "inc_working_50m.tif") if inc_path else None
    return hh_path, hv_path, inc_path


def prepare_ancillary_and_masks(args, reference_img, stacked_img, workdir):
    """Resample slope/HAND/forest mask to the SAR grid and mask the stacked image.

    The forest and slope masks are applied before segmentation, as in the
    original pipeline, and re-applied after the sieve filter as classes 5/6.
    """
    print("[3/10] Resampling ancillary data to working grid...")
    slope_re = os.path.join(workdir, "slope_resampled.tif")
    hand_re = os.path.join(workdir, "hand_resampled.tif")
    lcc_re = os.path.join(workdir, "lcc_resampled.tif")
    resample_to_match(args.slope, reference_img, slope_re)
    resample_to_match(args.hand, reference_img, hand_re)
    resample_to_match(args.lcc, reference_img, lcc_re)

    print("[4/10] Creating and applying slope/LCC masks to stacked image...")
    slope_arr = load_band(slope_re)
    slope_mask_arr = np.where(slope_arr > SLOPE_MASK_THRESH, 1, 0).astype(np.uint8)
    slope_mask_img = os.path.join(workdir, "slope_mask.tif")
    save_geotiff(reference_img, [slope_mask_arr.astype(np.float32)], slope_mask_img)
    print(f"  Slope mask: {np.sum(slope_mask_arr)} pixels > {SLOPE_MASK_THRESH} deg")

    stacked_lcc_masked = os.path.join(workdir, "stacked_4band_lcc_masked.tif")
    mask_image(stacked_img, lcc_re, stacked_lcc_masked, mask_value=1)

    stacked_masked = os.path.join(workdir, "stacked_4band_masked.kea")
    rsgislib.imageutils.mask_img(
        stacked_lcc_masked, slope_mask_img, stacked_masked, "KEA",
        rsgislib.TYPE_32FLOAT, 0, 1,
    )
    print("  Applied LCC and slope masks to stacked image")

    valid_mask = os.path.join(workdir, "valid_data_mask.tif")
    gen_valid_mask(stacked_lcc_masked, valid_mask)

    return {
        "slope": slope_re,
        "hand": hand_re,
        "lcc": lcc_re,
        "slope_mask": slope_mask_img,
        "stacked_masked": stacked_masked,
        "valid_mask": valid_mask,
    }


def preprocess_and_segment(args, processing_hh, processing_hv, processing_inc, workdir):
    """Steps 1-6 for a fresh run: calibrate, filter, mask, segment, extract stats."""
    print("[1/10] Preprocessing SAR bands...")
    hh_raw = load_band(processing_hh)
    hv_raw = load_band(processing_hv)

    if args.level == "L11" and not processing_inc:
        sys.exit("ERROR: --inc is required for Level 1.1 data")

    # L1.1 and L2.2 both store amplitude DN: dB = 10*log10(DN^2) - 83
    hh_db = dn_to_db(hh_raw)
    hv_db = dn_to_db(hv_raw)
    print("  Applied DN-to-dB calibration: 10*log10(DN^2) - 83")

    if processing_inc:
        inc_arr = load_incidence(processing_inc)
    else:
        inc_arr = np.full_like(hh_raw, FALLBACK_INCIDENCE_DEG)
        print(f"  Using nominal incidence angle {FALLBACK_INCIDENCE_DEG} deg (ScanSAR centre)")

    ndpi = compute_ndpi(hh_db, hv_db)
    print(f"  HH range : [{np.nanmin(hh_db):.1f}, {np.nanmax(hh_db):.1f}] dB")
    print(f"  HV range : [{np.nanmin(hv_db):.1f}, {np.nanmax(hv_db):.1f}] dB")
    print(f"  NDPI range: [{np.nanmin(ndpi):.3f}, {np.nanmax(ndpi):.3f}]")

    pre_img = os.path.join(workdir, "preprocessed_HH_HV_NDPI.tif")
    save_geotiff(processing_hh, [hh_db, hv_db, ndpi], pre_img)

    print("[2/10] Applying Lee speckle filter (3x3)...")
    filt_img = os.path.join(workdir, "filtered_HH_HV_NDPI.tif")
    apply_lee_filter(pre_img, filt_img)

    inc_img = os.path.join(workdir, "incidence_angle.tif")
    save_geotiff(processing_hh, [inc_arr], inc_img)

    stacked_img = os.path.join(workdir, "stacked_4band.kea")
    stack_bands([filt_img, inc_img], stacked_img)
    print("  Stacked: HH_filt, HV_filt, NDPI_filt, IncAngle -> 4-band KEA")

    prepared = prepare_ancillary_and_masks(args, processing_hh, stacked_img, workdir)
    stacked_masked = prepared["stacked_masked"]

    print("[5/10] Running two-stage Shepherd segmentation on masked image...")
    ds = gdal.Open(stacked_masked)
    hh_filt = ds.GetRasterBand(1).ReadAsArray()
    hv_filt = ds.GetRasterBand(2).ReadAsArray()
    ndpi_filt = ds.GetRasterBand(3).ReadAsArray()
    del ds

    hh_low, hv_low, ndpi_low = create_low_backscatter_image(hh_filt, hv_filt, ndpi_filt)
    has_low_bs = bool(np.any(hv_low != 0))

    low_bs_img = os.path.join(workdir, "low_backscatter.tif")
    save_geotiff(processing_hh, [hh_low, hv_low, ndpi_low], low_bs_img)
    print(f"  Low backscatter pixels (HV < -15 dB): {np.sum(hv_low != 0)}")

    clumps_img = os.path.join(workdir, "clumps_merged.kea")
    segment_tile(stacked_masked, low_bs_img, clumps_img, has_low_bs=has_low_bs)

    print("[6/10] Extracting per-segment statistics from masked image...")
    extract_segment_stats(clumps_img, stacked_masked, prepared["slope"], prepared["hand"])
    return clumps_img, prepared


def reuse_segments(args, processing_hh, workdir):
    """Steps 1-6 when reusing clumps from a previous run on the same grid."""
    reuse_dir = args.reuse_segments
    reuse_clumps = os.path.join(reuse_dir, "clumps_merged.kea")
    reuse_stacked = os.path.join(reuse_dir, "stacked_4band.kea")
    if not os.path.exists(reuse_clumps):
        sys.exit(f"ERROR: Cannot find clumps_merged.kea in reuse dir: {reuse_dir}")
    if not os.path.exists(reuse_stacked):
        sys.exit(f"ERROR: Cannot find stacked_4band.kea in reuse dir: {reuse_dir}")

    reuse_x_res, reuse_y_res = get_pixel_size(reuse_stacked)
    processing_x_res, processing_y_res = get_pixel_size(processing_hh)
    if abs(reuse_x_res - processing_x_res) > 1e-9 or abs(reuse_y_res - processing_y_res) > 1e-9:
        sys.exit(
            "ERROR: reuse-segments grid does not match the current working grid. "
            "Recreate the clumps on the same working grid before reusing them."
        )

    print(f"[1-2/10] SKIPPED — reusing preprocessed data from: {reuse_dir}")
    clumps_img = os.path.join(workdir, "clumps_merged.kea")
    stacked_img = os.path.join(workdir, "stacked_4band.kea")
    if os.path.abspath(reuse_clumps) != os.path.abspath(clumps_img):
        shutil.copy2(reuse_clumps, clumps_img)
    if os.path.abspath(reuse_stacked) != os.path.abspath(stacked_img):
        shutil.copy2(reuse_stacked, stacked_img)

    prepared = prepare_ancillary_and_masks(args, processing_hh, stacked_img, workdir)

    print("[5/10] SKIPPED — reusing segmentation (clumps_merged.kea)")
    print("[6/10] Re-extracting per-segment statistics from new masked image...")
    extract_segment_stats(clumps_img, prepared["stacked_masked"],
                          prepared["slope"], prepared["hand"])
    return clumps_img, prepared


def write_final_output(final_arr, reference_img, final_path):
    """Write the classified raster as a paletted Byte GeoTIFF."""
    ref_ds = gdal.Open(reference_img)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        final_path, ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal.GDT_Byte,
    )
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(final_arr)
    band.SetNoDataValue(0)
    out_ds.FlushCache()
    del band, out_ds, ref_ds

    # Reopen to set the colour table (PhotometricInterpretation can't change mid-write)
    ds = gdal.Open(final_path, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    ct = gdal.ColorTable()
    for cls_val, hex_color in COLOR_TABLE.items():
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        ct.SetColorEntry(cls_val, (r, g, b, 255))
    band.SetRasterColorTable(ct)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    ds.FlushCache()
    del band, ds


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    workdir = os.path.join(args.outdir, "work")
    os.makedirs(workdir, exist_ok=True)

    print("=" * 60)
    print("  RadWet-L Classification Pipeline (Replication)")
    print("=" * 60)
    print(f"  Data level : {args.level}")
    print(f"  HH         : {args.hh}")
    print(f"  HV         : {args.hv}")
    if args.inc:
        print(f"  Inc angle  : {args.inc}")
    print(f"  Slope      : {args.slope}")
    print(f"  HAND       : {args.hand}")
    print(f"  LCC mask   : {args.lcc}")
    print(f"  Work grid  : {args.working_grid}")
    print(f"  Scene prep : {args.scene_mosaic}")
    print(f"  Output     : {args.outdir}")
    print()

    model_dir = Path(args.models)
    water_model = model_dir / "XGBoost_Water.ubj"
    water_scaler = model_dir / "Scaler_Water.pkl"
    flood_model = model_dir / "XGBoost_Flood.ubj"
    flood_scaler = model_dir / "Scaler_Flood.pkl"
    for p in (water_model, water_scaler, flood_model, flood_scaler):
        if not p.exists():
            sys.exit(f"ERROR: Model file not found: {p}")

    input_hh, input_hv, input_inc, scene_groups = prepare_scene_inputs(args, workdir)
    if any(len(group) > 1 for group in scene_groups.values()):
        print("[prep] Auto-mosaicked matching scene inputs:")
        for label, group, resolved in (
            ("HH", scene_groups["hh"], input_hh),
            ("HV", scene_groups["hv"], input_hv),
            ("Inc", scene_groups["inc"], input_inc),
        ):
            if len(group) > 1:
                names = ", ".join(os.path.basename(scene_path) for scene_path in group)
                print(f"  {label:<10}: {names}")
                print(f"  {label} mosaic : {resolved}")
        print()

    processing_hh, processing_hv, processing_inc = prepare_working_inputs(
        input_hh, input_hv, input_inc, args.working_grid, workdir
    )
    if (processing_hh, processing_hv, processing_inc) != (input_hh, input_hv, input_inc):
        print("  Prepared 50m working-grid SAR inputs in work/")
        print()

    if args.reuse_segments:
        clumps_img, prepared = reuse_segments(args, processing_hh, workdir)
    else:
        clumps_img, prepared = preprocess_and_segment(
            args, processing_hh, processing_hv, processing_inc, workdir
        )

    # ---- Step 7: XGBoost classification ----
    print("[7/10] Classifying water and flood forest...")
    water_classes = classify_water(clumps_img, str(water_model), str(water_scaler))
    flood_classes = classify_flood(clumps_img, str(flood_model), str(flood_scaler))
    print(f"  Water segments: {np.sum(water_classes == 2)}, "
          f"Flood segments: {np.sum(flood_classes == 1)}")

    rios.rat.writeColumn(clumps_img, "WaterClass", water_classes)
    rios.rat.writeColumn(clumps_img, "FloodClass", flood_classes)

    water_img = os.path.join(workdir, "classified_water.tif")
    flood_img = os.path.join(workdir, "classified_flood.tif")
    rsgislib.rastergis.export_col_to_gdal_img(
        clumps_img, water_img, "GTIFF", rsgislib.TYPE_8UINT, "WaterClass", rat_band=1,
    )
    rsgislib.rastergis.export_col_to_gdal_img(
        clumps_img, flood_img, "GTIFF", rsgislib.TYPE_8UINT, "FloodClass", rat_band=1,
    )

    # ---- Step 8: Combine classifications ----
    print("[8/10] Combining water and flood classifications...")
    water_arr = load_band(water_img).astype(np.uint8)
    flood_arr = load_band(flood_img).astype(np.uint8)
    lcc_arr = load_band(prepared["lcc"]).astype(np.uint8)
    combined = combine_classifications(water_arr, flood_arr, lcc_arr)

    combined_img = os.path.join(workdir, "classified_combined.tif")
    save_geotiff(processing_hh, [combined.astype(np.float32)], combined_img)

    combined_vm = os.path.join(workdir, "classified_combined_vm.tif")
    mask_image_with_class(combined_img, prepared["valid_mask"], combined_vm,
                          mask_value=0, class_value=0)

    # ---- Step 9: Mode + sieve filtering ----
    print("[9/10] Post-processing (mode filter, sieve)...")
    filtered_img = os.path.join(workdir, "classified_mode_filtered.tif")
    apply_mode_filter(combined_vm, filtered_img)
    sieved_img = os.path.join(workdir, "classified_sieved.tif")
    apply_sieve_filter(filtered_img, sieved_img)

    # ---- Step 10: Re-apply masks and write final output ----
    print("[10/10] Re-applying LCC and slope masks, writing final output...")
    sieved_lcc = os.path.join(workdir, "classified_sieved_lcc.tif")
    mask_image_with_class(sieved_img, prepared["lcc"], sieved_lcc,
                          mask_value=1, class_value=CLASS_NON_FOREST)
    final_masked = os.path.join(workdir, "classified_sieved_lcc_slope.tif")
    mask_image_with_class(sieved_lcc, prepared["slope_mask"], final_masked,
                          mask_value=1, class_value=CLASS_STEEP_SLOPE)

    final_arr = load_band(final_masked).astype(np.uint8)
    final_path = os.path.join(args.outdir, OUTPUT_NAME)
    write_final_output(final_arr, processing_hh, final_path)

    unique, counts = np.unique(final_arr, return_counts=True)
    print()
    print("  Classification summary:")
    for val, cnt in zip(unique, counts):
        pct = 100 * cnt / final_arr.size
        name = CLASS_NAMES.get(int(val), f"Unknown({val})")
        print(f"    {val} {name:15s}: {cnt:>10,} px ({pct:5.1f}%)")

    print(f"\nDone! Output: {final_path}")
    return final_path


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
