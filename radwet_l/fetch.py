"""Download RadWet-L inputs for an area and orbit cycle(s) from Google Earth Engine.

For each cycle, all ALOS-2 PALSAR-2 ScanSAR Level 2.2 passes over the area are
combined into one per-pixel *angle-optimal* HH/HV/LIN mosaic: each pixel comes
from the pass whose local incidence angle is closest to the ScanSAR nominal
37 degrees. Averaging passes instead mixes swath geometries and biases the
classification (see docs/findings.md).

The static ancillary layers are exported once per output folder, using the
sources listed in the paper (Table 1):

  - slope: WWF HydroSHEDS hydrologically conditioned DEM (100 m)
  - HAND:  MERIT Hydro v1.0.1 (90 m)
  - forest mask: Copernicus Global Land Cover (non-forest classes) OR
    Hansen GFC loss after 2016 -> 1 = non-forest, 0 = forest

Exports go to Google Drive and are downloaded locally (the Earth Engine
credentials need Drive scope; see the README).

Usage (from the repository root):

    python -m radwet_l.fetch --aoi data/aoi/loreto_test_area.geojson \
        --cycle 226 --out inputs/loreto --project my-ee-project

    python -m radwet_l.fetch --bbox -74.06 -4.46 -73.21 -3.78 \
        --cycle 119 122 --out inputs/site

Output layout:

    <out>/aux/Slope_HydroSHEDS_100m.tif
    <out>/aux/HAND_MERITHydro_90m.tif
    <out>/aux/ForestMask_Copernicus_Hansen.tif
    <out>/C226/C226_angleopt.tif                 3-band HH, HV, LIN
    <out>/C226/C226_angleopt_{HH,HV,LIN}.tif     single bands for the pipeline
"""
import argparse
import datetime as dt
import json
import os
import shlex
import sys
from pathlib import Path

import ee
from osgeo import gdal, ogr, osr

from .config import NOMINAL_INCIDENCE_DEG
from .cycles import cycle_dates
from .gee import export_and_download_batch, init_gee

SCANSAR_COLLECTION = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
SAR_SCALE_M = 25
EXPORT_CRS = "EPSG:4326"

AUX_FILES = {
    "slope": "Slope_HydroSHEDS_100m.tif",
    "hand": "HAND_MERITHydro_90m.tif",
    "lcc": "ForestMask_Copernicus_Hansen.tif",
}


# --- area of interest --------------------------------------------------------

def load_aoi(path):
    """Read a vector file (GeoJSON, shapefile, or zipped shapefile) as an ee.Geometry.

    All features are unioned and reprojected to EPSG:4326.
    """
    ogr.UseExceptions()
    src = str(path)
    if src.lower().endswith(".zip"):
        src = f"/vsizip/{os.path.abspath(src)}"
    ds = ogr.Open(src)
    layer = ds.GetLayer(0)

    union = None
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        union = geom.Clone() if union is None else union.Union(geom)
    if union is None:
        raise ValueError(f"No geometries found in {path}")

    source_srs = layer.GetSpatialRef()
    if source_srs is not None:
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not source_srs.IsSame(wgs84):
            union.Transform(osr.CoordinateTransformation(source_srs, wgs84))

    return ee.Geometry(json.loads(union.ExportToJson()))


# --- server-side image builders ----------------------------------------------

def scansar_passes(start, end, region):
    """ScanSAR L2.2 images over ``region`` between two inclusive dates."""
    end_exclusive = end + dt.timedelta(days=1)
    return (ee.ImageCollection(SCANSAR_COLLECTION)
            .filterDate(start.isoformat(), end_exclusive.isoformat())
            .filterBounds(region))


def build_angleopt_sar(passes):
    """Per-pixel angle-optimal HH/HV/LIN mosaic of a collection of passes."""
    def add_quality(img):
        # LIN is the local incidence angle in degrees x 100.
        distance = img.select("LIN").divide(100).subtract(NOMINAL_INCIDENCE_DEG).abs()
        return img.addBands(distance.multiply(-1).rename("QUAL"))

    mosaic = (passes.select(["HH", "HV", "LIN"])
              .map(add_quality)
              .qualityMosaic("QUAL")
              .select(["HH", "HV", "LIN"]))
    return mosaic.toUint16()


def build_slope(region):
    dem = ee.Image("WWF/HydroSHEDS/03CONDEM").select("b1")
    return ee.Terrain.slope(dem).clip(region)


def build_hand(region):
    return ee.Image("MERIT/Hydro/v1_0_1").select("hnd").clip(region)


def build_forest_mask(region):
    """1 = non-forest (masked), 0 = forest, following paper Sec. 2.3.4-2.3.5."""
    cgls = (ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
            .sort("system:time_start", False).first()
            .select("discrete_classification"))
    # Herbaceous (30), cultivated (40), urban (50), bare/sparse (60)
    unsuitable = cgls.eq(30).Or(cgls.eq(40)).Or(cgls.eq(50)).Or(cgls.eq(60))
    lossyear = ee.Image("UMD/hansen/global_forest_change_2024_v1_12").select("lossyear")
    deforested = lossyear.gt(16)
    return unsuitable.Or(deforested).rename("lcc_mask").toByte().clip(region)


# --- local helpers -------------------------------------------------------------

def split_bands(combined):
    """Split the 3-band SAR export into single-band HH / HV / LIN GeoTIFFs."""
    out = {}
    for index, name in ((1, "HH"), (2, "HV"), (3, "LIN")):
        dst = combined.with_name(f"{combined.stem}_{name}.tif")
        if not dst.exists():
            gdal.Translate(str(dst), str(combined), bandList=[index])
        out[name] = dst
    return out


def pipeline_command(bands, aux_dir, outdir):
    args = [
        "python", "-m", "radwet_l.pipeline",
        "--hh", bands["HH"], "--hv", bands["HV"], "--inc", bands["LIN"],
        "--slope", aux_dir / AUX_FILES["slope"],
        "--hand", aux_dir / AUX_FILES["hand"],
        "--lcc", aux_dir / AUX_FILES["lcc"],
        "--level", "L22",
        "--outdir", outdir,
    ]
    return " ".join(shlex.quote(str(a)) for a in args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m radwet_l.fetch",
        description="Export RadWet-L inputs (angle-optimal ALOS-2 mosaic + "
                    "ancillary layers) from Google Earth Engine.",
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--aoi", type=Path,
                       help="Area of interest: GeoJSON, shapefile or zipped shapefile")
    where.add_argument("--bbox", type=float, nargs=4,
                       metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                       help="Area of interest as a lon/lat bounding box")
    parser.add_argument("--cycle", type=int, nargs="+", required=True,
                        help="ALOS-2 orbit cycle number(s), see data/cycle_dates.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    parser.add_argument("--project",
                        help="Earth Engine Cloud project (default: $EE_PROJECT or "
                             "the project set with `earthengine set_project`)")
    parser.add_argument("--drive-folder", default="radwet_l",
                        help="Google Drive folder used for the exports. Default: radwet_l")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    init_gee(args.project)

    if args.aoi:
        region = load_aoi(args.aoi).bounds()
    else:
        region = ee.Geometry.Rectangle(args.bbox)

    tag = args.out.name or "radwet_l"
    aux_dir = args.out / "aux"
    aux_dir.mkdir(parents=True, exist_ok=True)

    exports = []
    aux_specs = {
        "slope": (build_slope, 100),
        "hand": (build_hand, 90),
        "lcc": (build_forest_mask, 100),
    }
    for key, (builder, scale) in aux_specs.items():
        path = aux_dir / AUX_FILES[key]
        if path.exists():
            print(f"Ancillary layer present, skipping: {path}")
            continue
        name = f"{tag}_{Path(AUX_FILES[key]).stem}"
        exports.append({
            "image": builder(region),
            "description": name,
            "filename": name,
            "output_path": str(path),
            "scale": scale,
            "region": region,
            "crs": EXPORT_CRS,
        })

    cycle_files = {}
    for cycle in args.cycle:
        start, end = cycle_dates(cycle)
        cycle_dir = args.out / f"C{cycle}"
        combined = cycle_dir / f"C{cycle}_angleopt.tif"
        cycle_files[cycle] = combined
        if combined.exists():
            print(f"C{cycle}: SAR mosaic present, skipping: {combined}")
            continue

        passes = scansar_passes(start, end, region)
        dates = sorted(set(passes.aggregate_array("system:time_start")
                           .map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()))
        if not dates:
            print(f"C{cycle} ({start} to {end}): no ScanSAR passes over this area, skipping")
            del cycle_files[cycle]
            continue
        print(f"C{cycle} ({start} to {end}): {len(dates)} pass date(s): {', '.join(dates)}")

        cycle_dir.mkdir(parents=True, exist_ok=True)
        name = f"{tag}_C{cycle}_angleopt"
        exports.append({
            "image": build_angleopt_sar(passes),
            "description": name,
            "filename": name,
            "output_path": str(combined),
            "scale": SAR_SCALE_M,
            "region": region,
            "crs": EXPORT_CRS,
        })

    if exports:
        export_and_download_batch(exports, folder=args.drive_folder)

    missing = [p for p in (aux_dir / f for f in AUX_FILES.values()) if not p.exists()]
    missing += [p for p in cycle_files.values() if not p.exists()]
    if missing:
        print("\nSome files were not downloaded:")
        for p in missing:
            print(f"  {p}")
        return 1

    print("\nInputs ready. Run the classifier with:\n")
    for cycle, combined in cycle_files.items():
        bands = split_bands(combined)
        print(pipeline_command(bands, aux_dir, Path("runs") / f"C{cycle}") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
