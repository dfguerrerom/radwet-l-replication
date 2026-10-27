"""Download the published RadWet-L products and compare a run against them.

The products are Oakes (2024), Zenodo record 10566417 (CC BY 4.0):
43 Amazon-wide classifications for orbit cycles 119-240 (2019-2023),
about 125 MB each.

Usage (from the repository root):

    # download the products for specific cycles (or all, if --cycle is omitted)
    python -m radwet_l.zenodo download --out zenodo --cycle 119 226

    # compare a pipeline output with the matching product
    python -m radwet_l.zenodo compare --ours runs/C226/classified.tif \
        --cycle 226 --zenodo-dir zenodo
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from osgeo import gdal

from .config import CLASS_NAMES

RECORD = "10566417"
API = f"https://zenodo.org/api/records/{RECORD}"
PRODUCT_RE = re.compile(r"^Classified_Output_Orbit-Cycle_(\d+)-.*\.tif$")
EXTRA_FILES = ("Merge_QA_Points.geojson", "ClassifiedProductsMetadata.txt")
COMPARED_CLASSES = (1, 2, 4, 5, 6)


def fetch_filelist():
    with urllib.request.urlopen(API, timeout=60) as r:
        return json.load(r).get("files", [])


def download(out, cycles=None, include_extras=False):
    """Download products (optionally only some cycles). Resumable."""
    out.mkdir(parents=True, exist_ok=True)
    wanted = []
    for f in fetch_filelist():
        match = PRODUCT_RE.match(f["key"])
        if match and (not cycles or int(match.group(1)) in cycles):
            wanted.append(f)
        elif include_extras and f["key"] in EXTRA_FILES:
            wanted.append(f)
    print(f"{len(wanted)} file(s) to fetch from Zenodo record {RECORD}")

    failed = 0
    for f in wanted:
        key, size, url = f["key"], f.get("size", 0), f["links"]["self"]
        dst = out / key
        if dst.exists() and dst.stat().st_size == size:
            print(f"[skip] {key}")
            continue
        print(f"[get ] {key} ({size / 1e6:.0f} MB)")
        rc = subprocess.call(["curl", "-fL", "-C", "-", "--retry", "5",
                              "--retry-delay", "5", "-o", str(dst), url])
        if rc != 0 or not dst.exists() or dst.stat().st_size != size:
            failed += 1
            print(f"[FAIL] {key} (curl exit {rc})", file=sys.stderr)
    return 1 if failed else 0


def find_product(zenodo_dir, cycle):
    for p in sorted(Path(zenodo_dir).glob("Classified_Output_Orbit-Cycle_*.tif")):
        match = PRODUCT_RE.match(p.name)
        if match and int(match.group(1)) == cycle:
            return p
    return None


def read_on_grid(path, reference):
    """Read ``path`` warped (nearest neighbour) onto the grid of ``reference``."""
    ref = gdal.Open(str(reference))
    gt = ref.GetGeoTransform()
    nx, ny = ref.RasterXSize, ref.RasterYSize
    bounds = (gt[0], gt[3] + ny * gt[5], gt[0] + nx * gt[1], gt[3])
    ds = gdal.Warp("", str(path), format="MEM", outputBounds=bounds,
                   width=nx, height=ny, dstSRS=ref.GetProjection(),
                   resampleAlg="near")
    arr = ds.GetRasterBand(1).ReadAsArray()
    return arr


def class_percentages(arr):
    """Percent of each class among classified pixels (value 0 = no data)."""
    valid = arr > 0
    total = valid.sum()
    return {c: (100.0 * (arr == c).sum() / total if total else 0.0)
            for c in COMPARED_CLASSES}


def compare(ours_path, zenodo_path):
    """Print per-class percentages and pixel agreement; return the agreement %."""
    ds = gdal.Open(str(ours_path))
    ours = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    zen = read_on_grid(zenodo_path, ours_path)

    ours_pct = class_percentages(ours)
    zen_pct = class_percentages(zen)
    valid = (ours != 0) & (zen != 0)
    agreement = 100.0 * (ours[valid] == zen[valid]).sum() / valid.sum() if valid.any() else float("nan")

    print(f"ours:   {ours_path}")
    print(f"Zenodo: {zenodo_path}\n")
    print(f"  {'Class':<16} {'Ours %':>8} {'Zenodo %':>9} {'Δ pp':>7}")
    for c in COMPARED_CLASSES:
        print(f"  {CLASS_NAMES[c]:<16} {ours_pct[c]:8.2f} {zen_pct[c]:9.2f} "
              f"{ours_pct[c] - zen_pct[c]:+7.2f}")
    print(f"\n  Pixel agreement (pixels classified in both): {agreement:.2f}%")
    return agreement


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m radwet_l.zenodo",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download published products")
    dl.add_argument("--out", type=Path, required=True, help="Destination folder")
    dl.add_argument("--cycle", type=int, nargs="+",
                    help="Only these orbit cycles (default: all 43)")
    dl.add_argument("--extras", action="store_true",
                    help="Also fetch the QA points and the metadata file")

    cmp_ = sub.add_parser("compare", help="Compare a pipeline output with a product")
    cmp_.add_argument("--ours", type=Path, required=True,
                      help="Pipeline output (classified.tif)")
    product = cmp_.add_mutually_exclusive_group(required=True)
    product.add_argument("--zenodo", type=Path, help="Path to the Zenodo product")
    product.add_argument("--cycle", type=int,
                         help="Orbit cycle; the product is looked up in --zenodo-dir")
    cmp_.add_argument("--zenodo-dir", type=Path, default=Path("zenodo"),
                      help="Folder with downloaded products. Default: zenodo/")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "download":
        return download(args.out, set(args.cycle or ()), args.extras)

    zenodo_path = args.zenodo or find_product(args.zenodo_dir, args.cycle)
    if zenodo_path is None:
        sys.exit(f"No product for cycle {args.cycle} in {args.zenodo_dir}. "
                 f"Download it with: python -m radwet_l.zenodo download "
                 f"--out {args.zenodo_dir} --cycle {args.cycle}")
    compare(args.ours, zenodo_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
