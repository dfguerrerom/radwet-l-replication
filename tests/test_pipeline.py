"""Tests for run_pipeline CLI defaults."""


from radwet_l.config import DEFAULT_WORKING_GRID
from radwet_l.pipeline import parse_args


def test_parse_args_defaults_to_50m_working_grid():
    """Inference should default to the paper-matched 50m working grid."""
    args = parse_args([
        "--hh", "hh.tif",
        "--hv", "hv.tif",
        "--slope", "slope.tif",
        "--hand", "hand.tif",
        "--lcc", "lcc.tif",
        "--level", "L22",
        "--outdir", "out",
    ])

    assert args.working_grid == DEFAULT_WORKING_GRID == "50m"


def test_parse_args_allows_native_working_grid():
    """Native-grid runs remain available for 25m diagnostic experiments."""
    args = parse_args([
        "--hh", "hh.tif",
        "--hv", "hv.tif",
        "--slope", "slope.tif",
        "--hand", "hand.tif",
        "--lcc", "lcc.tif",
        "--level", "L22",
        "--outdir", "out",
        "--working-grid", "native",
    ])

    assert args.working_grid == "native"


def test_parse_args_defaults_to_auto_scene_mosaic():
    """Single-pass runs should auto-detect sibling scenes unless disabled."""
    args = parse_args([
        "--hh", "hh_0.tif",
        "--hv", "hv_0.tif",
        "--slope", "slope.tif",
        "--hand", "hand.tif",
        "--lcc", "lcc.tif",
        "--level", "L22",
        "--outdir", "out",
    ])

    assert args.scene_mosaic == "auto"


def test_parse_args_allows_disabling_scene_mosaic():
    """Users can still force a strict single-scene run for diagnostics."""
    args = parse_args([
        "--hh", "hh_0.tif",
        "--hv", "hv_0.tif",
        "--slope", "slope.tif",
        "--hand", "hand.tif",
        "--lcc", "lcc.tif",
        "--level", "L22",
        "--outdir", "out",
        "--scene-mosaic", "off",
    ])

    assert args.scene_mosaic == "off"
