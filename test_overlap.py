import warnings

import geopandas as gpd
import pandas as pd
import pytest

from make_examples import CASES, build
from overlap import DEFAULT_PLAIN_CRS, OverlapParams, find_overlaps, load_lines

PARAMS = OverlapParams(tolerance_m=20.0, min_overlap_m=50.0)


@pytest.fixture(scope="module")
def result():
    a, b = build()
    return find_overlaps(a, b, PARAMS, id_col_a="road_id", id_col_b="road_id")


def _for_case(result, case):
    return result[result["id_a"] == f"A-{case}"]


@pytest.mark.parametrize("case", list(CASES))
def test_case_detected_as_expected(result, case):
    expected, note = CASES[case]
    found = len(_for_case(result, case)) > 0
    assert found is expected, f"{case} ({note}): expected overlap={expected}"


def test_identical_covers_whole_line(result):
    row = _for_case(result, "identical").iloc[0]
    assert row["frac_a"] > 0.99
    assert row["mean_offset_m"] < 0.1


def test_offset_line_still_matches_with_right_offset(result):
    row = _for_case(result, "offset_15m").iloc[0]
    assert row["frac_a"] > 0.99
    assert row["mean_offset_m"] == pytest.approx(15.0, abs=0.5)


def test_partial_overlap_reports_only_the_shared_stretch(result):
    # B spans 300-700 m at a 10 m offset, so points on A up to sqrt(20^2 - 10^2)
    # ~= 17 m beyond each end of B are still within tolerance of B's endpoint.
    # The reported stretch therefore overshoots B slightly at both ends.
    row = _for_case(result, "partial").iloc[0]
    assert row["overlap_length_m"] == pytest.approx(400.0, abs=40.0)
    assert row["frac_a"] == pytest.approx(0.4, abs=0.05)
    assert row["a_start_m"] == pytest.approx(300.0, abs=20.0)
    assert row["a_end_m"] == pytest.approx(700.0, abs=20.0)


def test_reversed_line_matches_unless_direction_is_enforced():
    a, b = build()
    directional = find_overlaps(
        a, b, OverlapParams(directional=True), id_col_a="road_id", id_col_b="road_id"
    )
    assert len(directional[directional["id_a"] == "A-reversed"]) == 0
    assert len(directional[directional["id_a"] == "A-identical"]) == 1


def test_noisy_trace_matches_most_of_the_path(result):
    row = _for_case(result, "noisy").iloc[0]
    assert row["frac_a"] > 0.8


def test_tolerance_controls_whether_far_lines_match():
    a, b = build()
    loose = find_overlaps(
        a, b, OverlapParams(tolerance_m=250.0), id_col_a="road_id", id_col_b="road_id"
    )
    assert len(loose[loose["id_a"] == "A-far_apart"]) == 1


def test_result_roundtrips_through_geoparquet(result, tmp_path):
    path = tmp_path / "overlaps.parquet"
    result.to_parquet(path)
    back = gpd.read_parquet(path)
    assert len(back) == len(result)
    assert back.crs == result.crs
    assert back.geometry.geom_type.eq("LineString").all()


@pytest.fixture(scope="module")
def parquet_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    a, b = build()
    a.to_parquet(d / "a.parquet")
    b.to_parquet(d / "b.parquet")
    return d / "a.parquet", d / "b.parquet"


def test_accepts_parquet_paths(parquet_pair, result):
    path_a, path_b = parquet_pair
    from_paths = find_overlaps(
        str(path_a), str(path_b), PARAMS, id_col_a="road_id", id_col_b="road_id"
    )
    assert list(from_paths["id_a"]) == list(result["id_a"])
    assert from_paths["overlap_length_m"].tolist() == pytest.approx(
        result["overlap_length_m"].tolist()
    )


def test_accepts_a_path_and_a_frame_together(parquet_pair, result):
    path_a, _ = parquet_pair
    _, gdf_b = build()
    mixed = find_overlaps(
        path_a, gdf_b, PARAMS, id_col_a="road_id", id_col_b="road_id"
    )
    assert list(mixed["id_a"]) == list(result["id_a"])


def test_missing_file_and_bad_type_are_reported_clearly(parquet_pair):
    path_a, _ = parquet_pair
    with pytest.raises(FileNotFoundError):
        find_overlaps(path_a, "does_not_exist.parquet", PARAMS)
    with pytest.raises(TypeError):
        find_overlaps(path_a, 42, PARAMS)


def _plain_parquet(gdf, path, col="complete_cable_geometry", as_wkt=False):
    """Write a parquet with no geo metadata, geometry as a plain WKB/WKT column."""
    df = pd.DataFrame(gdf.drop(columns=[gdf.geometry.name]))
    df[col] = gdf.geometry.to_wkt() if as_wkt else gdf.geometry.to_wkb()
    df.to_parquet(path)
    return path


def test_plain_parquet_with_wkb_column_is_loaded(tmp_path):
    _, gdf_b = build()
    path = _plain_parquet(gdf_b, tmp_path / "plain_wkb.parquet")
    loaded = load_lines(path, crs="EPSG:4326")
    assert loaded.crs == gdf_b.crs
    assert loaded.geometry.geom_type.eq("LineString").all()
    assert len(loaded) == len(gdf_b)


def test_plain_parquet_with_wkt_column_is_loaded(tmp_path):
    _, gdf_b = build()
    path = _plain_parquet(gdf_b, tmp_path / "plain_wkt.parquet", as_wkt=True)
    loaded = load_lines(path, crs="EPSG:4326")
    assert loaded.geometry.geom_type.eq("LineString").all()


def test_plain_parquet_matches_a_geoparquet_side(tmp_path, result):
    gdf_a, gdf_b = build()
    path_b = _plain_parquet(gdf_b, tmp_path / "b_plain.parquet")
    mixed = find_overlaps(
        gdf_a,
        load_lines(path_b, crs="EPSG:4326"),
        PARAMS,
        id_col_a="road_id",
        id_col_b="road_id",
    )
    assert list(mixed["id_a"]) == list(result["id_a"])


def test_plain_parquet_defaults_to_the_house_crs_and_warns(tmp_path):
    _, gdf_b = build()
    path = _plain_parquet(gdf_b.to_crs(DEFAULT_PLAIN_CRS), tmp_path / "default_crs.parquet")
    with pytest.warns(UserWarning, match="records no CRS"):
        loaded = load_lines(path)
    assert loaded.crs == DEFAULT_PLAIN_CRS


def test_explicit_crs_overrides_the_default_and_is_quiet(tmp_path):
    _, gdf_b = build()
    path = _plain_parquet(gdf_b, tmp_path / "explicit_crs.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an explicit CRS must not warn
        loaded = load_lines(path, crs="EPSG:4326")
    assert loaded.crs == "EPSG:4326"


def test_crs_b_parameter_reaches_a_plain_parquet(tmp_path, result):
    gdf_a, gdf_b = build()
    path_b = _plain_parquet(gdf_b, tmp_path / "b_crs_param.parquet")
    matched = find_overlaps(
        gdf_a, path_b, PARAMS, id_col_a="road_id", id_col_b="road_id", crs_b="EPSG:4326"
    )
    assert list(matched["id_a"]) == list(result["id_a"])


def test_wrong_crs_finds_nothing(tmp_path):
    """A CRS mismatch is silent in the data but obvious in the result."""
    gdf_a, gdf_b = build()
    path_b = _plain_parquet(gdf_b, tmp_path / "b_wrong.parquet")
    # The file is really lat/lon; claiming Gauss-Kruger metres puts it elsewhere.
    wrong = find_overlaps(
        gdf_a, path_b, PARAMS, id_col_a="road_id", id_col_b="road_id",
        crs_b=DEFAULT_PLAIN_CRS,
    )
    assert len(wrong) == 0


def test_explicit_geometry_col_is_used(tmp_path):
    _, gdf_b = build()
    path = _plain_parquet(gdf_b, tmp_path / "named_col.parquet", col="cable_wkt", as_wkt=True)
    loaded = load_lines(path, geometry_col="cable_wkt", crs="EPSG:4326")
    assert loaded.geometry.geom_type.eq("LineString").all()


def test_no_overlaps_still_produces_a_valid_empty_frame(tmp_path):
    a, b = build()
    empty = find_overlaps(a, b.iloc[0:0], PARAMS, id_col_a="road_id", id_col_b="road_id")
    assert len(empty) == 0
    path = tmp_path / "empty.parquet"
    empty.to_parquet(path)
    assert len(gpd.read_parquet(path)) == 0
