import geopandas as gpd
import pytest

from make_examples import CASES, build
from overlap import OverlapParams, find_overlaps

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


def test_no_overlaps_still_produces_a_valid_empty_frame(tmp_path):
    a, b = build()
    empty = find_overlaps(a, b.iloc[0:0], PARAMS, id_col_a="road_id", id_col_b="road_id")
    assert len(empty) == 0
    path = tmp_path / "empty.parquet"
    empty.to_parquet(path)
    assert len(gpd.read_parquet(path)) == 0
