"""Build two example GeoParquet files covering the cases the matcher must handle.

Each case lives in its own patch of space (5 km apart) so cases cannot interfere
with one another. Lines are laid out in metres in UTM 32N and then written out in
EPSG:4326, which is the awkward-but-common shape of real input.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

WORK_CRS = "EPSG:32632"
OUT_CRS = "EPSG:4326"
ORIGIN_X, ORIGIN_Y = 500_000.0, 5_600_000.0
CASE_SPACING_M = 5_000.0

# case name -> (expected to be reported as an overlap?, note)
CASES = {
    "identical": (True, "same path, same vertices"),
    "offset_15m": (True, "same path shifted 15 m sideways"),
    "partial": (True, "B only covers the middle of A"),
    "crossing": (False, "different paths that merely cross"),
    "far_apart": (False, "parallel but 200 m away"),
    "reversed": (True, "same path digitized the other way"),
    "noisy": (True, "same path with GPS-like wander"),
    "short_touch": (False, "runs alongside for only ~20 m"),
}


def _line(case: str, coords: list[tuple[float, float]]) -> LineString:
    """Place metre-space coords into this case's own patch of the world."""
    i = list(CASES).index(case)
    ox = ORIGIN_X + i * CASE_SPACING_M
    return LineString([(ox + x, ORIGIN_Y + y) for x, y in coords])


def _straight(case: str, x0, y0, x1, y1, n=50) -> LineString:
    t = np.linspace(0, 1, n)
    return _line(case, list(zip(x0 + t * (x1 - x0), y0 + t * (y1 - y0))))


def build() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    a_rows, b_rows = [], []

    def add(case, geom_a, geom_b):
        a_rows.append({"road_id": f"A-{case}", "case": case, "geometry": geom_a})
        b_rows.append({"road_id": f"B-{case}", "case": case, "geometry": geom_b})

    add("identical", _straight("identical", 0, 0, 1000, 0), _straight("identical", 0, 0, 1000, 0))

    add(
        "offset_15m",
        _straight("offset_15m", 0, 0, 1000, 0),
        _straight("offset_15m", 0, 15, 1000, 15),
    )

    # A runs the full kilometre; B only exists between 300 m and 700 m, 10 m off.
    add(
        "partial",
        _straight("partial", 0, 0, 1000, 0),
        _straight("partial", 300, 10, 700, 10),
    )

    add(
        "crossing",
        _straight("crossing", 0, 0, 1000, 0),
        _straight("crossing", 500, -500, 500, 500),
    )

    add(
        "far_apart",
        _straight("far_apart", 0, 0, 1000, 0),
        _straight("far_apart", 0, 200, 1000, 200),
    )

    rev = _straight("reversed", 0, 0, 1000, 0)
    add("reversed", rev, LineString(list(rev.coords)[::-1]))

    # Same corridor, but B wanders ±12 m the way a GPS trace does.
    xs = np.linspace(0, 1000, 120)
    rng = np.random.default_rng(7)
    wander = 12 * np.sin(xs / 90.0) + rng.normal(0, 2.0, xs.size)
    add(
        "noisy",
        _straight("noisy", 0, 0, 1000, 0),
        _line("noisy", list(zip(xs, wander))),
    )

    # B approaches, skims A for ~20 m, and leaves again — too short to count.
    add(
        "short_touch",
        _straight("short_touch", 0, 0, 1000, 0),
        _line("short_touch", [(400, 150), (490, 8), (510, 8), (600, 150)]),
    )

    a = gpd.GeoDataFrame(a_rows, geometry="geometry", crs=WORK_CRS).to_crs(OUT_CRS)
    b = gpd.GeoDataFrame(b_rows, geometry="geometry", crs=WORK_CRS).to_crs(OUT_CRS)
    return a, b


def main() -> None:
    a, b = build()
    a.to_parquet("example_a.parquet")
    b.to_parquet("example_b.parquet")
    print(f"wrote example_a.parquet ({len(a)} lines) and example_b.parquet ({len(b)} lines)")
    for case, (expected, note) in CASES.items():
        print(f"  {case:12s} overlap={str(expected):5s}  {note}")


if __name__ == "__main__":
    main()
