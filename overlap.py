"""Find path overlaps between two sets of linestrings that are near, but not
exactly on top of, each other.

Two lines are treated as running along the same path where they stay within
`tolerance_m` of each other *and* point in a similar direction. The direction
check is what keeps a road that merely crosses another road from being reported
as an overlap: crossing lines are close for a moment, but their bearings differ.

The result is one row per matched stretch, with the geometry of that stretch as
it appears in the left dataset.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, substring
from shapely.strtree import STRtree

DEFAULT_TOLERANCE_M = 20.0
DEFAULT_STEP_M = 5.0
DEFAULT_MAX_BEARING_DIFF_DEG = 30.0
DEFAULT_MIN_OVERLAP_M = 50.0
DEFAULT_MAX_GAP_M = 25.0


@dataclass
class OverlapParams:
    tolerance_m: float = DEFAULT_TOLERANCE_M
    step_m: float = DEFAULT_STEP_M
    max_bearing_diff_deg: float = DEFAULT_MAX_BEARING_DIFF_DEG
    min_overlap_m: float = DEFAULT_MIN_OVERLAP_M
    max_gap_m: float = DEFAULT_MAX_GAP_M
    directional: bool = False  # True = a line running the other way is not a match


PathOrFrame = "str | os.PathLike | gpd.GeoDataFrame"

# Plain parquets record no CRS, so one has to be assumed. Gauss-Kruger zone 3
# (metres, Germany) is the house default; override with crs= per call.
DEFAULT_PLAIN_CRS = "EPSG:31467"

_GEOM_NAME_HINTS = ("geometry", "geom", "shape", "wkb", "wkt", "linestring")


def _guess_geometry_col(df: pd.DataFrame) -> str:
    """Find the column holding WKB/WKT geometry in a plain parquet."""
    named = [c for c in df.columns if any(h in str(c).lower() for h in _GEOM_NAME_HINTS)]
    candidates = named or list(df.columns)
    for col in candidates:
        sample = df[col].dropna().head(1)
        if sample.empty:
            continue
        val = sample.iloc[0]
        if isinstance(val, (bytes, bytearray)):
            return col
        if isinstance(val, str) and val.strip().upper().startswith(
            ("LINESTRING", "MULTILINESTRING", "SRID=")
        ):
            return col
    raise ValueError(
        "could not find a geometry column; columns are "
        f"{list(df.columns)}. Pass geometry_col= to say which one it is."
    )


def load_lines(src, geometry_col: str | None = None, crs=None) -> gpd.GeoDataFrame:
    """Load linestrings from a GeoParquet, a plain parquet, or a loaded frame.

    Plain parquet files (no geo metadata) are read with pandas and their WKB or
    WKT geometry column is decoded. Those files carry no CRS of their own, so
    `crs` is assumed to be DEFAULT_PLAIN_CRS unless given. Pass `crs=` whenever
    the file is not in that CRS: the numbers are interpreted, not validated, and
    a wrong CRS yields confident but meaningless distances.
    """
    if isinstance(src, gpd.GeoDataFrame):
        gdf = src
    elif isinstance(src, pd.DataFrame):
        gdf = _decode_geometry(src, geometry_col, crs)
    elif isinstance(src, (str, os.PathLike)):
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"no such parquet file: {path}")
        try:
            gdf = gpd.read_parquet(path)
        except ValueError as err:
            if "geo metadata" not in str(err).lower():
                raise
            gdf = _decode_geometry(pd.read_parquet(path), geometry_col, crs)
    else:
        raise TypeError(
            f"expected a GeoDataFrame or a path to a parquet file, got {type(src).__name__}"
        )

    # Only meaningful when the named column is still a column: a decoded plain
    # parquet has already consumed it into the active geometry.
    if geometry_col is not None and geometry_col in gdf.columns:
        gdf = gdf.set_geometry(geometry_col)
    if crs is not None and gdf.crs is None:
        gdf = gdf.set_crs(crs)
    return gdf


def _decode_geometry(df: pd.DataFrame, geometry_col: str | None, crs) -> gpd.GeoDataFrame:
    col = geometry_col or _guess_geometry_col(df)
    values = df[col]
    sample = values.dropna()
    if sample.empty:
        raise ValueError(f"geometry column {col!r} is empty")
    if crs is None:
        crs = DEFAULT_PLAIN_CRS
        warnings.warn(
            f"{col!r} came from a plain parquet, which records no CRS; assuming "
            f"{DEFAULT_PLAIN_CRS}. Pass crs= if that is wrong — distances depend on it.",
            stacklevel=3,
        )
    if isinstance(sample.iloc[0], (bytes, bytearray)):
        geom = gpd.GeoSeries.from_wkb(values, crs=crs)
    else:
        geom = gpd.GeoSeries.from_wkt(values.str.replace(r"^SRID=\d+;", "", regex=True), crs=crs)
    return gpd.GeoDataFrame(df.drop(columns=[col]), geometry=geom, crs=crs)


def _as_gdf(src, geometry_col: str | None = None, crs=None) -> gpd.GeoDataFrame:
    return load_lines(src, geometry_col=geometry_col, crs=crs)


def _to_single_line(geom):
    """Collapse a MultiLineString into one LineString where the parts connect."""
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        # Disjoint parts: keep the longest, which is the best single-path guess.
        return max(merged.geoms, key=lambda g: g.length)
    return geom


def _bearing_at(line: LineString, dist: float, h: float) -> float:
    """Direction of `line` at measure `dist`, in degrees, via a short chord."""
    d0 = max(0.0, dist - h)
    d1 = min(line.length, dist + h)
    if d1 - d0 < 1e-9:
        return 0.0
    p0 = line.interpolate(d0)
    p1 = line.interpolate(d1)
    return np.degrees(np.arctan2(p1.y - p0.y, p1.x - p0.x))


def _bearing_diff(a: float, b: float, directional: bool) -> float:
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    if not directional:
        d = min(d, 180.0 - d)  # digitization direction should not matter
    return d


def _runs(flags: np.ndarray, step_m: float, max_gap_m: float) -> list[tuple[int, int]]:
    """Contiguous True stretches, bridging gaps shorter than `max_gap_m`."""
    max_gap_samples = int(round(max_gap_m / step_m))
    runs: list[list[int]] = []
    for i, ok in enumerate(flags):
        if not ok:
            continue
        if runs and i - runs[-1][1] - 1 <= max_gap_samples:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs]


def _match_pair(line_a: LineString, line_b: LineString, p: OverlapParams) -> list[dict]:
    """Stretches of `line_a` that run alongside `line_b`."""
    if line_a.length < 1e-9 or line_b.length < 1e-9:
        return []

    n = max(2, int(np.ceil(line_a.length / p.step_m)) + 1)
    dists = np.linspace(0.0, line_a.length, n)
    step = dists[1] - dists[0]
    half = max(step, p.step_m) / 2.0

    matched = np.zeros(n, dtype=bool)
    offsets = np.full(n, np.nan)
    b_measures = np.full(n, np.nan)

    for i, d in enumerate(dists):
        pt = line_a.interpolate(d)
        gap = pt.distance(line_b)
        if gap > p.tolerance_m:
            continue
        db = line_b.project(pt)
        diff = _bearing_diff(
            _bearing_at(line_a, d, half),
            _bearing_at(line_b, db, half),
            p.directional,
        )
        if diff > p.max_bearing_diff_deg:
            continue
        matched[i] = True
        offsets[i] = gap
        b_measures[i] = db

    out = []
    for i0, i1 in _runs(matched, step, p.max_gap_m):
        start, end = dists[i0], dists[i1]
        if end - start < p.min_overlap_m:
            continue
        seg = substring(line_a, start, end)
        if seg.is_empty or seg.length < 1e-9:
            continue
        window = slice(i0, i1 + 1)
        off = offsets[window][matched[window]]
        meas = b_measures[window][matched[window]]
        b_start, b_end = float(np.min(meas)), float(np.max(meas))
        out.append(
            {
                "geometry": seg,
                "overlap_length_m": float(seg.length),
                "a_start_m": float(start),
                "a_end_m": float(end),
                "b_start_m": b_start,
                "b_end_m": b_end,
                "frac_a": float(seg.length / line_a.length),
                "frac_b": float((b_end - b_start) / line_b.length),
                "mean_offset_m": float(np.mean(off)),
                "max_offset_m": float(np.max(off)),
                "b_geometry_wkt": substring(line_b, b_start, b_end).wkt,
            }
        )
    return out


def find_overlaps(
    source_a,
    source_b,
    params: OverlapParams | None = None,
    id_col_a: str | None = None,
    id_col_b: str | None = None,
    crs_a=None,
    crs_b=None,
    geometry_col_a: str | None = None,
    geometry_col_b: str | None = None,
) -> gpd.GeoDataFrame:
    """Match every line in `source_a` against every nearby line in `source_b`.

    Each source is either a path to a parquet file (GeoParquet, or plain with a
    WKB/WKT geometry column) or an already-loaded GeoDataFrame; the two can be
    mixed freely. `crs_a`/`crs_b` and `geometry_col_a`/`geometry_col_b` apply
    only to plain parquets, which carry neither.

    Both inputs are projected to a metre-based CRS so that the tolerances mean
    what they say. Geometry in the result is in that same metric CRS reprojected
    back to the left input's original CRS.
    """
    p = params or OverlapParams()
    gdf_a = _as_gdf(source_a, geometry_col=geometry_col_a, crs=crs_a)
    gdf_b = _as_gdf(source_b, geometry_col=geometry_col_b, crs=crs_b)
    if gdf_a.crs is None or gdf_b.crs is None:
        raise ValueError("both inputs need a CRS")

    out_crs = gdf_a.crs
    work_crs = out_crs if out_crs.is_projected else gdf_a.estimate_utm_crs()
    a = gdf_a.to_crs(work_crs)
    b = gdf_b.to_crs(work_crs)

    a_ids = a[id_col_a] if id_col_a else a.index
    b_ids = b[id_col_b] if id_col_b else b.index

    b_lines = [_to_single_line(g) for g in b.geometry]
    tree = STRtree(b_lines)

    rows = []
    for a_pos, geom_a in enumerate(a.geometry):
        line_a = _to_single_line(geom_a)
        for b_pos in tree.query(line_a.buffer(p.tolerance_m)):
            line_b = b_lines[int(b_pos)]
            for rec in _match_pair(line_a, line_b, p):
                rec["id_a"] = a_ids[a_pos]
                rec["id_b"] = b_ids[int(b_pos)]
                rows.append(rec)

    cols = [
        "id_a",
        "id_b",
        "overlap_length_m",
        "frac_a",
        "frac_b",
        "mean_offset_m",
        "max_offset_m",
        "a_start_m",
        "a_end_m",
        "b_start_m",
        "b_end_m",
        "b_geometry_wkt",
        "geometry",
    ]
    if not rows:
        return gpd.GeoDataFrame(
            {c: [] for c in cols}, geometry="geometry", crs=out_crs
        )

    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=work_crs)[cols]
    return result.to_crs(out_crs).sort_values(
        "overlap_length_m", ascending=False
    ).reset_index(drop=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("out")
    ap.add_argument("--tolerance-m", type=float, default=DEFAULT_TOLERANCE_M)
    ap.add_argument("--step-m", type=float, default=DEFAULT_STEP_M)
    ap.add_argument("--max-bearing-diff-deg", type=float, default=DEFAULT_MAX_BEARING_DIFF_DEG)
    ap.add_argument("--min-overlap-m", type=float, default=DEFAULT_MIN_OVERLAP_M)
    ap.add_argument("--max-gap-m", type=float, default=DEFAULT_MAX_GAP_M)
    ap.add_argument("--directional", action="store_true")
    ap.add_argument("--id-col-a")
    ap.add_argument("--id-col-b")
    ap.add_argument("--crs-a", help=f"CRS for a plain parquet A (default {DEFAULT_PLAIN_CRS})")
    ap.add_argument("--crs-b", help=f"CRS for a plain parquet B (default {DEFAULT_PLAIN_CRS})")
    ap.add_argument("--geometry-col-a", help="WKB/WKT column in a plain parquet A")
    ap.add_argument("--geometry-col-b", help="WKB/WKT column in a plain parquet B")
    args = ap.parse_args()

    res = find_overlaps(
        args.file_a,
        args.file_b,
        OverlapParams(
            tolerance_m=args.tolerance_m,
            step_m=args.step_m,
            max_bearing_diff_deg=args.max_bearing_diff_deg,
            min_overlap_m=args.min_overlap_m,
            max_gap_m=args.max_gap_m,
            directional=args.directional,
        ),
        id_col_a=args.id_col_a,
        id_col_b=args.id_col_b,
        crs_a=args.crs_a,
        crs_b=args.crs_b,
        geometry_col_a=args.geometry_col_a,
        geometry_col_b=args.geometry_col_b,
    )
    res.to_parquet(args.out)
    print(f"{len(res)} overlap(s) written to {args.out}")
    if len(res):
        print(
            res[["id_a", "id_b", "overlap_length_m", "frac_a", "mean_offset_m"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
