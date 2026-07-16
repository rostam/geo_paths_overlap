"""Print what a parquet actually contains: columns, geometry, CRS, id candidates.

Useful before a first run against unfamiliar data:

    python inspect_inputs.py a.parquet b.parquet
"""

from __future__ import annotations

import sys

import geopandas as gpd
import pandas as pd


def describe(path: str) -> None:
    print(f"\n=== {path} ===")
    try:
        gdf = gpd.read_parquet(path)
        print("read as: GeoParquet (has geo metadata)")
        print(f"geometry column: {gdf.geometry.name}")
        crs = gdf.crs
        print(f"CRS: {crs.name if crs else None} ({crs.to_string() if crs else '-'})")
        print(f"geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
    except Exception as err:
        print(f"read as: plain parquet (gpd.read_parquet failed: {err})")
        gdf = pd.read_parquet(path)

    print(f"rows: {len(gdf)}")
    print("columns:")
    for col in gdf.columns:
        sample = gdf[col].dropna().head(1)
        val = sample.iloc[0] if not sample.empty else None
        if isinstance(val, (bytes, bytearray)):
            preview = f"<{len(val)} bytes WKB>"
        else:
            preview = str(val)[:60]
        nunique = gdf[col].nunique() if gdf[col].dtype != "geometry" else "-"
        print(f"  {str(col):40s} {str(gdf[col].dtype):12s} unique={nunique!s:8s} e.g. {preview}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        describe(path)


if __name__ == "__main__":
    main()
