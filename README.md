# Fuzzy path overlap between two GeoParquet linestring files

Finds stretches where a line in file A and a line in file B follow the same path,
even when the two are offset from each other by some distance.

```bash
.venv/bin/python make_examples.py                 # writes example_a/b.parquet
.venv/bin/python overlap.py example_a.parquet example_b.parquet overlaps.parquet \
    --id-col-a road_id --id-col-b road_id
.venv/bin/python -m pytest test_overlap.py
```

## How matching works

Line A is sampled every `--step-m` metres. A sample counts as overlapping when it
is within `--tolerance-m` of line B *and* the two lines point in a similar
direction there (`--max-bearing-diff-deg`). The direction test is what stops a
road that merely crosses another from being reported. Neighbouring matched
samples are joined into runs, gaps shorter than `--max-gap-m` are bridged, and
runs shorter than `--min-overlap-m` are dropped.

Inputs are reprojected to a metric CRS (UTM is estimated if the input is
lat/lon), so all tolerances are real metres. Results come back in A's CRS.

By default a line digitized in the opposite direction still matches; pass
`--directional` if a reversed line should not count.

## Output

One row per matched stretch. The geometry is that stretch as it runs in A;
`b_geometry_wkt` holds the corresponding stretch of B.

| column | meaning |
| --- | --- |
| `id_a`, `id_b` | the two matched lines |
| `overlap_length_m` | length of the shared stretch |
| `frac_a`, `frac_b` | share of each line covered by it |
| `mean_offset_m`, `max_offset_m` | how far apart the lines run |
| `a_start_m`, `a_end_m` | position of the stretch along A |
| `b_start_m`, `b_end_m` | position of the stretch along B |

## Tuning

`--tolerance-m` is the main dial: it is the largest sideways separation you still
consider "the same path". Raise it for loosely-surveyed data, lower it if
parallel neighbouring roads are being merged. Note that a matched stretch can
overshoot the end of B by up to `sqrt(tolerance^2 - offset^2)`, since points just
past B's endpoint are still within tolerance of it.
