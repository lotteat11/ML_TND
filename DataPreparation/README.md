# DataPreparation

Builds the merged dataset the model is trained on. Four steps, run in order;
each reads the previous step's parquet and writes its own. The pipeline runner
does the same thing in one command:

```bash
./run_pipeline.sh new dns,tec,msis,merge
```

## 1. Satellite density — `download_dns.py`

Downloads GRACE or Swarm DNS (version 02) products from the TU Delft
Thermosphere Data Centre over HTTPS, parses the ASCII format, and stacks all
years into one parquet. GRACE-A and GRACE-B are concatenated with the
spacecraft identifier retained.

| Variable | Default | Effect |
|---|---|---|
| `DNS_MISSION` | `GRACE` | `GRACE` or `Swarm` |
| `DNS_YEARS` | `2002,2008` | Inclusive year range |
| `DNS_OUTDIR` | `GRACE_0208_v02` | Where the zip files land |
| `DNS_PARQUET_OUT` | `grace_dns_2002_2008.parquet` | Output |

Downloads are resumable: files already on disk are skipped.

## 2. Ionospheric TEC — `download_tec.py`

Downloads CODE GIM IONEX files from NASA CDDIS and parses them into
epoch / latitude / longitude / TEC rows. Requires an Earthdata login in
`~/.netrc`:

```
machine urs.earthdata.nasa.gov login USERNAME password PASSWORD
```

| Variable | Default | Effect |
|---|---|---|
| `TEC_START_YEAR` | `2009` | First year |
| `TEC_END_YEAR` | `2017` | Last year |
| `TEC_OUT_DIR` | `ionex_files_0917_v4` | Download folder, also read from here |

## 3. NRLMSIS-2.1 baseline — `run_pymsis.py` and `run_pymsis_swarm.py`

Runs NRLMSIS-2.1 through PyMSIS at every observation point and adds the
`msis_rho` column. PyMSIS fetches F10.7, F10.7a, daily Ap and the 3-hourly ap
history itself. `pymsis_utils.py` holds the Dask loaders and the hourly
interpolation of the space-weather indices.

The Swarm variant also runs MSIS at a fixed 400 km reference altitude, which
is what the off-track validation uses to scale Swarm densities to the GRACE
orbit.

| Variable | Default | Effect |
|---|---|---|
| `PYMSIS_INPUT` | `grace_dns_2009_2016.parquet` | Step-1 output |
| `PYMSIS_OUTPUT` | `grace_dns_with_tnd_y200916_v4_0809.parquet` | Output |
| `PYMSIS_TIME_MIN` / `_MAX` | `2009-06-06` / `2016-01-01` | Time window kept |

## 4. Merge — `merge_tec_grace.py`

Collocates TEC with each satellite point. For every observation the nearest
TEC epoch within ±3 h is chosen, then the nearest grid cell in 3-D Cartesian
space via a k-d tree, subject to a chord-distance threshold so that a distant
cell is never accepted. Writes the merged dataset with `matched_tec_value`.

| Variable | Default | Effect |
|---|---|---|
| `MERGE_GRACE_PARQUET` | step-3 output | Input densities |
| `MERGE_TEC_PARQUET` | `tec_codg_2009-2017_doy1-365_v2.parquet` | Input TEC |
| `MERGE_OUTPUT` | `grace_data_merged_v3.parquet` | Output |
| `MERGE_TIME_MAX` | `2016-01-01` | Rows after this are dropped |
| `MERGE_CHUNKED` | unset | `1` merges year by year — required for the full 2002–2017 mission on a 24 GB machine |
| `MERGE_YEARS` | unset | Restrict a chunked merge to these years |

`diagnose_tec_missing.py` reports rows without a TEC match and separates
absent IONEX epochs from failed spatial matches.

## Figure

`plot_density_altitude.py` draws the weekly observed and MSIS density and the
GRACE altitude decay across the mission (manuscript Fig. 2). It streams the
parquet in batches, so the full mission does not need to fit in memory.
