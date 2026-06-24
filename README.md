# Mediterranean Sea — default dataset

Place these 6 files in this folder. Each one can be either a plain `.csv`
**or** a gzip-compressed `.csv.gz` — the app checks for both automatically,
so just pick whichever fits under GitHub's upload size comfortably (gzip
typically shrinks wave time-series CSVs to a third or less of their size).

| Base name      | Accepted filenames                          | Description |
|---|---|---|
| `baseline_swh` | `baseline_swh.csv` or `baseline_swh.csv.gz` | Baseline (2005) significant wave height |
| `baseline_tm`  | `baseline_tm.csv` or `baseline_tm.csv.gz`   | Baseline (2005) mean wave period |
| `rcp45_swh`    | `rcp45_swh.csv` or `rcp45_swh.csv.gz`       | RCP 4.5 projection (2041–2100), SWH |
| `rcp45_tm`     | `rcp45_tm.csv` or `rcp45_tm.csv.gz`         | RCP 4.5 projection (2041–2100), Tm |
| `rcp85_swh`    | `rcp85_swh.csv` or `rcp85_swh.csv.gz`       | RCP 8.5 projection (2041–2100), SWH |
| `rcp85_tm`     | `rcp85_tm.csv` or `rcp85_tm.csv.gz`         | RCP 8.5 projection (2041–2100), Tm |

Files must use a `time` column, plus `swh` or `tm`/`Hs`/`Tp`-style columns
(see `analysis.py`'s `standardize_column_names` for accepted header variants).

**To compress a file before uploading**, on macOS/Linux:
```bash
gzip -k baseline_swh.csv      # creates baseline_swh.csv.gz, keeps the original
```
On Windows, right-click the CSV → "Send to" → "Compressed (zipped) folder"
won't produce a `.gz` (it makes a `.zip`, which pandas won't auto-read here).
Use 7-Zip with the "gzip" format instead, or run the command above in WSL/Git Bash.

If any of the 6 base names is missing in both forms, the app will tell the
user exactly which one, and disable the "Load Mediterranean Sea dataset"
button until all 6 are present.
