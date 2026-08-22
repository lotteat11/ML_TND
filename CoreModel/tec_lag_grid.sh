#!/usr/bin/env bash
# Hyperparameter sweep for tec_lag_sensitivity.py.
#
# Runs every (learning rate x tree depth) combination with a CONSTANT learning
# rate (--lr-decay 1.0 disables the decay schedule) and collects each run's
# results.csv into one combined table.
#
# Usage:
#   CoreModel/tec_lag_grid.sh
#   START=2002-08-01 END=2004-01-01 ROUNDS=3000 CoreModel/tec_lag_grid.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-ven_2404/bin/python}
# Solar maximum: the window the winning parameters were selected on.
START=${START:-2002-01-01}
END=${END:-2004-01-01}
# The production feature set uses a single 3 h lag (vtec_matched_lag). Each
# cell compares three candidates: no_TEC, current_TEC, and current+3h -- the
# last being the variant the CoreModel actually uses. Widen LAGS to re-open
# the lag choice, e.g. LAGS=2500s,3h,6h,24h.
LAGS=${LAGS:-3h}
ROUNDS=${ROUNDS:-3000}
EARLY=${EARLY:-200}
CYCLES=${CYCLES:-9}
GAP=${GAP:-3d}
# Swept like LRS/DEPTHS. Defaults to the single CYCLES value so existing
# invocations behave exactly as before; set CYCLESET="7 9" to vary it.
CYCLESET=${CYCLESET:-$CYCLES}
SEEDS=${SEEDS:-42,7,13}
# Row set independent of --lags, so cells stay comparable across lag sets.
ROWBASIS=${ROWBASIS:-none}
MAXROWS=${MAXROWS:-0}
OUTROOT=${OUTROOT:-tec_lag_grid}

LRS=${LRS:-"0.1 0.05 0.01"}
DEPTHS=${DEPTHS:-"4 6 8"}

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/grid_summary.csv"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"

echo "lr,max_depth,n_cycles,model,n_features,validation_log_rmse,validation_log_rmse_std,validation_density_rmse,validation_mape_pct,selected_rounds,n_seeds,selected,status" > "$SUMMARY"

total=0; failed=0
for lr in $LRS; do
  for depth in $DEPTHS; do
   for cycles in $CYCLESET; do
    total=$((total + 1))
    tag="lr${lr}_depth${depth}_cyc${cycles}"
    outdir="$OUTROOT/$tag"
    log="$LOGDIR/$tag.log"
    echo "=== lr=$lr max_depth=$depth n_cycles=$cycles ==="

    "$PY" -u CoreModel/tec_lag_sensitivity.py \
      --start "$START" --end "$END" \
      --lags "$LAGS" --row-basis "$ROWBASIS" \
      --lr "$lr" --lr-decay 1.0 \
      --max-depth "$depth" \
      --rounds "$ROUNDS" \
      --early-stopping-rounds "$EARLY" \
      --n-cycles "$cycles" --gap-time "$GAP" \
      --seeds "$SEEDS" --max-rows "$MAXROWS" \
      --output-dir "$outdir" > "$log" 2>&1
    rc=$?

    if [[ $rc -ne 0 || ! -f "$outdir/results.csv" ]]; then
      failed=$((failed + 1))
      echo "  FAILED (exit $rc) - see $log"
      echo "$lr,$depth,$cycles,,,,,,,,,failed" >> "$SUMMARY"
      continue
    fi

    # Append every candidate row, prefixed with this run's hyperparameters.
    "$PY" - "$outdir/results.csv" "$lr" "$depth" "$cycles" >> "$SUMMARY" <<'EOF'
import csv, sys
path, lr, depth, cycles = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cols = ["model", "n_features", "validation_log_rmse", "validation_log_rmse_std",
        "validation_density_rmse", "validation_mape_pct", "selected_rounds",
        "n_seeds", "selected"]
with open(path) as fh:
    writer = csv.writer(sys.stdout)
    for row in csv.DictReader(fh):
        writer.writerow([lr, depth, cycles] + [row.get(c, "") for c in cols] + ["ok"])
EOF
    # `selected` is the last data column before `status`; read it by name so
    # this does not silently break again when columns move.
    best=$("$PY" - "$outdir/results.csv" <<'EOF'
import csv, sys
for row in csv.DictReader(open(sys.argv[1])):
    if row.get("selected") == "True":
        print(row["model"]); break
EOF
)
    echo "  ok - validation winner: ${best:-?}"
   done
  done
done

echo
echo "Runs: $total, failed: $failed"
echo "Combined table: $SUMMARY"

# Compact per-run view: the validation-selected candidate from each cell.
"$PY" - "$SUMMARY" <<'EOF'
import csv, sys
all_rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["status"] == "ok"]
if not all_rows:
    print("No completed runs to summarise.")
    raise SystemExit

w = "{:>8} {:>5} {:>6} {:>18} {:>10} {:>12} {:>10}"
print()
print("Validation-selected candidate per grid cell (best first):")
print(w.format("lr", "depth", "cycles", "model", "val_rmse", "val_mape", "rounds"))
for r in sorted((r for r in all_rows if r["selected"] == "True"),
                key=lambda r: float(r["validation_log_rmse"])):
    print(w.format(r["lr"], r["max_depth"], r["n_cycles"], r["model"][:18],
                   f'{float(r["validation_log_rmse"]):.6f}',
                   f'{float(r["validation_mape_pct"]):.3f}',
                   r["selected_rounds"]))

# How much does TEC actually buy in each (lr, depth) cell?  This is the
# question the grid is run to answer, so report it directly rather than
# leaving it to be read off the combined table by hand.
cells = {}
for r in all_rows:
    cells.setdefault((r["lr"], r["max_depth"], r["n_cycles"]), {})[r["model"]] = r
gains = []
for (lr, depth, cycles), models in cells.items():
    if "no_TEC" not in models:
        continue
    baseline = float(models["no_TEC"]["validation_log_rmse"])
    for name, row in models.items():
        if name == "no_TEC":
            continue
        rmse = float(row["validation_log_rmse"])
        gains.append((100.0 * (rmse - baseline) / baseline, lr, depth, cycles,
                      name, baseline, rmse))
if gains:
    gains.sort()
    g = "{:>8} {:>5} {:>6} {:>18} {:>11} {:>11} {:>9}"
    print()
    print("TEC improvement vs no_TEC in the same cell (most negative = best):")
    print(g.format("lr", "depth", "cycles", "model", "no_TEC", "with_TEC", "change"))
    for pct, lr, depth, cycles, name, baseline, rmse in gains:
        print(g.format(lr, depth, cycles, name[:18], f"{baseline:.6f}",
                       f"{rmse:.6f}", f"{pct:+.2f}%"))

    # Does n_cycles change the CONCLUSION, or only the absolute numbers?  The
    # split is what cycles controls, so a TEC gain that survives both settings
    # is a property of the data rather than of how it was blocked.
    by_cycles = {}
    for pct, lr, depth, cycles, name, _, _ in gains:
        by_cycles.setdefault((lr, depth, name), {})[cycles] = pct
    stable = [(k, v) for k, v in by_cycles.items() if len(v) > 1]
    if stable:
        print()
        print("Same (lr, depth, model) across cycle settings:")
        cyc_names = sorted({c for v in by_cycles.values() for c in v})
        h = "{:>8} {:>5} {:>18}" + "{:>11}" * len(cyc_names) + "{:>9}"
        print(h.format("lr", "depth", "model", *[f"cyc{c}" for c in cyc_names], "spread"))
        for (lr, depth, name), v in sorted(stable, key=lambda kv: min(kv[1].values())):
            vals = [v.get(c) for c in cyc_names]
            spread = max(x for x in vals if x is not None) - min(x for x in vals if x is not None)
            print(h.format(lr, depth, name[:18],
                           *[f"{x:+.2f}%" if x is not None else "-" for x in vals],
                           f"{spread:.2f}pp"))
EOF
