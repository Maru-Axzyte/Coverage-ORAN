# Main WE + five baselines: restored normalized plan

Run from the repository root:

```powershell
python -B baseline_150maps/run_all.py --prepare-only
python -B baseline_150maps/run_all.py
```

## Frozen 150-map plan

20/30/40/50/60/70 UE use 10/10/10/20/50/50 maps per method.

The ONLY historical source is we_normalized_300maps. Matching results_300maps records take
priority; compatible results_85maps records are deduplicated. Different code/settings,
including generations and objective, are rejected instead of mixed.

Preparation keeps compatible completed maps first. For each remaining shortage it randomly
selects unused IDs, without replacement, from the approved original normalized plan.
For those IDs it reconstructs ONLY the UE inputs using that plan's original scenario seed
and distribution. It does not invent served counts, power, PRBs or runtime results.

The selected IDs, exact UE arrays, demand and seeds are saved once in selection.json and inputs/.
All six methods use these same inputs. Rerunning does not draw another selection, even if
more normalized source maps have completed since preparation. If the source plan lacks
enough unused IDs, preparation stops instead of reading another collection.

At the current restored state, preparation retains 55 completed WE maps and selects 95
remaining cases: 5/5/5/10/20/50 new cases for the six UE groups, respectively.
Duplicate source records are never counted twice.

The retired GA_logs archive, legacy copied study inputs, results_current and the old
baseline_150maps/results study are NOT read. Existing directories are preserved.
No --source option permits selecting the retired archive.

## Execution and restart

The default output is baseline_150maps/results_normalized.
Execution is sequential: WE, Random Search, GA, PSO, adapted Zhong, Grid MILP.
There are 150 maps for each of six methods; compatible completed jobs are skipped.

WE imports the actual compatible completed normalized results named in the frozen provenance.
Selected unfinished native checkpoints may also be copied from the approved source after
validating their manifest, experiment and map identity. A checkpoint is not a completed result.
WE computes the remaining cases for real; its own checkpoints are saved every 10 completed
generations. An interruption can lose work after the last checkpoint, not completed maps.

Random/GA/PSO/Zhong commit solver-confirmed allocations to per-map SQLite checkpoints.
Restart replays those labels without solving the committed allocations again; replay itself
still takes time. Grid saves modes and incumbents, rebuilding and warm-starting on restart;
its solver tree is not saved.

Use the SAME command after interruption and preserve the whole output directory.
Do not change code/configuration or package versions mid-experiment.
The runner rejects code/configuration changes instead of mixing experiments.

```powershell
python -B baseline_150maps/run_we.py
python -B baseline_150maps/main.py --methods pso
python -B baseline_150maps/run_all.py --methods we pso
python -B baseline_150maps/run_all.py --summary-only
```

Run preparation first if invoking an individual runner. Run WE before Grid: Grid requires
the same-map real normalized WE runtime already imported or computed in this study.

## What is unchanged

This update changes map scheduling, provenance, reuse and resume only.
The main WE algorithm, physical parameters, constraints, generations and population are unchanged.
Search baseline solve budgets still come from current configured settings (currently 1280),
never from the retired archive. This is not an early-stopping change.
Grid retains its existing count/demand/PRB/power objective; this runner does not silently
replace it with the normalized fixed-placement oracle objective.

## Outputs

- source_availability.json: compatible source plans and completed/new counts.
- selection.json and inputs/: frozen common map plan and exact inputs.
- we/reuse_report.json: completed WE imports, copied checkpoints and pending cases.
- METHOD/{runs,logs,checkpoints}: real results, logs and restart data.
- all_methods_comparison.csv and all_methods_summary.csv: all-method summaries.

Partial groups are labelled completed/required, not reported as finished groups.
Development runs only unit tests and --prepare-only, not the full experiments.
