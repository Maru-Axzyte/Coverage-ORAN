# Normalized MILP sweep: reduced 85-map plan

## Extend to all 300 maps without rerunning completed maps

Run `python we_normalized_300maps/run_all_300.py` from the project root.
This checks that the code, physics, algorithm seeds and generations/map
match `results_85maps/manifest.json`, changing only the number of maps.
It copies completed maps into `results_300maps/` and skips them during
optimization. Source results/logs/checkpoints remain untouched. It also
copies partial-map checkpoints, accepting their original experiment identity
only after the source manifest has been verified; subsequent checkpoints
are written with the new 300-map experiment identity. A map-ID or seed
mismatch is still rejected. Completed-map checkpoint files need not be copied.

Use `--prepare-only` to import results without starting optimization, or
`--plot-only` to view the extended experiment after all groups are complete.
Rerunning the same command continues unfinished maps in the extended output.
Stop the old 85-map runner before starting the 300-map runner. Do not run
both at once: they would do overlapping computations. Do not edit configuration
between runs. Current settings are 300 generations/map; 300 maps means
50 maps at each of the six UE counts, not a change in generations/map.

## Original reduced plan

Run `main.py` in this directory. The output is now kept in `results_85maps/`,
separate from the original `results/` and all historical logs. Nothing is deleted.

The six UE loads are 20/30/40/50/60/70, with 5/5/5/10/30/30 maps respectively
(85 total). Each group uses map IDs 1 through its count, selected without
looking at performance. Coordinates/demands use the original deterministic
map seed policy. Algorithm and physics settings are read unchanged from
the current config; currently this is **300 WE generations per map**.
The old `results/` experiment used 400 generations/map and is not mixed
into the new experiment, including its completed maps or checkpoints.
The new MILP objective is `10*served - normalized_PRBs - normalized_power`.

Completed maps are skipped on rerun. Every 10 completed WE generations,
a checkpoint preserves the optimizer, population, RNG, learning state,
CMA, exact cache, counters and histories. Rerun the same command to continue
from the latest successfully written checkpoint. At most the work after that
checkpoint must be repeated. Interrupting a single ongoing MILP does not
resume its internal solver tree. Warm-up is checkpointed after it completes.

The same Python environment and unchanged experiment code/settings are
required to resume. The manifest rejects mixed experiment versions. Pickle
checkpoints are trusted local files: do not load checkpoints from strangers.
Checkpoints can occupy considerable disk space; they are retained alongside
JSON results. Logs are appended, including explicit START/RESUME markers;
repeated work after interruption may appear in logs, but final counters and
histories are restored from the checkpoint rather than double-counted.

Commands from the project root:

```
python we_normalized_300maps/main.py
python we_normalized_300maps/main.py --prepare-only
python we_normalized_300maps/main.py --plot-only
```

Progress logs: `results_85maps/logs/ue_XXX_map_YYY.log`.
Final map results: `results_85maps/runs/ue_XXX_map_YYY.json`.
Resume files: `results_85maps/checkpoints/ue_XXX_map_YYY.pkl`.
Summary: `results_85maps/summary.csv`.
Final figure: `results_85maps/mean_served_ues.png`.
