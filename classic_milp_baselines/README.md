# Random Search–MILP and GA–MILP

Run from this directory with `python main.py`, or from the project root with
`python classic_milp_baselines/main.py`. This runs both methods sequentially;
it does not rerun WE. No core project files are modified.

Both methods use the same fixed selection of 60 historical maps:
5 each for 20/30/40/50 UE, and 20 each for 60/70 UE. Sampling uses seed
20261015 without replacement and without selecting by performance.
The exact UE positions and demands are frozen in `results/inputs`;
`results/selection.json` records map IDs, seeds, source hashes and settings.
Each method receives the historical WE unique-MILP budget of that map
(currently 1280), not a matching number of generations.

Commands:

```
python main.py --prepare-only
python main.py
python main.py --method random
python main.py --method ga
python main.py --summary-only
```

Completed maps are skipped on rerun. An interrupted map restarts from its
original seed; this is not an in-map checkpoint. Different method options
can share the same output directory. Changing code/settings/inputs requires
a new `--output-dir` to avoid mixing experiments.

Windows result replacement retries temporary file locks for about 7 seconds.
Each save uses a unique temporary file. If the lock persists, the old result
and the new temporary JSON are both retained, and the pending path is printed.
The known original runner hash is accepted for this I/O-only repair when all
other input/settings/core hashes match; completed maps remain reusable.

The default GA population is 80. Both methods start with the same 80
centre-x/y, random-h/theta warm-up placements as the historical WE exact
seed policy. Random Search then samples all four coordinates uniformly
and independently. GA uses the shared tournament, UAV-block SBX and Gaussian
mutation operators, plus generational replacement and 15% parent elitism.
Physical evaluation uses the shared clipping/projection and violation
definitions; no geometric constraint is relaxed or removed.

The shared epsilon-Deb comparator ranks V, coverage, demand service, then
resource fitness. Both methods retain the existing contextual weight learner,
updated every 40 exact evaluations, and the strict exact-incumbent comparator.
This is a shared-scoring comparison, not a minimal GA with a new count-only
objective. There is no WE survival, surrogate screening or CMA search.
The scoring helper is instantiated from the main optimizer but its `run()`
method is never called. The original MILP objective is unchanged.

Historical source hashes are checked. The known ga.py default-generation
change (300 to 400) is allowed only if replacing that one default exactly
recovers the historical hash; historical settings are loaded explicitly.
No other historical code differences are silently accepted.

Results and detailed logs are in `results/random` and `results/ga`.
`comparison.csv` and `comparison.json` pair each result with its exact WE map;
summaries print feasible mean served counts per load. Full final MILP payloads,
placements, resource use, violations, evaluation traces, weights, elapsed time
and MILP calls are retained. Resource-comparable flags require feasible equal
served count and admitted demand; actual served-ID equality is also reported.

Historical WE timing may come from another computer. These runs compare
complete search systems on a common oracle budget, not isolated WE effects,
and do not establish a controlled wall-clock speed comparison. One algorithm
seed is used per map, matching the saved WE runs. Learning/update schedules
and population sizes differ from the full WE system and are recorded.

Tests use a fake MILP evaluator, not optimization experiments:
`python -m unittest discover -s classic_milp_baselines -p "test_*.py"`
(run the test command from the project root).
