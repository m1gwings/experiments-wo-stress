# A sequential learning study

This example compares UCB and epsilon-greedy on Gaussian bandits with 10, 20, and
50 arms, using 20 independent repetitions: 120 runs of 1,000 rounds. It demonstrates
repeatable execution and analysis; it is not a published benchmark or performance
claim.

## Run, pause, and resume

From the repository root:

```bash
python -m pip install -e '.[plot]'
ews plan examples/sequential_study/experiment.yml
ews run examples/sequential_study/experiment.yml --output outputs/bandits
ews inspect outputs/bandits
ews plot examples/sequential_study/experiment.yml --output outputs/bandits
```

For a deliberate interruption, use a fresh output directory:

```bash
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --max-steps 300
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --workers 2
ews plot examples/sequential_study/experiment.yml --output outputs/resume-demo
```

A normal interruption checkpoints at the next complete step. A forced termination
can lose work since the previous committed checkpoint. Checkpoints default to a
two-minute interval; short runs may finish before the first periodic checkpoint.
Changing worker count preserves the numerical results.

Repeated execution validates and reuses completed runs. Increasing `budget.steps`
continues this example from retained final state: its protocol, generator, and
algorithms explicitly support extension. Recording a different set of fields or
at a different interval creates a distinct retained recording variant. Code,
scientific parameter, or input changes select new affected variants while preserving
previous artifacts. A separate directory is useful for an unrelated study.

## Instance, observations, and metrics

`paper.py` contains the scientific components:

- `GaussianBandit.create_instance` draws arm means with the separate instance RNG.
  The library saves those means and instance metadata for later analysis.
- Each generator receives that saved instance and its own RNG. It owns the evolving
  round counter and produces rewards in response to actions.
- `UCB` and `EpsilonGreedy` choose actions and learn from observable rewards.
- `ClippedFeedbackBandit` demonstrates a custom feedback rule.

Before each action, `context()` exposes the number of arms. Neither algorithm sees
the hidden means. A fixed size and repetition shares the generated instance across
algorithms; their actions and resulting trajectories can differ.

The recorder saves every action and reward. The built-in `pseudo_regret` metric
uses saved actions and instance means to compute:

```text
regret[t] = sum(best_arm_mean - chosen_arm_mean[s], s = 1 ... t)
```

Regret is analysis output, not generator state. This keeps the generator reusable
with other scientific metrics. Full cumulative regret requires every action;
the metric rejects a sparse recording rather than reporting a partial sum as a
full trajectory. Change plotting resolution after analysis when only fewer figure
points are needed.

The aggregator computes mean regret and one standard error across independent
repetitions. This band is not a confidence interval with a stated coverage guarantee.
Figures are exported as PDF, JPG, and editable TikZ/PGFPlots source. Valid cached
metrics, summaries, and figures are reused; changing a metric or figure invalidates
its affected cache entries without rerunning simulations.

## Add an algorithm

Define a class beside the YAML file:

```python
class MyAlgorithm:
    supports_extension = True  # Only if state can continue at a larger budget.

    def __init__(self, *, rng, **parameters): ...
    def act(self, context=None): ...
    def observe(self, action, feedback): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```

Use the injected RNG for random choices. Include all changing values needed to
continue learning in checkpoint state. The library separately stores RNG state
and provides a run logger as `self.logger`. Use standard logger methods instead
of printing from a worker; debug output can explain scientific decisions when
investigating a run.

Add a YAML algorithm entry with `name`, `type: paper:MyAlgorithm`, and its `params`.
All parameters except the execution budget belong to the scientific configuration.
An algorithm whose behavior depends intrinsically on a chosen horizon should keep
that horizon as an explicit scientific parameter and should not claim automatic
budget extension.

## Change the feedback rule

Set `data.type` to `paper:ClippedFeedbackBandit` in a new study output. This generator
clips the observable reward to `[0, 1]`, records it as `reward`, and also exposes
`raw_reward` to the recorder. Add `raw_reward` to `recording.fields` to save it.
The saved instance retains the underlying Gaussian means; whether the same regret
comparator remains scientifically appropriate is a choice for the paper.

`Feedback(value, measurements)` separates learner-visible information from saved
evaluator observations. A new feedback rule does not require a new executor,
checkpoint implementation, or figure exporter.

## Resources and inspection

Each run has independent artifacts and a rotating log. Instances are immutable;
checkpoints contain evolving state and RNGs. The 16 MiB buffer limit is per active
run, with additional memory for scientific state and serialization. Numerical data
is uncompressed by default.

Dense summary CSV files and analysis caches can use more disk than the original
trajectory. Analysis cleanup removes these rebuildable outputs while retaining
the underlying runs and instances.

Use one worker as a modest default. Additional workers help when their computation
outweighs process and storage costs. Use `ews inspect` to inspect status, selected
recordings, and available results. See the [configuration guide](../../docs/CONFIGURATION.md)
for logging, recording variants, budget rules, and storage cleanup.
