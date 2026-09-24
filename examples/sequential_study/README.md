# A sequential learning study

This self-contained example compares UCB and epsilon-greedy on Gaussian bandits
with 10, 20, and 50 arms. Each combination has 20 independent repetitions:
120 runs, each with 1,000 rounds. It is an illustrative experiment for validating
the workflow, not a published benchmark or a performance claim.

## Run and resume

From the repository root:

```bash
python -m pip install -e '.[plot]'
ews plan examples/sequential_study/experiment.yml
ews run examples/sequential_study/experiment.yml --output output/bandits
ews inspect output/bandits
ews plot examples/sequential_study/experiment.yml --output output/bandits
```

For a repeatable demonstration of interruption and resumption, start a fresh output
directory, pause after 300 new rounds per run, and resume using two workers:

```bash
ews run examples/sequential_study/experiment.yml --output output/resume-demo --max-steps 300
ews run examples/sequential_study/experiment.yml --output output/resume-demo --workers 2
ews plot examples/sequential_study/experiment.yml --output output/resume-demo
```

Changing worker count preserves the numerical results. A normal interruption
saves progress at a safe boundary; a forced termination can lose work since the
previous committed checkpoint. The default checkpoint interval is two minutes.
Short runs may finish before their first periodic checkpoint.

Re-running a completed experiment validates and skips its completed runs. Changing
scientific parameters or recording resolution requires a new output directory.
Plot settings can change, and `ews plot` rebuilds figures from saved results.

## Scientific choices

`paper.py` contains the whole scientific part of the study:

- `GaussianBandit` draws arm means once, produces Gaussian rewards, and owns the
  evolving round counter and cumulative pseudo-regret.
- `UCB` and `EpsilonGreedy` learn from the scalar reward passed to `observe`.
- `ClippedFeedbackBandit` demonstrates a different observable feedback rule.

Before requesting an action, the online protocol calls `context()` when available.
This example exposes only the number of arms, so algorithms initialize their
arrays using the same size as the environment without duplicating grid parameters.
Neither algorithm receives the hidden arm means or evaluator measurements.

An algorithm and its generator receive independent RNGs. For a fixed size and
repetition, the two algorithms start with the same generated arm means. Their
chosen actions can differ, so rewards and subsequent trajectories can differ too.

The generator accumulates pseudo-regret on every round:

```text
regret[t] = sum(best_arm_mean - chosen_arm_mean[s], s = 1 ... t)
```

The recorder saves every tenth round and always the final round. Consequently,
saved regret values describe the full trajectory at those points. Saved rewards
are only a subsample; summing them would not recover full cumulative reward.

The default analysis groups by arm count and algorithm, computes mean regret, and
shows one standard error across independent repetitions. This band is not a
confidence interval with a stated coverage guarantee. Figures are exported as
PDF, JPG, and editable TikZ/PGFPlots source.

## Add an algorithm

Define an ordinary Python class beside the YAML file with these methods:

```python
class MyAlgorithm:
    def __init__(self, *, rng, **parameters): ...
    def act(self, context=None): ...
    def observe(self, action, feedback): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```

Use the supplied `rng` for every random choice. Include all changing values needed
to continue learning in `state_dict`; the library separately saves the RNG state.
State may contain numerical arrays, scalar values, lists, tuples, and mappings
with string keys. Add an entry under `algorithms`:

```yaml
- name: my_algorithm
  type: paper:MyAlgorithm
  params:
    learning_rate: 0.1
```

The library resolves `paper` beside the configuration file. Paper-specific classes
are supplied by configuration; the library has no dependency on this example.

## Change the feedback rule

Change `data.type` to `paper:ClippedFeedbackBandit` and choose a fresh output
directory. This generator clips the reward delivered to the algorithm into
`[0, 1]`, records it as `reward`, and retains the original draw as `raw_reward` in
its measurements. Add `raw_reward` to `recording.fields` to store that field.
Pseudo-regret still uses the underlying Gaussian arm means; this is an explicit
scientific choice that should be reconsidered for a different paper's question.

More generally, `Feedback(value, measurements)` separates information visible to
the learner from numerical observations available for evaluation. Custom feedback
does not require changing execution, checkpointing, or plotting code.

## Artifacts and resources

Each run has its own metadata, result chunks, and recent checkpoints under
`output/bandits/runs/`. Figures and summary tables live under `analysis/`. Numerical
chunks are uncompressed by default. The 16 MiB buffer limit is per active run;
these small examples usually finish with much less buffered data.

Use one worker as a modest default. More workers increase memory usage and help
only when the additional computation outweighs process and storage overhead.
