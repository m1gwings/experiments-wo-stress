# A sequential learning study

This study compares UCB and epsilon-greedy on Gaussian bandits with 10, 20, and
50 arms. Each algorithm and bandit size has 20 independent repetitions: 120 runs
of 1,000 rounds. The settings illustrate the workflow; they are not a published
benchmark.

## How the study fits together

The Python classes live in `experiment_code/`.
[`data.py`](experiment_code/data.py) defines the bandit and draws an immutable
instance of arm means for each size and repetition.
[`algorithms.py`](experiment_code/algorithms.py) defines the two learning rules.
The algorithms see the arm count and observed rewards, but not the hidden means.

[`experiment.yml`](experiment.yml) selects the online protocol, generator, and
algorithms. Its grid varies the number of arms. For each grid choice and
repetition, the library plans a run for each algorithm; the algorithms can share
the same saved instance while producing different trajectories.

The online protocol records every action and reward. After execution, the
`pseudo_regret` metric combines those actions with the saved means. The
aggregator averages regret over repetitions and computes a standard error; the
figure exporter writes PDF, JPG, and editable TikZ files. Regret is computed
from saved data, rather than accumulated inside the generator.

## Run it

From the repository root:

```bash
python -m pip install -e '.[plot]'
ews plan examples/sequential_study/experiment.yml
ews build examples/sequential_study/experiment.yml --output outputs/bandits
ews inspect outputs/bandits
```

`plan` shows the 120 runs before computation. `build` runs them and produces the
configured analysis; figures appear under `outputs/bandits/analysis/figures/`.
Running `build` again reuses compatible completed work.

To see interruption and resumption, use a separate output directory:

```bash
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --max-steps 300
ews run examples/sequential_study/experiment.yml --output outputs/resume-demo --workers 2
ews plot examples/sequential_study/experiment.yml --output outputs/resume-demo
```

The first command pauses each run after 300 steps. The second continues the runs
with two workers, and `plot` produces the analysis from their completed results.
The [configuration guide](../../docs/CONFIGURATION.md#budget-continuation-and-retained-artifacts)
explains when a run can continue and how other changes affect reuse.

## Try a scientific change

Add another learning rule to `experiment_code/algorithms.py`, then add a named
entry in YAML with `type: experiment_code.algorithms:MyAlgorithm` and its
parameters. The class needs `act`, `observe`, `state_dict`, and
`load_state_dict`; use its injected `rng` for random choices. The
[extension reference](../../docs/CONFIGURATION.md#instances-and-scientific-components)
describes the full contract.

For a different feedback rule, select
`experiment_code.data:ClippedFeedbackBandit` as `data.type` in a new study
output. It clips the reward shown to the learner and offers `raw_reward` as an
additional recordable field. Whether Gaussian-mean pseudo-regret remains the
right comparator for that changed learning problem is a scientific decision.
