# Offline CSV study

This study estimates the mean and unbiased sample variance of the `measurement`
column in a small CSV file. It shows how an offline algorithm uses the same
data-generator interface as the sequential study, while fitting a complete
dataset in one protocol step.

[`experiment.yml`](experiment.yml) selects the built-in CSV generator and
`experiment_code.algorithms:MeanEstimator`, defined in
[`experiment_code/algorithms.py`](experiment_code/algorithms.py). The CSV path
is relative to the YAML file. The estimator accepts an injected RNG as part of
the component contract, although the calculation itself is deterministic.

From the repository root, after `python -m pip install -e .`:

```bash
ews run examples/offline_csv/experiment.yml --output output/offline
ews inspect output/offline
```

The protocol loads the data and calls `fit(dataset)` once. That fit returns
named numerical outputs. After successful execution, `run` applies the configured
metrics to those saved outputs. The CSV generator also saves the input matrix
as the run's instance, so analysis can inspect it later without loading the
estimator. There is one repetition:
repeating the same deterministic fit on the same data would not add independent
evidence.

After changing metrics or aggregation settings, update summaries from saved
results without fitting the estimator again:

```bash
ews analyze examples/offline_csv/experiment.yml --output output/offline
```

A later run validates and reuses the completed result. If execution stops inside
`fit`, the entire fit starts again because this protocol has one step. For a
longer computation that needs intermediate checkpoints, use a protocol with
smaller steps. See [the offline and instance contracts](../../docs/CONFIGURATION.md#instances-and-scientific-components)
for extending this study.
