# Offline CSV study

This example uses the same data-generator interface as the sequential study, with
a stored dataset and an offline algorithm. It computes the mean and unbiased
sample variance of the `measurement` column in a small illustrative CSV.

From the repository root, after `python -m pip install -e .`:

```bash
ews run examples/offline_csv/experiment.yml --output output/offline
ews analyze examples/offline_csv/experiment.yml --output output/offline
ews inspect output/offline
```

The loader resolves `observations.csv` relative to the configuration file.
`offline_paper:MeanEstimator` names the class defined beside that file. Its
constructor accepts the injected RNG even though sample-mean estimation is
deterministic. `fit(dataset)` returns a mapping of numerical results; the offline
protocol performs the whole fit as one logical step.

There is only one repetition because repeating a deterministic estimator on the
same fixed dataset would not provide independent statistical evidence. The
reported `variance` is the sample variance of the observations, not uncertainty
across experiment repetitions.

Run the same command again to validate and skip the completed run. If an offline
fit is interrupted before its single step finishes, it restarts that fit. For
checkpointing inside a long fit, supply an incremental interaction protocol with
safe steps.

To use another estimator, define a class with `fit`, `state_dict`, and
`load_state_dict`, then change `algorithms[].type` and its `params` in YAML. To
replace the CSV source with synthetic data, provide a generator implementing
`generate(request)`, `state_dict`, and `load_state_dict`. Scientific changes belong
in a new output directory.
