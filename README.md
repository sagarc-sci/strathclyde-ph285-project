# Development

## Setting up your workspace

To setup your workspace, create an Anaconda environment using

```
conda env create --file environment.yml
```

This sets up an environment `ph285-project` and installs dependencies under `$CONDA_PREFIX/ph285-project`.

To update the environment following changes to `environment.yml` run

```
conda env update --file environment.yml
```

Alternatively, environment can be specified at a desired location using the `--prefix` option.
However, this overrides the `name` parameter in `environment.yml`.
To update an environment created this way, the `--prefix` option must always be specified.