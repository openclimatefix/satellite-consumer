# Satellite Consumer

**Download and convert satellite data for use in ML pipelines**
 
[![workflows badge](https://img.shields.io/github/actions/workflow/status/openclimatefix/satellite-consumer/merged_ci.yml?label=workflow&color=FFD053)](https://github.com/openclimatefix/satellite-consumer/actions/workflows/merged_ci.yml)
[![tags badge](https://img.shields.io/github/v/tag/openclimatefix/satellite-consumer?include_prereleases&sort=semver&color=FFAC5F)](https://github.com/openclimatefix/satellite-consumer/tags)
[![contributors badge](https://img.shields.io/github/contributors/openclimatefix/satellite-consumer?color=FFFFFF)](https://github.com/openclimatefix/satellite-consumer/graphs/contributors)
[![ease of contribution: medium](https://img.shields.io/badge/ease%20of%20contribution:%20medium-f4900c)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

Satellite data is a valuable resource for training machine learning models.
Forecasting renewable generation requires knowledge of the weather conditions,
and those weather conditions can be inferred and enriched using satellite data.

EUMETSAT provide a range of satellite data products, which are easily available
in `NAT` image format. In order to improve its accessibility for training models,
this consumer processes downloaded data into the `Zarr` format.

> [!Note]
> This repo is in early development and so will undergo rapid changes.
> Breaking changes may occur in the CLI and the API without warning.


## Installation

Install using the container image:

```bash
$ docker pull ghcr.io/openclimatefix/satellite-consumer
```

or, if you prefer a CLI:

```bash
$ pip install git+https://github.com/openclimatefix/satellite-consumer.git
```

This will put the `sat-consumer-cli` command in your virtual environments `bin` directory.

## Example usage

```bash
$ docker run \
    -e SATCONS_COMMAND=consume \
    -e SATCONS_SATELLITE=rss \
    -e EUMETSAT_CONSUMER_KEY=<your-key> \
    -e EUMETSAT_CONSUMER_SECRET=<your-secret> \
    -v $(pwd)/work:/work \
    ghcr.io/openclimatefix/satellite-consumer
```

This will download the latest available data for the `rss` satellite and store it in the `/work` directory.
For a description of all the possible configuration options, see [Documentation](#documentation).

## Documentation

The satellite consumer provides a number of commands for different logical processing of raw data.
These commands (and their options) can be seen when using the cli entrypoint:

```bash
$ sat-consumer-cli --help
```

When running the satellite consumer using the environment entrypoint (as in the docker container)

```bash
$ sat-consumer
```

the command is chosen via an environment variable. There are also a number of common configuration
options that are shared between all commands:

| Variable | Default | Description |
|----------|---------|-------------|
| `SATCONS_COMMAND` |  | The command to run (consume/merge). |
| `SATCONS_SATELLITE` | | The satellite to consume data from. |
| `SATCONS_WORKDIR` | `/mnt/disks/sat` | The working directory. In the container, this is set to `/work` for easy mounting. |
| `SATCONS_RESOLUTION` | `3000` | The desired resolution of the satellite images in meters ('3000', '1000'). |
| `EUMETSAT_CONSUMER_KEY` |  | The EUMETSAT consumer key. |
| `EUMETSAT_CONSUMER_SECRET` |  | The EUMETSAT consumer secret. |

Each command then has its own set of configuration options:

**Consume:**

*Downloads scans for a given time and window into a zarr store in the given working directory.*

| Variable | Default | Description |
|----------|---------|-------------|
| `SATCONS_TIME` | | The time to consume data for (when using the `consume` command). Leave unset to download latest available. | 
| `SATCONS_WINDOW_MINS` | `0` | The time window to consume data for in minutes (defaults to a single scan). |
| `SATCONS_WINDOW_MONTHS` | `0` | The number of months to consume data for (takes precedence over `SATCONS_WINDOW_MINS`). |
| `SATCONS_VALIDATE` | `false` | Whether to validate the downloaded data. |
| `SATCONS_RESCALE` | `false` | Whether to rescale the downloaded data to the unit interval. |
| `SATCONS_NUM_WORKERS` | `1` | The number of workers to use for processing. |
| `SATCONS_ICECHUNKS` | `false` | Whether to use icechunk repositories for storage. |
| `SATCONS_CROP_REGION` | `` | The region string to crop data to ('uk', 'india', 'west-europe') |

**Merge:**

*Merges consumed stores for a given time window into a single store in the working directory.*

| Variable | Default | Description |
|----------|---------|-------------|
| `SATCONS_SATELLITE` | | The satellite to consume data from. |
| `SATCONS_WINDOW_MINS` | `210` | The time window to merge data for. |
| `SATCONS_CONSUME_MISSING` | `false` | Whether to consume missing data. |

## FAQ

### How do I add a new satellite to the consumer?

Currently the consumer is built to the specific data requirements of Open Climate Fix.
However, adding a new satellite in the from EUMETSAT shouldn't be too hard, provided it uses
the same `seviri_l1b_native` format and sensor channels - just update the available satellites
in `config.py`.

## Development

OCF recommends using [uv](https://docs.astral.sh/uv/) for managing your virtual environments.

```bash
$ git clone git@github.com:openclimatefix/satellite-consumer.git
$ cd satellite-consumer
$ uv sync
```

### Running the CLI

The python package contains a CLI entrypoint for ease of use when developing, which is available
to your shell via the `sat-consumer-cli` command, assuming you have built the project in a virtual
environment, and activated it.

### Linting and static type checking
 
This project uses [MyPy](https://mypy.readthedocs.io/en/stable/) for static type checking
and [Ruff](https://docs.astral.sh/ruff/) for linting.
Installing the development dependencies makes them available in your virtual environment.

Use them via:

```bash
$ python -m mypy .
$ python -m ruff check .
```

Be sure to do this periodically while developing to catch any errors early
and prevent headaches with the CI pipeline. It may seem like a hassle at first,
but it prevents accidental creation of a whole suite of bugs.

### Running the test suite

There are some additional dependencies to be installed for running the tests,
be sure to pass `--extra=dev` to the `pip install -e .` command when creating your virtualenv
(`uv sync` includes the development dependencies by default, so `uv` users can ignore this!).

Run the unit tests with:

```bash
$ python -m unittest discover -s src/satellite_consumer -p "test_*.py"
```

> [!Note]
> If you have created your virtual environment using `uv`, the above can be run via
> the `Makefile`, using `make typecheck`, `make lint`, and `make test` respectively.

 
## Further reading

On the directory structure:
- The official [PyPA discussion](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) on 
"source" and "flat" layouts.


---

## Contributing and community

[![issues badge](https://img.shields.io/github/issues/openclimatefix/ocf-template?color=FFAC5F)](https://github.com/openclimatefix/ocf-template/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)

- PR's are welcome! See the [Organisation Profile](https://github.com/openclimatefix) for details on contributing
- Find out about our other projects in the [here](https://github.com/openclimatefix/.github/tree/main/profile)
- Check out the [OCF blog](https://openclimatefix.org/blog) for updates
- Follow OCF on [LinkedIn](https://uk.linkedin.com/company/open-climate-fix)


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)

