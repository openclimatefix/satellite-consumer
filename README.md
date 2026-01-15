# Satellite Consumer
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

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

## Example usage

```bash
$ docker run \
    -e <SATCONS_CONFIG_VARIABLE>=<your-value>
    -e EUMETSAT_CONSUMER_KEY=<your-key> \
    -e EUMETSAT_CONSUMER_SECRET=<your-secret> \
    -v $(pwd)/work:/work \
    ghcr.io/openclimatefix/satellite-consumer
```

This will download the latest available data for the `rss` satellite and store it in the `/work` directory.
For a description of all the possible configuration options, see [Documentation](#documentation).

## Configuration

There are a number of configuration options exposed via environment variables. For the full list,
see `cmd/application.conf.

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
```

Initialize the repository with recommended settings for development via 

```bash
$ make init
```

### Linting and static type checking
 
This project uses [MyPy](https://mypy.readthedocs.io/en/stable/) for static type checking
and [Ruff](https://docs.astral.sh/ruff/) for linting.
Installing the development dependencies makes them available in your virtual environment.

There is a makefile target to automatically lint, typecheck, and format the codebase:

```bash
$ make lint
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
$ make test
```
 

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
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/devsjc"><img src="https://avatars.githubusercontent.com/u/47188100?v=4?s=100" width="100px;" alt="devsjc"/><br /><sub><b>devsjc</b></sub></a><br /><a href="https://github.com/openclimatefix/satellite-consumer/commits?author=devsjc" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt="Jacob Prince-Bieker"/><br /><sub><b>Jacob Prince-Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/satellite-consumer/commits?author=jacobbieker" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/satellite-consumer/issues?q=author%3Apeterdudfield" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.linkedin.com/in/ram-from-tvl"><img src="https://avatars.githubusercontent.com/u/114728749?v=4?s=100" width="100px;" alt="Ramkumar R"/><br /><sub><b>Ramkumar R</b></sub></a><br /><a href="https://github.com/openclimatefix/satellite-consumer/commits?author=ram-from-tvl" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)

