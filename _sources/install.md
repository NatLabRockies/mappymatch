# Install

## From PyPI

```bash
pip install mappymatch
```

```{note}
While mappymatch is a pure python package, some of the geospatial dependnecies can be hard to install via pip.
If you encounted issues, our recommended solution is to install the package from the source code, using conda
to facilitate the packages that can be challenging to install.

We hope to eventually provide a conda distribution (help doing this would be greatly appreciated!)
```

## From Source

Clone the repo:

```bash
git clone https://github.com/NatLabRockies/mappymatch.git && cd mappymatch
```

Get [Anaconda](https://www.anaconda.com/download) or [miniconda](https://docs.anaconda.com/miniconda/).

Then, use the `environment.yml` file (in the repo) to install dependencies:

```bash
conda env create -f environment.yml
```

To activate the `mappymatch` environment:

```bash
conda activate mappymatch
```
