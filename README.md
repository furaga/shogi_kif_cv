# shogi_kif_cv

## Requirements

### Windows

- Anaconds (for python3.6)
- chocolatey
- Git Bash

### Mac

- pyenv
- virtualenv
- python3.6

## Setup

### Windows

In Anaconda Prompt:
```bash
conda create -n shogi pip=9.0 python=3.6
```

In Git Bash:
```bash
choco install tesseract
source activate  shogi
pip install -r requirements.txt
```

### Mac (We have not confirmed yet)

```bash
brew install tesseract
pyenv virtualenv shogi python=3.6
source ~/.pyenv/versions/shogi/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
jupyter notebook
```

Open and run `notebooks/img2kif.ipynb`