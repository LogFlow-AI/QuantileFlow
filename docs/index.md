# QuantileFlow

> This package provides API and functionality to efficiently compute quantiles for anomaly detection in service/system logs. Developed under LogFlow-AI initiative.

[![Latest Version on PyPI](https://img.shields.io/pypi/v/QuantileFlow.svg)](https://pypi.python.org/pypi/QuantileFlow/)
[![Supported Implementations](https://img.shields.io/pypi/pyversions/QuantileFlow.svg)](https://pypi.python.org/pypi/QuantileFlow/)
![Build Status](https://github.com/LogFlow-AI/QuantileFlow/actions/workflows/test.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/QuantileFlow/badge/?version=latest)](https://QuantileFlow.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/LogFlow-AI/QuantileFlow/badge.svg?branch=master)](https://coveralls.io/github/LogFlow-AI/QuantileFlow?branch=master)
[![Built with PyPi Template](https://img.shields.io/badge/PyPi_Template-v0.8.0-blue.svg)](https://github.com/christophevg/pypi-template)

## Key Features

- **Multiple Algorithms**: Includes both DDSketch and MomentSketch implementations
- **Memory Efficient**: Uses compact data structures regardless of data stream size
- **Mergeable**: Supports distributed processing by merging sketches
- **Accuracy Guarantees**: Provides configurable error bounds
- **Fast Operations**: O(1) insertions and efficient quantile queries
- **Python API**: Simple and intuitive interface for Python applications

## Contents

```{toctree}
:maxdepth: 2

whats-in-the-box.md
getting-started.md
examples.md
api/index.md
api/ddsketch.md
api/momentsketch.md
contributing.md
code.md
```

