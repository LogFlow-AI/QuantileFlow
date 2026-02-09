# QuantileFlow

> This package provides API and functionality to efficiently compute quantiles for anomaly detection in service/system logs. Developed under LogFlow-AI initiative.

[![Latest Version on PyPI](https://img.shields.io/pypi/v/QuantileFlow.svg)](https://pypi.python.org/pypi/QuantileFlow/)
![Build Status](https://github.com/LogFlow-AI/QuantileFlow/actions/workflows/test.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/QuantileFlow/badge/?version=latest)](https://QuantileFlow.readthedocs.io/en/latest/?badge=latest)
[![Built with PyPi Template](https://img.shields.io/badge/PyPi_Template-v0.8.0-blue.svg)](https://github.com/christophevg/pypi-template)
[![DOI](https://img.shields.io/badge/DOI-10.32628%2FCSEIT261212-blue)](https://doi.org/10.32628/CSEIT261212)

## Key Features

- **Multiple Algorithms**: Includes DDSketch, MomentSketch and HDRHistogram implementations
- **Memory Efficient**: Uses compact data structures regardless of data stream size
- **Mergeable**: Supports distributed processing by merging sketches
- **Accuracy Guarantees**: Provides configurable error bounds
- **Fast Operations**: O(1) insertions and efficient quantile queries
- **Python API**: Simple and intuitive interface for Python applications

## Citation

If you use QuantileFlow in your research or project, please cite our paper:

**Plain Text:**
```
Dhyey Mavani, Tairan (Ryan) Ji, and Marius Cotorobai, "QuantileFlow: A Unified and Accelerated Quantile Sketching Framework for Anomaly Detection in Streaming Log Data", Int. J. Sci. Res. Comput. Sci. Eng. Inf. Technol, vol. 12, no. 1, pp. 250–259, Jan. 2026, doi: 10.32628/CSEIT261212.
```

**BibTeX:**
```bibtex
@article{mavani2026quantileflow,
  title={QuantileFlow: A Unified and Accelerated Quantile Sketching Framework for Anomaly Detection in Streaming Log Data},
  author={Mavani, Dhyey and Ji, Tairan (Ryan) and Cotorobai, Marius},
  journal={International Journal of Scientific Research in Computer Science, Engineering and Information Technology},
  volume={12},
  number={1},
  pages={250--259},
  year={2026},
  month={January},
  doi={10.32628/CSEIT261212}
}
```

**APA:**
```
Mavani, D., Ji, T. (R.), & Cotorobai, M. (2026). QuantileFlow: A Unified and Accelerated Quantile Sketching Framework for Anomaly Detection in Streaming Log Data. International Journal of Scientific Research in Computer Science, Engineering and Information Technology, 12(1), 250-259. https://doi.org/10.32628/CSEIT261212
```

**DOI:** [https://doi.org/10.32628/CSEIT261212](https://doi.org/10.32628/CSEIT261212)

## Contents

```{toctree}
:maxdepth: 2

whats-in-the-box.md
getting-started.md
examples.md
api/index.md
api/ddsketch.md
api/momentsketch.md
api/hdrhistogram.md
contributing.md
code.md
```

