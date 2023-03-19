<p align="center">
  <img width="372" height="200" src="assets/valinor_logo.png">
  <br>
  Identifying genetic interactions in CRISPR double knock-out experiments
</p>

![Tests!](https://github.com/dpohanlon/brioche/actions/workflows/python-package.yml/badge.svg)

Installation
---
Install from the Github repository
```bash
git clone git@github.com:dpohanlon/brioche.git
pip install .
```
Usage
---
Prepare some data in a contingency table format, with row and column set annotations
```python
row_names = ["Gene1", "Gene2", "Gene3"]
col_names = ["TF1", "TF2", "TF3"]

data = np.array([[30, 27, 10], [28, 25, 11], [31, 29, 15])
```
