<p align="center">
  <img width="372" height="200" src="assets/valinor_logo.png">
  <br>
  Identifying genetic interactions in CRISPR double knock-out experiments
</p>

![Tests!](https://github.com/dpohanlon/valinor/actions/workflows/python-app.yml/badge.svg)

Installation
---
Install from the Github repository.
Make sure that you are installing Valinor in an environment that has python version >3.7 but smaller than 3.11.
```bash
git clone git@github.com:dpohanlon/valinor.git
pip install -e .
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Usage
---
```bash
valinor \
        --combinationsFile combs.pq \
        --singletonsFile singles.pq \
        --epochs 10000 \
        --lr 0.01 \
        -n test
```

This will output the Valinor fit for combination and singleton effects and several additional files, that store the Valinor settings and priors used for running Valinor, and samples from the posterior predictive distribution.

This README contains a brief guide to getting started, however for more information please see the dedicated documentation [here](https://dpohanlon.github.io/valinor/).

Model
-----

The model is a Bayesiam hierarchical model that partially-pools different observations of the CRISPR guide parameters across experiments and controls. It uses a negative-binomial distribution to describe the final counts, subject to an overdispersion parameter estimated from the experiment replicates. The structure of the parameter dependencies can be seen in the plate diagram below.

<p align="center">
  <img width="600" height="360" src="assets/valinor_plate.png">
</p>

This is inferred using variational inference in NumPyro, assuming normal posterior distributions with a low-rank approximation of the covariance matrix.

Create static plots
---
This script creates some plots to investigate Valinor's output using static images instead of using the report. This is recommended when having a lot of gene pairs and cell lines.
```bash
python produce_plots.py --combfile ../valinoroutput_processed_ito.pq --output_folder ../valinoroutputplots
```
Plots will be found in the specified `output_folder`. Within that folder there will be a `topSLgenepairs` directory, that contains plots for gene pairs that were among the top 10 most synthetic lethal gene pairs per cell line.
