<p align="center">
  <img width="372" height="200" src="assets/valinor_logo.png">
  <br>
  Identifying genetic interactions in CRISPR double knock-out experiments
</p>

![Tests!](https://github.com/dpohanlon/valinor/actions/workflows/python-app.yml/badge.svg)

Installation
---
Install from the Github repository. 
Make sure that you are installing valinor in an environment that has python version >3.7 but smaller than 3.11.
```bash
git clone git@github.com:dpohanlon/valinor.git
pip install -e .
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Run Valinor
---
```bash
valinor \
        --combinationsFile duspCombs_good.h5 \
        --singletonsFile duspSingles_good.h5 \
        --no-controls \
        --epochs 10000 \
        --nSamples 100 \
        -n duspoutput
```

Run processing script on Valinor output
---
This creates some dignostic plots that are relevant for the Valinor report (s. below) and a file that combines the data and Valinor output into one table for downstream usage.
```bash
cd valinor
python processvalinoroutput.py --val_combo ../combsModel_duspoutput.pq --val_single ../singlesModel_duspoutput.pq --data_combo ../duspCombs_good.h5 --data_single ../duspSingles_good.h5
```
This will create a file called `valinoroutput_processed.pq` in `valinor`.

Create the Valinor report
---
1. Create the report: use the `subsetSLpairs` flag if you have many genepairs and many cell lines, this will create the report showing data from only the top 100 synthetic lethal gene pairs + some randomly selected gene pairs across the range of Valinor scores.
```bash
python build_report.py --combfile valinoroutput_processed.pq --subsetSLpairs
```
2. Open the report
Navigate into the `valinorreport` folder and open `report.html`. Then upload the prepared datafile which is stored in `valinorreport/input/report_input.csv`.

Create static plots
---
This script creates some plots to investigate Valinor's output using static images instead of using the report. This is recommended when having a lot of gene pairs and cell lines.
```bash
python produce_plots.py --combfile valinoroutput_processed.pq
```
Plots will be found in the `plots` subfolder. Within that will also be a `topSLgenepairs` directory, that contains plots for gene pairs that were among the top 10 most synthetic lethal gene pairs in at least one cell line.