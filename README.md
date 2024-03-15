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
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Run Valinor
---
```bash
valinor \
        --combinationsFile itoCombs.pq \
        --singletonsFile itoSingles.pq \
        --no-controls \
        --epochs 10000 \
        --nSamples 100 \
        -n ito
```
This will output the Valinor fit for combination and singleton effects and several additional files, that store the Valinor settings and priors used for running Valinor.

Now there are two options to build the report. Either using the executable or just running the python scripts.

Option 1: Use `valinorreport` executable to create the valinor report
---
This takes the Valinor results and the original input files to generate a user-friendly processed table `valinoroutput_processed_ito.pq` (`--output_file` flag). This table is necessary for creating the report html. The folder in which report files should be saved needs to specified with `--report_folder`. 
```bash
valinorreport \
    --val_combo combsModel_ito.pq \
    --val_single singlesModel_ito.pq \
    --data_combo itoCombs.pq \
    --data_single itoSingles.pq \
    --valinorLossFile valinor_loss_ito.svg \
    --valinorConfigFile valinorrun_ito.json \
    --output_file valinoroutput_processed_ito.pq \
    --report_folder valinorreport \
    --subsetSLpairs
```
If custom priors were used to run Valinor, then they should be stored in a `.json` file and given to the processing script using the `--valinorPriorFile` flag. If the flag is not used, then Valinor's default priors are used.

Use the `subsetSLpairs` flag if you have many genepairs and many cell lines, this will create the report showing data from only the top 100 synthetic lethal (based on the Valinor scoer) gene pairs + some randomly selected gene pairs across the range of Valinor scores.

Once the command has finished running, navigate to the `valinorreport` and download the whole folder to your local computer if you ran valinor on a cluster.

Open `report.html` created inside the `valinorreport` folder. On the opened report page upload the prepared datafile which is stored in `valinorreport/input/report_input.csv`.


Option 2: Use individual scripts to create the valinor report
--- 
1. Run processing script on Valinor output
This creates some dignostic plots that are relevant for the Valinor report (s. below) and a file that combines the data and Valinor output into one table for downstream usage.
```bash
cd valinor
python processvalinoroutput.py \
    --val_combo ../combsModel_ito.pq \
    --val_single ../singlesModel_ito.pq \
    --data_combo ../itoCombs.pq \
    --data_single ../itoSingles.pq \
    --output_file ../valinoroutput_processed_ito.pq \
    --report_folder ../valinorreport
```
This will create a table stored in `valinoroutput_processed_ito.pq` that contains all necessary data and Valinor outputs averaged over replicates with singletons and combinations merged. For singleton data values were merged across control pairings so produce values per singleton gene.

If `--report_folder` is specified this needs to point to the `valinorreport` folder that was downloaded with this repository, then plots and output files are stored in the folder to be used for the subsequent report generation. If it is not specified, then no report outputs will be generated.

If custom priors were used to run Valinor, then they should be stored in a `.json` file and given to the processing script using the `--valinorPriorFile` flag. If the flag is not used, then Valinor's default priors are used.

2. Create the Valinor report

Create the report: use the `subsetSLpairs` flag if you have many genepairs and many cell lines, this will create the report showing data from only the top 100 synthetic lethal (based on the Valinor scoer) gene pairs + some randomly selected gene pairs across the range of Valinor scores.
```bash
cd valinor
python build_report.py \
      --combfile ../valinoroutput_processed.pq \
      --valinorLossFile ../valinor_loss_duspoutput.svg \
      --valinorConfigFile ../valinorrun_duspoutput.json \
      --report_folder ../valinorreport \
      --subsetSLpairs
```
Open `report.html` created inside the `valinorreport` folder. On the opened report page upload the prepared datafile which is stored in `valinorreport/input/report_input.csv`.

Create static plots
---
This script creates some plots to investigate Valinor's output using static images instead of using the report. This is recommended when having a lot of gene pairs and cell lines.
```bash
python produce_plots.py --combfile ../valinoroutput_processed.pq --output_folder ../valinoroutputplots
```
Plots will be found in the specified `output_folder`. Within that folder there will be a `topSLgenepairs` directory, that contains plots for gene pairs that were among the top 10 most synthetic lethal gene pairs per cell line.