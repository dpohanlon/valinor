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

Creating the Valinor report
---
1. Create singularity image by pulling it from dockerhub
```bash
cd env
singularity pull makevalinorreport.sif docker://phweide/valinorreport
```
OR create it using the `Dockerfile` and `requirements.txt`:
```bash
docker build -t valinorreport - < env_valreport/project.dockerfile
singularity build makevalinorreport.sif docker-daemon://project:latest
```
2. Create the report
Move into `valinor` directory. Run `build_report.py` with the singularity image, like so:
```bash
singularity exec --env HDF5_USE_FILE_LOCKING=FALSE ../env_valreport/makevalinorreport.sif python build_report.py --dataset DUSP --datacombinations ../duspCombs_good.h5 --datasingletons ../duspSingles_good.h5 --valinorcombinations ../combsModel_duspoutput.pq --valinorsingletons ../singlesModel_duspoutput.pq
```
3. Open the report
Navigate into the `valinorreport` folder and open `report.html`. Then upload the prepared datafile which is stored in `valinorreport/json/report_input.csv`.