<p align="center">
  <img width="372" height="200" src="assets/valinor_logo.png">
  <br>
  Identifying genetic interactions in CRISPR double knock-out experiments
</p>

![Tests!](https://github.com/dpohanlon/valinor/actions/workflows/python-app.yml/badge.svg)

Installation
---
Install from the Github repository
```bash
git clone git@github.com:dpohanlon/valinor.git
pip install -e .
```

```

Creating the Valinor report
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
singularity exec --env HDF5_USE_FILE_LOCKING=FALSE ../makevalinorreport.sif python report_build.py --dataset DUSP --datacombinations /homes/dohanlon/data/forPaula/duspCombs_good.h5 --datasingletons /homes/dohanlon/data/forPaula/duspSingles_good.h5 --valinorcombinations /homes/dohanlon/data/forPaula/combsModel_dusp_latest.pq --valinorsingletons /homes/dohanlon/data/forPaula/singlesModel_dusp_latest.pq
```
3. Open the report
Navigate into the `valinorreport` folder and open `report.html`. Then upload the prepared datafile which is stored in `valinorreport/json/report_input.csv`.