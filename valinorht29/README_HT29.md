
Usage
---
TBD

Run on HT-29
---
Conda environment: conda env export --no-builds > valinor-gpu.yml

Folder: `valinorht29`
```
bsub -q gpu -gpu "num=1:gmem=1" -M 16000 -o ~/logs/encore_valinor_ht29.log python gemini-numpyro-new-encore-new-process-ht29.py -n ht29_10k_narrow_sameGenes --combsFile /homes/dohanlon/data/ht29/encore-All-combs-All.pq --singlesFile /homes/dohanlon/data/ht29/encore-All-singles-All.pq --controlsFile /homes/dohanlon/data/ht29/encore-All-controls-All.pq
```

Report
---
Conda environment: conda env export --no-builds > valinor-report.yml

```
python build_report.py
```

-> outputs `report.html` + several files in subfolders (`json/scoreData_combined_sample.csv` is the file containing the data)