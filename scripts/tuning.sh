#!/bin/bash

# Tuning 
cd /storage/bt20d204/yeast_growth_pred
conda activate DL

python main_code/tune_multiview.py data=cancer_data  model=cancer_PRISM_clf

echo "\n PRISM_19Q4_lung_global_extreme Done"

python main_code/tune_multiview.py data=cancer_data \
 data.datamodule.path='${paths.data_dir}cancer/PRISM_19Q4_lung/clf' \
 model=cancer_PRISM_clf \
 data.metadata.groupname=PRISM_19Q4_lung_local_extreme