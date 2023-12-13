#!/bin/bash
for i in {0..1}
do
  CUDA_VISIBLE_DEVICES="2"
  module load devel/cuda/12.1
  python -m atommic.cli.launch --config-path=/scratch/tmpaquaij/Atommic/ATOMMIC_private/projects/MTL/rs/SKMTEA/conf/train/ --config-name= mtlrs.yaml model.train_ds.data_path="/data/projects/utwente/recon/SKM-TEA/v1-release/json/folds/files_recon_calib-24_fold_"$i"_train.json" model.validation_ds.data_path="/data/projects/utwente/recon/SKM-TEA/v1-release/json/folds/files_recon_calib-24_fold_"$i"_val.json"
done

