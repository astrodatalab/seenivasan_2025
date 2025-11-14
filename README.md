This codebase is used to train and evaluate the CNN models (CNN-Base, CNN-TL, CNN-LoRA and CNN-Combo) in the paper titled "Combining datasets with different ground truths using Low-Rank Adaptation to generalize image-based CNN models for photometric redshift prediction" by Seenivasan et. al. and submitted to NeurIPS ML4PS 2025. 

The TransferZ-Images dataset is present in this anonymous Zenodo repository: https://zenodo.org/records/16989604?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImFlOThiYmM0LTRlMGItNDgzNS1iNjlkLTZmOWM1NzMyYmI5NyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNGJjNjk1MDM5NTI5YWFmYjFmMDJjYjlhMTUzMzRlNCJ9.CQMk2RAOmlEFDWiu6ln6uQPc8Q_-0GTfzw47p_4EBz9ZjZjGJI_lrgpdSF-nwfrVboTfMOdBfnHe-tw8nPI5Xg

The GalaxiesML dataset is present in this Zenodo repository: https://zenodo.org/records/11117528

Please create the Combo dataset yourself by merging both TransferZ-Images and GalaxiesML, and retaining GalxiesML galaxies for the 557 duplicates in both datasets. 

The conda environment for this code is in neurips_env.yaml.

To train models, replace placeholder data paths in utils.py with the location of the HDF5 files to the aformentioned three datasets. The training script is built to use MLflow (https://mlflow.org/) for model logging; run with MLflow or replace MLflow with your preferred logging method. If you use MLFlow, please use the correct tracking_uri when calling the train function. 

Each model has its own configuration in file in the configs directory. 

To train CNN-Base, run python3 train_pytorch_CNN.py --config configs/transferz_baseline_config.yaml
To train CNN-Combo, run python3 train_pytorch_CNN.py --config configs/combo_config.yaml

To train CNN-LoRA, replace the model checkpoint path in tztogmlora.yaml with the CNN-Base checkpoint path, then run python3 train_pytorch_CNN.py --config configs/tztogmlora.yaml

To train CNN-TL, replace the model checkpoint path in tztogmlora with the CNN-Base checkpoint path, then run python3 train_pytorch_CNN.py --config configs/transfer_learn_unfreeze_tztogm.yaml

To train CNN-Base-Rev, run python3 train_pytorch_CNN.py --config configs/galaxiesml_baseline_config.yaml

To train CNN-LoRA-Rev, replace the model checkpoint path in gmtotzlora.yaml with the CNN-Base-Rev checkpoint path, then run python3 train_pytorch_CNN.py --config configs/gmtotzlora.yaml

To train CNN-TL-Rev, replace the model checkpoint path in gmtotzlora with the CNN-Base-Rev checkpoint path, then run python3 train_pytorch_CNN.py --config configs/transfer_learn_unfreeze_gmtotz.yaml

The model class is contained in modules.py. The custom loader for the data is contained in data_manage.py. Recreate predictions and plots from the paper using make_predictions.py and make_plots.py, ensuring you add in the correct model checkpoint paths and save directories. photoz_utils.py is a python utility library created by the UCLA Astrophysics Data Lab, and is available as part of an installable package called datalabutils available at https://github.com/astrodatalab/datalabutils. 

Additionally, lora_ablation_test.py, hidden_dim_search.py and activation_search.py reproduce the hyperparameter tuning from the paper and can be run as python3 script_name.py