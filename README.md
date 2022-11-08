# [When \& How to Transfer with Transfer Learning]()

This repository contains the code implementation for the experiments in the paper [When \& How to Transfer with Transfer Learning]().

> In deep learning, transfer learning (TL) has become the de facto approach when dealing with image related tasks. Visual features learnt for one task have been shown to be reusable for other tasks, improving performance significantly. By reusing deep representations, TL enables the use of deep models in domains with limited data availability, limited computational resources and/or limited access to human experts. Domains which include the vast majority of real-life applications. This paper conducts an experimental evaluation of TL, exploring its trade-offs with respect to performance, environmental footprint, human hours and computational requirements. Results highlight the cases were a cheap feature extraction approach is preferable, and the situations where an expensive fine-tuning effort may be worth the added cost. Finally, a set of guidelines on the use of TL are proposed.

## Running the code

### Clone the repository

```
git clone 
```

### Install dependencies

The following commands create a virtual environment and install the dependencies inside it.

```
cd tl-tradeoff
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r ./requirements.txt
```

### Prepare data

For the code in this repository to run correctly, some data need to be present:
- CSV files containing the images and labels included in each train, val and test split of a dataset.
  They must have the same structure and naming convention as the ones included in the repository.
- The images in the dataset need to be distributed following the `root/<split>/<label>/image.ext` structure.
  `<split>` is one of "train", "test" or "val". The `<label>` names are the same as in the CSV files.
- If you want to use 10 pre-computed crops (i.e. 4 corners and center, with horizontal mirror too), these need to be present
  as the code does not compute them itself. The folder structure must be the same as in the original dataset
  (i.e. each crop must be in the same  `root_crops/<split>/<label>/image.ext` as the original image). The folder containing
  the crops may or may not be the same as the one containing the original images. The cropped images must follow the
  naming convention `image_crop_i.ext`, where the original image was named `image.ext`.
- A model description JSON must be included for the feature extraction experiments.
  It must have the same structure as the ones included in this repository.
  

### Run the FE and FT codes

The files `feature_extraction.py` and `fine_tuning.py` allow you to run a feature extraction or fine-tuning experiment, respectively.
They allow to run both model selection experiments (one training) or the few-shot learning runs (several trainings on different amounts of samples per class).
Use the flag `--model-selection` to run the former, do not add it for the latter.

Add the `--use-crops` flag to use 10 pre-computed crops (i.e. 4 corners and center, with horizontal mirror too) per image in the dataset.
The code requires the path to the pre-computed crops.

Both codes use the MLFlow library to log the metrics and parameters of each run.
Use `--mlflow-path` to specify the path in which the artifacts will be stored.

You can find more information on each flag using `-h` or `--help`. 

```
python3 src/feature_extraction.py [-h] [--use-crops] [--model-selection]
                                  --model /path/to/model/weights.h5
                                  /path/to/model/description.json
                                  --classifier classifiername --dataset /path/to/dataset
                                  --precomputed-crops /path/to/dataset_crops 
                                  --split-path  /path/to/splits --splits nsplits
                                  --mlflow-path ../mlruns --job jobid
```

```
python3 src/fine_tuning.py [-h] [--use-crops] [--model-selection]
                           --architecture architecture --source source
                           --model /path/to/model/weights.h5 --dataset datasetname
                           --precomputed-crops /path/to/dataset_crops --epochs maxepochs
                           --batch-size bs --split-path  /path/to/splits --splits nsplits
                           --configuration c --optimizer optimizer --learning-rate lr
                           --weight-decay wd --momentum mom --mlflow-path ../mlruns --job jobid
```

### See the results

Using the command `mlflow ui` in the directory in which the MLFlow artifacts were stored in will open a graphical UI
in which the results will be shown.
Results will be categorized in two MLFlow experiments: `Feature extraction` and `Fine-tuning`, depending on the method that is being used.