import argparse
import json
import os
from collections import namedtuple

import mlflow
from sklearn.linear_model import SGDClassifier

from constants import ACTUAL_DATASET_NAMES, SAMPLES_PER_DATASET, CLASSIFIERS
from fne.data_handle_path import DataHandlePath
from fne.model import ModelHandleCluster
from fne.pipeline import Pipeline
from utils import get_experiments_configs, get_img_paths_and_labels

FEParameters = namedtuple('FNEParameters',
                          ['dataset', 'samples_per_class', 'split',
                           'train_split_path', 'val_split_path', 'test_split_path',
                           'model_folder', 'model_name', 'model_description',
                           'classifier',
                           'batch_size', 'use_crops', 'crops_path',
                           'output_path',
                           'model_selection'])


def run_one_fe_training(parameters: FEParameters, experiment_id):
    mlflow.set_tracking_uri(parameters.output_path)
    mlflow.start_run(experiment_id=experiment_id)

    with open(parameters.model_description, 'r') as json_file:
        model_description = json.load(json_file)
    layer_amount = len(model_description['target_tensors'])

    train_images, train_labels = get_img_paths_and_labels(parameters.train_split_path)
    test_images, test_labels = get_img_paths_and_labels(
        parameters.val_split_path if parameters.model_selection else parameters.test_split_path
    )

    mlflow.log_params({
        # MODEL
        'Model': parameters.model_name,
        # TRAINING PARAMETERS
        'Dataset': parameters.dataset,
        'Using crops': f'{parameters.use_crops}',
        'Crops path': f'{parameters.crops_path}',
        'Samples per class': f'{parameters.samples_per_class}',
        'Model selection': f'{parameters.model_selection}',
        'Split': f'{parameters.split}',
        'Train split': parameters.train_split_path,
        'Validation split': parameters.val_split_path,
        'Test split': parameters.test_split_path,
        # HYPERPARAMETERS
        'Model description': parameters.model_description,
        'Extracted layers': f'{layer_amount}',
        'Classifier': str(type(parameters.classifier)),
        # OTHERS
        'Job number': f'{args.job}'
    })
    if isinstance(parameters.classifier, SGDClassifier):
        mlflow.log_param('tol', str(parameters.classifier.tol))

    d = DataHandlePath()
    m = ModelHandleCluster(
        models_path=parameters.model_folder,
        model_name=parameters.model_name,
        model=None,
        model_description_path=parameters.model_description
    )
    p = Pipeline(
        model_handle=m,
        data_handle=d,
        batch_size=parameters.batch_size,
        n_crops=10 if parameters.use_crops else 1,
        load_crops=True,
        reduction_factor=None,
        min_axis_size=None
    )
    results = p.run_single_pipeline(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        fne='tf',
        classifier_method=parameters.classifier,
        include_l2=False
    )

    mlflow.log_metrics({
        'Train extraction time': results['time'][0]['extract_train_time'],
        'Test extraction time': results['time'][0]['extract_test_time'],
        'Postprocessing time': results['time'][0]['post_processings_time'],
        'Classifier training time': results['time'][0]['fne_time'],
        'Total execution time': results['time'][0]['extract_train_time']
                                + results['time'][0]['extract_test_time']
                                + results['time'][0]['post_processings_time']
                                + results['time'][0]['fne_time'],
        'Val accuracy' if parameters.model_selection else 'Test accuracy': results['fne'][0]
    })

    mlflow.end_run()


def run_fe_scalability(model_path: str,
                       model_description: str, classifier: str,
                       dataset: str, model_selection: bool, n_splits: int, split_path: str,
                       use_crops: bool, precomputed_crops_path: str,
                       mlflow_path: str, job: int):
    mlflow.set_tracking_uri(mlflow_path)

    model_folder, model_file = os.path.split(model_path)

    experiment_id = mlflow.set_experiment(
        f'Feature extraction'
    ).experiment_id

    if model_selection:
        configs = get_experiments_configs([SAMPLES_PER_DATASET[ACTUAL_DATASET_NAMES[dataset]][-1]], n_splits)
    else:
        configs = get_experiments_configs(SAMPLES_PER_DATASET[ACTUAL_DATASET_NAMES[dataset]], n_splits)

    for config in configs:
        run_one_fe_training(
            FEParameters(
                dataset=ACTUAL_DATASET_NAMES[dataset], samples_per_class=config['samples_per_class'],
                split=config['split'], use_crops=use_crops, crops_path=precomputed_crops_path,
                train_split_path=os.path.join(
                    split_path, f"train_samples_{config['samples_per_class']:0=2d}_split_{config['split']}.csv"
                ),
                val_split_path=os.path.join(split_path, 'val.csv'),
                test_split_path=os.path.join(split_path, 'test.csv'),
                model_folder=model_folder, model_name=model_file, model_description=model_description,
                classifier=CLASSIFIERS[classifier], batch_size=2,
                output_path=mlflow_path,
                model_selection=model_selection
            ),
            experiment_id
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # MODEL
    parser.add_argument('-m', '--model', help='Path to the model weights (.h5) to load.',
                        default='/path/to/model/weights/weights.h5')
    # HYPERPARAMETERS
    parser.add_argument('--model-description', help='Path to the JSON file description of the model. The JSON must '
                                                    'have the structure {"input_reshape": input_shape, "input_tensor": '
                                                    'name_of_tensor, "target_tensors": [tensor_to_extract1, '
                                                    'tensor_to_extract2, ...]}.',
                        default='/path/to/model/description/model_description.json')
    parser.add_argument('--classifier', help='Classifier to use after extracting the features. Must be one of the '
                                             'specified in constants.py',
                        default='LinearSVC')
    # TRAINING PARAMETERS
    parser.add_argument('-d', '--dataset', help='Path to the target dataset. Must have the structure '
                                                'root/split/label/image.',
                        default='/path/to/flowers102')
    parser.add_argument('--use-crops', action='store_true', help='Use precomputed 10 crops per image. If flag is not '
                                                                 'present, code uses the original images. Otherwise, '
                                                                 'it uses the precomputed 10 crops stored at '
                                                                 'the path specified with --precomputed-crops.')
    parser.add_argument('--precomputed-crops', help='Path to the precomputed 10 crops of the target dataset. Must have '
                                                    'the same structure as the target dataset folder '
                                                    '(root/split/label/image). Image crops must be called '
                                                    'imgname_crop_n.extension.',
                        default='/path/to/flowers102_crops')
    parser.add_argument('--model-selection', action='store_true',
                        help='Perform a single model selection with the specified hyperparameters. If flag is not '
                             'present, generate a few-shot learning curve for the dataset, running several jobs for '
                             'each sample per class value.')
    parser.add_argument('--split-path', help='Path to the folder where the split CSVs are.',
                        default='/path/to/splits')
    parser.add_argument('--splits', type=int, help='Amount of splits for each sample per class. Only used if '
                                                   '--model-selection is included.',
                        default=5)
    # OTHERS
    parser.add_argument('--mlflow-path', help='Path where the MLFlow artifacts will be stored at.',
                        default='../mlruns')
    parser.add_argument('--job', type=int, help='ID of the job that is executing the experiment '
                                                '(e.g. if running in a job-based environment).',
                        default=0)
    args = parser.parse_args()

    run_fe_scalability(args.model,
                       args.model_description, args.classifier,
                       args.dataset, args.model_selection, args.splits, args.split_path,
                       args.use_crops is not None, args.precomputed_crops,
                       args.mlflow_path, args.job)
