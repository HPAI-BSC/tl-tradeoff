import argparse
import os
import time
from collections import namedtuple

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import ACTUAL_DATASET_NAMES, CLASSES_PER_DATASET, \
    SAMPLES_PER_DATASET
from model import get_model
from preprocessing import get_preprocessing_function
from utils import get_dataset, get_optimizer, preprocess_batch, get_experiments_configs

FineTuningParameters = namedtuple('FTParameters',
                                  ['dataset', 'samples_per_class', 'split',
                                   'train_split_path', 'val_split_path', 'test_split_path',
                                   'architecture', 'source', 'config', 'model_path',
                                   'optimizer', 'learning_rate', 'weight_decay', 'momentum',
                                   'min_epochs', 'max_epochs', 'batch_size', 'use_crops', 'crops_path',
                                   'output_path',
                                   'model_selection'])


def train(dataset, model, optimizer, parameters: FineTuningParameters):
    accuracy = keras.metrics.SparseCategoricalAccuracy()
    losses = []
    preprocessing_function = get_preprocessing_function(parameters.architecture)

    # For each step in epoch
    for x in dataset.shuffle(100000).batch(parameters.batch_size).as_numpy_iterator():
        with tf.GradientTape() as tape:
            # Load and preprocess images
            images, labels = preprocess_batch(x, preprocessing_function)
            # Forward pass
            output = model(images, training=True)
            # Compute and save loss
            loss_value = keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, output)
            losses.append(loss_value)
        # Backward pass
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Compute accuracy
        accuracy.update_state(labels, output)

    return float(accuracy.result().numpy()) * 100, float(tf.math.reduce_mean(losses).numpy())


def val(dataset, model, num_classes, n_crops, parameters: FineTuningParameters):
    correct_predictions = np.zeros(num_classes)
    total_predictions = np.zeros(num_classes)
    losses = []
    preprocessing_function = get_preprocessing_function(parameters.architecture)

    # For each step in epoch
    for x in dataset.batch(4 * n_crops).as_numpy_iterator():
        # Load and preprocess images
        images, labels = preprocess_batch(x, preprocessing_function)
        # Forward pass
        output = model(images, training=False)
        # Compute accuracy using voting mechanism for the 10 crops of each image
        output = tf.reshape(output, (-1, n_crops, num_classes))
        labels = tf.reshape(labels, (-1, n_crops))
        for preds, l in zip(output, labels):
            values, _, counts = tf.unique_with_counts(tf.math.argmax(preds, axis=1))
            total_predictions[int(l[0])] += 1
            correct_predictions[int(l[0])] += 1 if values[tf.math.argmax(counts)] == int(l[0]) else 0
        # Compute and save loss
        loss_value = keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, output)
        losses.append(loss_value)

    return float(np.mean(correct_predictions / total_predictions)) * 100, float(tf.math.reduce_mean(losses).numpy())


def run_one_ft_training(parameters: FineTuningParameters, experiment_id):
    mlflow.set_tracking_uri(parameters.output_path)
    mlflow.start_run(experiment_id=experiment_id)

    train_dataset = get_dataset(split_path=parameters.train_split_path,
                                crop_dataset_path=parameters.crops_path, n_crops=10 if parameters.use_crops else 1)
    val_dataset = get_dataset(split_path=parameters.val_split_path,
                              crop_dataset_path=parameters.crops_path, n_crops=10 if parameters.use_crops else 1)
    test_dataset = get_dataset(split_path=parameters.test_split_path,
                               crop_dataset_path=parameters.crops_path, n_crops=10 if parameters.use_crops else 1)

    model = get_model(model=parameters.architecture, source=parameters.source, model_path=parameters.model_path,
                      num_classes=CLASSES_PER_DATASET[parameters.dataset], config=parameters.config)
    optimizer = get_optimizer(parameters.optimizer, parameters.learning_rate,
                              parameters.momentum, parameters.weight_decay)

    mlflow.log_params({
        # MODEL
        'Architecture': parameters.architecture,
        'Source': parameters.source,
        'Model': parameters.model_path,
        # TRAINING PARAMETERS
        'Dataset': parameters.dataset,
        'Using crops': f'{parameters.use_crops}',
        'Crops path': f'{parameters.crops_path}',
        'Samples per class': f'{parameters.samples_per_class}',
        'Split': parameters.split,
        'Train split': parameters.train_split_path,
        'Validation split': parameters.val_split_path,
        'Test split': parameters.test_split_path,
        'Minimum epochs': f'{parameters.min_epochs}',
        'Maximum epochs': f'{parameters.max_epochs}',
        'Model selection': f'{parameters.model_selection}',
        'Batch size': f'{parameters.batch_size}',
        # HYPERPARAMETERS
        'Configuration': parameters.config,
        'Optimizer': parameters.optimizer,
        'Learning rate': f'{parameters.learning_rate}',
        'Weight decay': f'{parameters.weight_decay}' if parameters.optimizer == 'sgd' else '-',
        'Momentum': f'{parameters.momentum}',
        # OTHERS
        'Job number': f'{args.job}'
    })

    best_weights = None
    maximum_val_acc = -1
    val_losses = []
    initial_time = time.time()
    for epoch in range(parameters.max_epochs):
        train_initial_time = time.time()
        train_acc, train_loss = train(train_dataset, model, optimizer, parameters)
        val_initial_time = time.time()
        val_acc, val_loss = val(val_dataset, model, CLASSES_PER_DATASET[parameters.dataset],
                                10 if parameters.use_crops else 1, parameters)
        val_final_time = time.time()
        mlflow.log_metrics(
            {
                'Train accuracy': train_acc,
                'Train loss': train_loss,
                'Train epoch time': val_initial_time - train_initial_time,
                'Val accuracy': val_acc,
                'Val loss': val_loss,
                'Val epoch time': val_final_time - val_initial_time
            },
            epoch)

        # If we get the best val acc until now, save the weights of the model
        if val_acc > maximum_val_acc:
            best_weights = model.get_weights()
            maximum_val_acc = val_acc

        val_losses.append(val_loss)
        # Early stopping policy:
        # Stop if test loss does not improve in 3 consecutive epochs
        if epoch >= 10 and val_losses[-4] <= val_losses[-3] <= val_losses[-2] <= val_losses[-1]:
            break

    if parameters.model_selection:
        mlflow.log_metrics({
            'Total time': time.time() - initial_time
        })
    else:
        # Get the best weights according to val accuracy and evaluate the test split
        model.set_weights(best_weights)
        test_initial_time = time.time()
        test_acc, test_loss = val(test_dataset, model, CLASSES_PER_DATASET[parameters.dataset],
                                  10 if parameters.use_crops else 1, parameters)
        test_final_time = time.time()
        mlflow.log_metrics({
            'Test accuracy': test_acc,
            'Test loss': test_loss,
            'Test time': test_final_time - test_initial_time,
            'Total time': test_final_time - initial_time
        })

    mlflow.end_run()


def run_ft_scalability(architecture: str, source: str, model_path: str,
                       layer_config: int, optimizer: str, lr: float, wd: float, mom: float,
                       dataset: str, max_epochs: int, model_selection: bool, batch_size: int,
                       use_crops: bool, precomputed_crops_path: str,
                       split_path: str, n_splits: int,
                       mlflow_path: str, job: int):
    mlflow.set_tracking_uri(mlflow_path)

    experiment_id = mlflow.set_experiment(
        f'Fine-tuning'
    ).experiment_id

    if model_selection:
        configs = get_experiments_configs([SAMPLES_PER_DATASET[ACTUAL_DATASET_NAMES[dataset]][-1]], n_splits)
    else:
        configs = get_experiments_configs(SAMPLES_PER_DATASET[ACTUAL_DATASET_NAMES[dataset]], n_splits)

    for config in configs:
        run_one_ft_training(
            FineTuningParameters(
                dataset=ACTUAL_DATASET_NAMES[dataset], samples_per_class=config['samples_per_class'],
                split=config['split'],
                train_split_path=os.path.join(
                    split_path, f"train_samples_{config['samples_per_class']:0=2d}_split_{config['split']}.csv"
                ),
                val_split_path=os.path.join(split_path, 'val.csv'),
                test_split_path=os.path.join(split_path, 'test.csv'),
                architecture=architecture, source=source, config=layer_config, model_path=model_path,
                optimizer=optimizer, learning_rate=lr, weight_decay=wd, momentum=mom,
                min_epochs=10, max_epochs=max_epochs, batch_size=batch_size,
                use_crops=use_crops, crops_path=precomputed_crops_path,
                output_path=mlflow_path,
                model_selection=model_selection
            ),
            experiment_id
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # MODEL
    parser.add_argument('-a', '--architecture', help='Name of the architecture to be used', default='VGG16')
    parser.add_argument('--source', help='Name of the source task', default='ImageNet')
    parser.add_argument('-m', '--model', help='Path to the model weights (.h5) to load.',
                        default='/path/to/model/weights/vgg16_imagenet.h5')
    # TRAINING PARAMETERS
    parser.add_argument('-d', '--dataset', help='Name of the target dataset', default='flowers102')
    parser.add_argument('--use-crops', action='store_true', help='Use precomputed 10 crops per image. If flag is not '
                                                                 'present, code uses the original images. Otherwise, '
                                                                 'it uses the precomputed 10 crops stored at '
                                                                 'the path specified with --precomputed-crops.')
    parser.add_argument('--precomputed-crops', help='Path to the precomputed 10 crops of the target dataset. Must have '
                                                    'the same structure as the target dataset folder '
                                                    '(root/split/label/image). Image crops must be called '
                                                    'imgname_crop_n.extension.',
                        default='/path/to/flowers102_crops')
    parser.add_argument('-e', '--epochs', type=int, help='Amount of epochs of the trainings', default=40)
    parser.add_argument('--model-selection', action='store_true',
                        help='Perform model selection. If not present, generate a few-shot learning curve.')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=64)
    parser.add_argument('--split-path', help='Path to the folder where the split CSVs are.',
                        default='/path/to/splits')
    parser.add_argument('--splits', type=int, help='Amount of splits for each sample per class. Only used if '
                                                   '--model-selection is included.',
                        default=5)
    # HYPERPARAMETERS
    parser.add_argument('-c', '--configuration', type=int, help='Architecture-specific configuration (1, 2 or 3).',
                        default=1)
    parser.add_argument('--optimizer', help='Optimizer for the main network', default='sgd')
    parser.add_argument('--learning-rate', type=float, help='Learning rate for the main network', default=0.01)
    parser.add_argument('--weight-decay', type=float, help='Weight decay value for the main network', default=0.0001)
    parser.add_argument('--momentum', type=float, help='Momentum value for the main network', default=0.9)
    # OTHERS
    parser.add_argument('--mlflow-path', help='Path where the MLFlow artifacts will be stored at.',
                        default='../mlruns')
    parser.add_argument('--job', type=int, help='ID of the job that is executing the experiment '
                                                '(e.g. if running in a job-based environment).',
                        default=0)
    args = parser.parse_args()

    run_ft_scalability(args.architecture, args.source, args.model,
                       args.configuration, args.optimizer, args.learning_rate, args.weight_decay, args.momentum,
                       args.dataset, args.epochs, args.model_selection, args.batch_size,
                       args.use_crops is not None, args.precomputed_crops, args.split_path, args.splits,
                       args.mlflow_path, args.job)
