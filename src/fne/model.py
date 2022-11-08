import json
import os
import sys

import torch.nn as nn
import tensorflow as tf
from hpai_utils.minio import Minio
from io import BytesIO
import shutil
from tensorflow.keras.models import model_from_json
import torch
import numpy as np

tf.compat.v1.disable_v2_behavior()


def get_weights_and_bias(model_path):
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = torch.load(model_path)
    weights_bias = checkpoint['model_state_dict']

    weights = []
    bias = []

    for key, value in weights_bias.items():
        if 'features' in key:
            if 'weight' in key:
                weights.append(np.transpose(value.cpu().numpy()))
            elif 'bias' in key:
                bias.append(np.transpose(value.cpu().numpy()))

    return weights, bias


class ModelHandle():
    def __init__(self):
        pass

    def get_model_paths(self):
        pass

    def load_model(self):
        pass

    def load_model_description(self):
        pass


class ModelHandleMinio(ModelHandle):
    def __init__(self, model_minio: Minio, models_path, model_name, model, model_description_path):
        self.model = None
        self.minio = model_minio
        self.model_name = model_name
        self.model_description_path = model_description_path
        if model is not None:
            self.model = model
        elif models_path is not None and model_name is not None:
            if '.h5' not in self.model_name:
                raise NotImplementedError
            else:
                self.models_path = models_path
                self.model_path, model_description_path = self.get_model_paths()
                if self.model_description_path is None:
                    self.model_description_path = model_description_path
        else:
            raise Exception('You must provide the torch model object or the models path and '
                            'the model name in case of using keras/tensorflow.')
        super().__init__()

    def get_model_paths(self):
        model_folder = os.path.join(self.models_path, self.model_name[:-3])
        model_path = os.path.join(model_folder, self.model_name)
        model_description_path = os.path.join(model_folder, 'model_description.json')
        return model_path, model_description_path

    def load_model(self):
        if self.model_path is None and self.model_name is None:
            return self.model
        if '.h5' in self.model_path:
            b_string = self.minio.load_bytes(self.model_path)
            with open('model.h5', 'wb') as f:
                shutil.copyfileobj(BytesIO(b_string), f, length=131072)
            f.close()
            tf.compat.v1.keras.models.load_model("model.h5", custom_objects=None, compile=False)
            if os.path.exists("model.h5"):
                os.remove("model.h5")
        else:
            raise NotImplementedError

    def load_model_description(self):
        model_description = self.minio.load_json(self.model_description_path)
        return model_description


class ModelHandleCluster(ModelHandle):
    def __init__(self, models_path, model_name, model, model_description_path):
        self.model = None
        self.model_name = model_name
        self.model_description_path = model_description_path
        if model is not None:
            self.model = model
        elif models_path is not None and model_name is not None:
            if '.h5' not in self.model_name:
                raise NotImplementedError
            else:
                self.models_path = models_path
                self.model_path, model_description_path = self.get_model_paths()
                if self.model_description_path is None:
                    self.model_description_path = model_description_path
        else:
           raise Exception('You must provide the torch model object or the models path and '
                           'the model name in case of using keras/tensorflow.')

        super().__init__()

    def get_model_paths(self):
        model_folder = os.path.join(self.models_path, self.model_name.split('.')[0])
        model_path = os.path.join(model_folder, self.model_name)
        model_description_path = os.path.join(model_folder, 'model_description.json')
        return model_path, model_description_path

    def load_model(self):
        if self.model is not None:
            return self.model
        if '.h5' in self.model_path:
            model = tf.keras.models.load_model(self.model_path, compile=False)
        elif '.hdf5' in self.model_path:
            json_path = self.model_path.replace('.hdf5', '.json')
            json_file = open(json_path, 'r')
            model_json = json_file.read()
            json_file.close()
            model = model_from_json(model_json)
            model.load_weights(self.model_path)
        else:
            raise NotImplementedError
        return model

    def load_model_description(self):
        with open(self.model_description_path, 'r') as json_file:
            model_description = json.load(json_file)
        return model_description
