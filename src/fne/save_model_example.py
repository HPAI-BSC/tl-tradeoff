'''
This is an example of the data and structure that I need for the models in brownie project.
I have a models folder which contains a folder with the model name for every model. This folder contains:

```
testnet.h5 model_description.json
```
where model description has:
```
{
  "description": "",
  "input_reshape": [
    32,
    32
  ],
  "input_tensor": "flatten_2_input:0",
  "target_tensors": [
    "dense_4/Relu:0",
    "dense_5/Softmax:0"
  ]
}
```
'''
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fne.data_handle import DataHandle
from fne.utils import resource, model_resource


def save_model_parameters(_local_model_path, _short_description, _input_reshape, _input_tensor, _target_tensors):
    description_dict = {'description': _short_description,
                        'input_reshape': _input_reshape,
                        'input_tensor': _input_tensor,
                        'target_tensors': _target_tensors}
    os.makedirs(_local_model_path, exist_ok=True)
    with open(os.path.join(_local_model_path, 'model_description.json'), 'w') as file:
        json.dump(description_dict, file)


if __name__ == '__main__':
    datasets_directory = resource('mnist')
    batch_size = 300
    data_handle = DataHandle()
    train_images, train_labels, test_images, test_labels = data_handle.load_target_task_imgs_labels(
        datasets_directory)

    # model_name = 'testnet'
    # short_description = ''
    # input_reshape = (32, 32)
    # input_tensor = 'flatten_2_input:0'
    # target_tensors = ['dense_4/Relu:0', 'dense_5/Softmax:0']

    model_name = 'pipes_model'
    short_description = 'This model was used to classify Barcelona pipes with their clogged type.'
    input_reshape = [150, 150]
    input_tensor = 'conv1_input:0'
    target_tensors = ['conv1/Relu:0', 'pool1/MaxPool:0', 'conv2/Relu:0', 'pool2/MaxPool:0', 'conv3/Relu:0',
                      'pool3/MaxPool:0', 'flatten/Reshape:0', 'fc1/Relu:0', 'absolute_output/Softmax:0']

    # path to models folder
    local_model_path = model_resource(model_name)

    save_model_parameters(local_model_path, short_description, input_reshape, input_tensor, target_tensors)
