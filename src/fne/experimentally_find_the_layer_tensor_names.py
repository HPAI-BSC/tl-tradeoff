"""
While testing with another models I found that I cannot use heuristics for finding the tensor names.
So I have changed the strategy for using try except until I find a better way.
"""
import os
import sys
import warnings
from tensorflow.keras import backend as K

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import model_from_json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fne.utils import resource
import tensorflow as tf
from fne.data_handle_path import DataHandlePath
from fne.full_pipeline import FullPipeline

from fne.loaded_model import LoadedModelFromSession

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def model_keras2tf():
    """
    Save pre-trained keras model.
    Keras pre-trained models download only the weights, so to load them with model.load (as does FullPipeline)
    we need to save them before with model.save
    :return:
    """
    model = VGG16(weights='imagenet',
                  include_top=True,
                  input_shape=(224, 224, 3))
    model.save(resource('vgg16_imagenet.h5'))


def _load_model(model_path):
    if '.h5' in model_path:
        model = tf.keras.models.load_model(model_path, compile=False)
    elif '.hdf5' in model_path:
        json_path = model_path.replace('.hdf5', '.json')
        json_file = open(json_path, 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(model_path)
    else:
        raise NotImplementedError
    return model


def find_operation_names(model_path):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        _load_model(model_path)
        model_with_session = LoadedModelFromSession(sess)
        operation_names = [op.name for op in model_with_session.graph.get_operations() if
                           len(op.name.split('/')) <= 2]
    return operation_names


def find_layer_tensor_names(operation_names, model_path, input_reshape, file, verbose=False):
    batch_size = 10
    input_tensor = operation_names[0] + ':0'

    data_handle = DataHandlePath()
    train_images, train_labels, test_images, test_labels = data_handle.load_target_task_imgs_labels(
        resource('mnist'))
    working_tensors = []
    for tensor in operation_names:
        target_tensors = [tensor + ':0']
        try:
            full_pipeline = FullPipeline(model_path=model_path, batch_size=batch_size,
                                         image_resize=input_reshape, input_tensor=input_tensor,
                                         target_tensors=target_tensors, data_handle=data_handle)
            best_acc, total_time = full_pipeline.run_full_pipeline(train_images, train_labels, test_images, test_labels)
            file.write('\n{}:0      {}\n'.format(tensor, best_acc))
            working_tensors.append(tensor + ':0')
            del full_pipeline, best_acc, total_time
            K.clear_session()
        except Exception as e:
            if verbose:
                print(tensor, e)
            # break
    return working_tensors


def default():
    input_reshape, model_name = (32, 32), 'testnet.h5'
    operation_names = find_operation_names(model_path=resource(model_name))
    print('All operation names: ', operation_names)
    layer_tensor_names = find_layer_tensor_names(operation_names, model_name, input_reshape, verbose=False)
    print('Valid operation names: ', layer_tensor_names)
    outputs_path = '../outputs'
    os.makedirs(outputs_path, exist_ok=True)
    print('Saving to: ', outputs_path + '/' + model_name.split('.')[0] + '.txt')
    with open(outputs_path + '/' + model_name.split('.')[0] + '.txt', 'w')as file:
        for item in layer_tensor_names:
            file.write("%s\n" % item)


if __name__ == '__main__':
    dl_models_path = '/media/raquel/5A0AA7B921A6636A/Datasets/students_models/'
    # default()
    shape_by_student = {'johannes.kruse': (400, 400), 'diego.saby': (100, 100), 'dominik.nitschmann': (224, 224),
                        'nicolas.pascual': (224, 300), 'xavier.timoneda': (80, 80),
                        'alessia.mondolo.aitor.urrutia': (125, 125), 'asier.gutierrez': (125, 125),
                        'angel.poc': (224, 300), 'marc.asenjo': (250, 250), 'roberto.fernandez.reguero': (7, 7),
                        'gisela.alessandrello': (10, 10), 'pau.li': (64, 64)
                        }
    for student in os.listdir(dl_models_path):
        student_path = os.path.join(dl_models_path, student)
        student = student_path.split('/')[-1]
        if 'NO' in student or 'DONE' in student:
            continue
        fh = open('find_tensor_names_' + student + '.txt', 'a')
        model_name = os.listdir(student_path)[0].split('.')[0]
        fh.write('Doing: {}\n'.format(student))
        print('Doing: {}\n'.format(student))
        input_reshape = shape_by_student[student]
        try:
            operation_names = find_operation_names(model_path=os.path.join(student_path, model_name + '.hdf5'))
        except Exception as e:
            fh.write('\nModel {} does not work because: {}\n'.format(model_name, e))
            continue
        fh.write('All operation names: {}'.format(operation_names))

        layer_tensor_names = find_layer_tensor_names(operation_names, os.path.join(student_path, model_name + '.hdf5'),
                                                     input_reshape, file=fh, verbose=False)
        fh.write('Valid operation names: {}\n'.format(layer_tensor_names))
        fh.write('-' * 55)
        fh.write('\n')
        fh.close()

        K.clear_session()
        # outputs_path = '../outputs'
        # os.makedirs(outputs_path, exist_ok=True)
        # print('Saving to: ', outputs_path + '/' + model_name.split('.')[0] + '.txt')
        # with open(outputs_path + '/' + model_name.split('.')[0] + '.txt', 'w')as file:
        #     for item in layer_tensor_names:
        #         file.write("%s\n" % item)
