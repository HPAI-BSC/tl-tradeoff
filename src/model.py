from tensorflow import keras


def get_vgg16_imagenet(model_path, source, n_classes, config):
    # Create the VGG16 from the model path
    vgg = keras.applications.VGG16(include_top=True, weights=model_path,
                                   input_tensor=keras.Input(shape=(224, 224, 3)),
                                   classes=365 if source == 'Places365' else 1000)
    # Remove the FC layers
    vgg = keras.models.Sequential(vgg.layers[:-3])
    # Replace them by smaller FC layers
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        vgg,
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    # Freeze the first 4*config layers
    # (0: 0 layers, 1: 4 layers, 2: 8 layers)
    VGG_CONV_LAYERS = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
    for layer in VGG_CONV_LAYERS[:config * 4]:
        model.layers[0].layers[layer].trainable = False
    return model


def get_resnet50(model_path, source, n_classes, config):
    # Load the ResNet50 from the model path
    resnet = keras.applications.ResNet50(include_top=True, weights=model_path,
                                         input_tensor=keras.Input(shape=(224, 224, 3)),
                                         classes=365 if source == 'Places365' else 1000)
    model = keras.Sequential([
        keras.Model(resnet.input, resnet.layers[-2].output),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    # Freeze the first x layers depending on config
    # (1: 12 layers, 2: 25 layers, 3: 37 layers)
    # Taking into account that there are some extra layers
    layers_to_freeze = {1: 13, 2: 28, 3: 40}
    RESNET_CONV_LAYERS = [2, 7, 10, 13, 14,
                          19, 22, 25, 29, 32, 35, 39, 42, 45, 46,
                          51, 54, 57, 61, 64, 67, 71, 74, 77, 81, 84, 87, 88,
                          93, 96, 99, 103, 106, 109, 113, 116, 119, 123, 126, 129, 133, 136, 139, 143, 146, 149, 150,
                          155, 158, 161, 165, 168, 171]
    for layer in RESNET_CONV_LAYERS[:layers_to_freeze[config]]:
        model.layers[0].layers[layer].trainable = False
    return model


def get_resnet152(model_path, source, n_classes, config):
    # Load the ResNet152 from the model path
    if source == 'ImageNet':
        resnet = keras.applications.ResNet152(include_top=True, weights=model_path,
                                              input_tensor=keras.Input(shape=(224, 224, 3)),
                                              classes=1000)
    elif source == 'Places365':
        resnet = keras.applications.ResNet152(classes=365)
        resnet.load_weights(model_path, by_name=True)
    model = keras.Sequential([
        keras.Model(resnet.input, resnet.layers[-2].output),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    # Freeze the first x layers depending on config
    # (1: 38 layers, 2: 76 layers, 3: 114 layers)
    RESNET152_CONV_LAYERS = [
        2, 7, 10,  # 13,
        14, 19, 22, 25, 29, 32, 35, 39, 42,  # 45,
        46, 51, 54, 57, 61, 64, 67, 71, 74, 77, 81, 84, 87, 91, 94, 97, 101, 104, 107, 111, 114, 117, 121, 124,  # 127,
        128, 133, 136, 139, 143, 146, 149, 153, 156, 159, 163, 166, 169, 173, 176, 179, 183, 186, 189, 193, 196, 199,
        203, 206, 209, 213, 216, 219, 223, 226, 229, 233, 236, 239, 243, 246, 249, 253, 256, 259, 263, 266, 269, 273,
        276, 279, 283, 286, 289, 293, 296, 299, 303, 306, 309, 313, 316, 319, 323, 326, 329, 333, 336, 339, 343, 346,
        349, 353, 356, 359, 363, 366, 369, 373, 376, 379, 383, 386, 389, 393, 396, 399, 403, 406, 409, 413, 416, 419,
        423, 426, 429, 433, 436, 439, 443, 446, 449, 453, 456, 459, 463, 466, 469, 473, 476, 479, 483, 486,  # 489,
        490, 495, 498, 501, 505, 508, 511
    ]
    for layer in RESNET152_CONV_LAYERS[:config * 38]:
        model.layers[0].layers[layer].trainable = False
    return model


def get_model(model: str = 'VGG16', source: str = 'ImageNet', model_path: str = '', num_classes: int = 1000,
              config: int = 0):
    if model == 'VGG16':
        return get_vgg16_imagenet(model_path, source, num_classes, config)
    if model == 'ResNet50':
        return get_resnet50(model_path, source, num_classes, config)
    if model == 'ResNet152':
        return get_resnet152(model_path, source, num_classes, config)
    else:
        raise NotImplementedError
