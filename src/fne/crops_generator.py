import PIL
from PIL import Image
from tensorflow.keras.utils import img_to_array


def downaxis_shape(min_axis_size, height, width):
    if height < width:  # Landscape
        target_height = min_axis_size
        down_ratio = target_height / height
        target_width = round(width * down_ratio)
    else:  # Portrait (or squared)
        target_width = min_axis_size
        down_ratio = target_width / width
        target_height = round(height * down_ratio)

    return target_height, target_width


def get_crop_images_list_with_input_reshape(img_path: str, reduction_factor=None, input_reshape: tuple = (224, 224)):
    # Load the image
    image = Image.open(img_path).convert('RGB')
    image.load()
    image = image.resize(input_reshape, Image.ANTIALIAS)
    width, height = image.size
    image_mirror = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    if reduction_factor is not None:
        crop_width = round(reduction_factor * input_reshape[0])
        crop_height = round(reduction_factor * input_reshape[1])
    else:
        crop_width = input_reshape[0]
        crop_height = input_reshape[1]

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = (width + crop_width) // 2
    bottom = (height + crop_height) // 2

    crop_boxes = []
    crop_boxes.append((0, 0, crop_width, crop_height))                               # Upper left corner
    crop_boxes.append((width - crop_width, 0, width, crop_height))                   # Upper right corner
    crop_boxes.append((0, height - crop_height, crop_width, height))                 # Lower left corner
    crop_boxes.append((width - crop_width, height - crop_height, width, height))     # Lower right corner
    crop_boxes.append((left, top, right, bottom))

    img_list = []
    for crop_box in crop_boxes:
        for img in [image, image_mirror]:
            cropped_im = img.crop(crop_box)
            # Convert PIL image to np array
            img_list.append(img_to_array(cropped_im))
    return img_list


def get_crop_images_list_with_min_axis_size(img_path: str, reduction_factor=0.875, min_axis_size=None):
    # Load the image
    image = Image.open(img_path).convert('RGB')
    image.load()
    width, height = image.size
    if min_axis_size is not None:
        target_height, target_width = downaxis_shape(min_axis_size, height, width)
        image = image.resize((target_width, target_height), Image.ANTIALIAS)
        width, height = target_width, target_height

    new_width = round(reduction_factor * width)
    new_height = round(reduction_factor * height)
    image_mirror = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    crop_boxes = []
    crop_boxes.append((0, 0, new_width, new_height))                               # Upper left corner
    crop_boxes.append((width - new_width, 0, width, new_height))                   # Upper right corner
    crop_boxes.append((0, height - new_height, new_width, height))                 # Lower left corner
    crop_boxes.append((width - new_width, height - new_height, width, height))     # Lower right corner
    crop_boxes.append((left, top, right, bottom))

    img_list = []
    for crop_box in crop_boxes:
        for img in [image, image_mirror]:
            cropped_im = img.crop(crop_box)
            # Convert PIL image to np array
            img_list.append(img_to_array(cropped_im))
    return img_list