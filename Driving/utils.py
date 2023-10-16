import cv2
import math
import random
from collections import defaultdict
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
import shutil

def draw_arrow(img, angle1, angle2, angle3):
    pt1 = (img.shape[1] // 2, img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2)))
    pt2_angle3 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle3)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle3)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle3, (255, 0, 0), 1)
    return img


def angle_diverged(angle1, angle2, angle3):
    if (abs(angle1 - angle2) > 0.2 or abs(angle1 - angle3) > 0.2 or abs(angle2 - angle3) > 0.2) and not (
                (angle1 > 0 and angle2 > 0 and angle3 > 0) or (
                                angle1 < 0 and angle2 < 0 and angle3 < 0)):
        return True
    return False


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape, dtype=None):
    return K.truncated_normal(shape, stddev=0.1, dtype=dtype)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False


def cutVideoFrame():
    # Define the path to the video file
    # Copy Video
    source_folder = './testing/video'
    destination_folder = './testing/video_output'

    # Replace 'video_file.mp4' with the name of your video file
    file_name = 'Video1.mp4'

    source_path = f'{source_folder}/{file_name}'
    destination_path = f'{destination_folder}/{file_name}'
    shutil.copy(source_path, destination_path)
    # Specify the directory where your image files are located
    video_directory = './testing/video'
    # List all image files in the directory (e.g., JPEG and PNG)
    video_file = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(('.mp4'))]
    # Create a directory to save the frames
    output_dir = './testing/video_frame_image'
    output_dir_resize = './testing/video_resize_image'
    toCleanUpFiles = os.listdir(output_dir)
    toCleanUpFiles2 = os.listdir(output_dir_resize)
    for file in toCleanUpFiles:
        file_path = os.path.join(output_dir, file)
        print(file_path)
        # Check if the file is an image (you can adjust this condition)
        if file.lower().endswith(('.jpg', '.jpeg')):
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {file_path}")      
    for file in toCleanUpFiles2:
        file_path = os.path.join(output_dir_resize, file)
        print(file_path)
        # Check if the file is an image (you can adjust this condition)
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {file_path}")   
    gen_img = (random.choice(video_file))
    os.makedirs(output_dir, exist_ok=True)
    # Open the video file
    cap = cv2.VideoCapture(gen_img)
    frame_skip = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #if frame_count:
            # Save the frame as an image
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    # Release the video capture object
    cap.release()
    print(f'{frame_count} frames extracted and saved to {output_dir}')

def generateVideo():
    print('start to resize the images')
    # Define the new dimensions (width and height) for the resized images
    new_width = 800  # Higher resolution width
    new_height = 600  # Higher resolution height
    # Create the output directory if it doesn't exist
    input_dir = './generated_inputs'
    output_dir = './testing/video_resize_image'
    os.makedirs(output_dir, exist_ok=True)
    # List all files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    # Sort the image files by modification time in ascending order
    image_files.sort(key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))
    for image_file in image_files:
        # Create the input and output file paths
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        # Open the image
        image = Image.open(input_path)
        # Resize the image with higher resolution and using bicubic interpolation
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)
        # Save the resized image to the output path
        resized_image.save(output_path)
        # Close the images
        image.close()
        resized_image.close()
    print("All images resized with higher resolution and saved in ascending date modified order.")
    video_name = './testing/video_output/output_video.mp4'
    images = [img for img in os.listdir(output_dir) if img.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    images.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
    frame = cv2.imread(os.path.join(output_dir, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(output_dir, image)))
    cv2.destroyAllWindows()
    video.release()
    print('video generated')
