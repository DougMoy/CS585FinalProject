import numpy as np
from PIL import Image
from tqdm import tqdm


# Helper function to parse the tripod sequence file
def parse_tripod_seq_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        image_dims = list(map(int, lines[0].split()))
        num_frames = list(map(int, lines[1].split()))
        frames_360 = list(map(int, lines[4].split()))
        frontal_frames = list(map(int, lines[5].split()))
        rotation_sense = list(map(int, lines[6].split()))
    return image_dims, num_frames, frames_360, frontal_frames, rotation_sense


# Function to load and resize an image using PIL
def load_and_resize_image(filename, img_height, img_width):
    # Open the image file
    img = Image.open(filename)
    # Resize the image
    img = img.resize((img_width, img_height))
    # Convert the image to a numpy array
    img_array = np.array(img)
    return img_array


# Function to load and preprocess image and bbox data
def load_and_preprocess_data(base_path, sequence_ids, img_width, img_height,
                             frames_per_seq, frames_360, frontal_frames, rotation_sense):
    data = []
    labels = []
    bboxes = []

    for i, seq_id in enumerate(tqdm(sequence_ids, desc='Loading sequences')):
        num_frames = frames_per_seq[i]
        num_frames_360 = frames_360[i]
        frontal_frame = frontal_frames[i]
        sense = rotation_sense[i]
        bbox_path = f"{base_path}/bbox_{seq_id:02d}.txt"
        bbox_data = np.loadtxt(bbox_path, delimiter=' ')

        # for frame_id in tqdm(range(1, num_frames + 1), desc=f'Processing seq {seq_id}', leave=False):
        for frame_id in range(1, num_frames + 1):
            filename = f"{base_path}/tripod_seq_{seq_id:02d}_{frame_id:03d}.jpg"

            img = load_and_resize_image(filename, img_height, img_width)

            # img /= 255.0  # Normalize to [0, 1]

            relative_position = (frame_id - frontal_frame) % num_frames_360
            rotation_angle = relative_position * (360 / num_frames_360) * sense

            data.append(img)
            labels.append(rotation_angle)
            bboxes.append(bbox_data[frame_id - 1])  # Add bbox data

    return np.array(data), np.array(labels), np.array(bboxes)


def load_data(sequence_ids):
    base_path = r'./data/epfl-gims08/tripod-seq'
    file_path = r'./data/epfl-gims08/tripod-seq/tripod-seq.txt'

    image_dims, num_frames, frames_360, frontal_frames, rotation_sense = parse_tripod_seq_file(file_path)
    img_width, img_height = image_dims[1], image_dims[2]

    # Load data
    images, labels, bboxes = load_and_preprocess_data(
        base_path, sequence_ids, img_width, img_height,
        num_frames[:10], frames_360[:10], frontal_frames[:10], rotation_sense[:10])

    return images, labels, bboxes
