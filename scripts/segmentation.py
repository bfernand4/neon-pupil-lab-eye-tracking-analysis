import argparse
import collections
import os
import cv2
import glob
import numpy as np
from PIL import Image
import pandas as pd
from scipy import stats
from collections import deque
import tensorflow as tf
import csv
from tqdm import tqdm

# Information about COCO categories
COCO_META = [
    {
        'color': [220, 20, 60],
        'isthing': 1,
        'id': 1,
        'name': 'person'
    },
    {
        'color': [119, 11, 32],
        'isthing': 1,
        'id': 2,
        'name': 'bicycle'
    },
    {
        'color': [0, 0, 142],
        'isthing': 1,
        'id': 3,
        'name': 'car'
    },
    {
        'color': [0, 0, 230],
        'isthing': 1,
        'id': 4,
        'name': 'motorcycle'
    },
    {
        'color': [106, 0, 228],
        'isthing': 1,
        'id': 5,
        'name': 'airplane'
    },
    {
        'color': [0, 60, 100],
        'isthing': 1,
        'id': 6,
        'name': 'bus'
    },
    {
        'color': [0, 80, 100],
        'isthing': 1,
        'id': 7,
        'name': 'train'
    },
    {
        'color': [0, 0, 70],
        'isthing': 1,
        'id': 8,
        'name': 'truck'
    },
    {
        'color': [0, 0, 192],
        'isthing': 1,
        'id': 9,
        'name': 'boat'
    },
    {
        'color': [250, 170, 30],
        'isthing': 1,
        'id': 10,
        'name': 'traffic light'
    },
    {
        'color': [100, 170, 30],
        'isthing': 1,
        'id': 11,
        'name': 'fire hydrant'
    },
    {
        'color': [220, 220, 0],
        'isthing': 1,
        'id': 13,
        'name': 'stop sign'
    },
    {
        'color': [175, 116, 175],
        'isthing': 1,
        'id': 14,
        'name': 'parking meter'
    },
    {
        'color': [250, 0, 30],
        'isthing': 1,
        'id': 15,
        'name': 'bench'
    },
    {
        'color': [165, 42, 42],
        'isthing': 1,
        'id': 16,
        'name': 'bird'
    },
    {
        'color': [255, 77, 255],
        'isthing': 1,
        'id': 17,
        'name': 'cat'
    },
    {
        'color': [0, 226, 252],
        'isthing': 1,
        'id': 18,
        'name': 'dog'
    },
    {
        'color': [182, 182, 255],
        'isthing': 1,
        'id': 19,
        'name': 'horse'
    },
    {
        'color': [0, 82, 0],
        'isthing': 1,
        'id': 20,
        'name': 'sheep'
    },
    {
        'color': [120, 166, 157],
        'isthing': 1,
        'id': 21,
        'name': 'cow'
    },
    {
        'color': [110, 76, 0],
        'isthing': 1,
        'id': 22,
        'name': 'elephant'
    },
    {
        'color': [174, 57, 255],
        'isthing': 1,
        'id': 23,
        'name': 'bear'
    },
    {
        'color': [199, 100, 0],
        'isthing': 1,
        'id': 24,
        'name': 'zebra'
    },
    {
        'color': [72, 0, 118],
        'isthing': 1,
        'id': 25,
        'name': 'giraffe'
    },
    {
        'color': [255, 179, 240],
        'isthing': 1,
        'id': 27,
        'name': 'backpack'
    },
    {
        'color': [0, 125, 92],
        'isthing': 1,
        'id': 28,
        'name': 'umbrella'
    },
    {
        'color': [209, 0, 151],
        'isthing': 1,
        'id': 31,
        'name': 'handbag'
    },
    {
        'color': [188, 208, 182],
        'isthing': 1,
        'id': 32,
        'name': 'tie'
    },
    {
        'color': [0, 220, 176],
        'isthing': 1,
        'id': 33,
        'name': 'suitcase'
    },
    {
        'color': [255, 99, 164],
        'isthing': 1,
        'id': 34,
        'name': 'frisbee'
    },
    {
        'color': [92, 0, 73],
        'isthing': 1,
        'id': 35,
        'name': 'skis'
    },
    {
        'color': [133, 129, 255],
        'isthing': 1,
        'id': 36,
        'name': 'snowboard'
    },
    {
        'color': [78, 180, 255],
        'isthing': 1,
        'id': 37,
        'name': 'sports ball'
    },
    {
        'color': [0, 228, 0],
        'isthing': 1,
        'id': 38,
        'name': 'kite'
    },
    {
        'color': [174, 255, 243],
        'isthing': 1,
        'id': 39,
        'name': 'baseball bat'
    },
    {
        'color': [45, 89, 255],
        'isthing': 1,
        'id': 40,
        'name': 'baseball glove'
    },
    {
        'color': [134, 134, 103],
        'isthing': 1,
        'id': 41,
        'name': 'skateboard'
    },
    {
        'color': [145, 148, 174],
        'isthing': 1,
        'id': 42,
        'name': 'surfboard'
    },
    {
        'color': [255, 208, 186],
        'isthing': 1,
        'id': 43,
        'name': 'tennis racket'
    },
    {
        'color': [197, 226, 255],
        'isthing': 1,
        'id': 44,
        'name': 'bottle'
    },
    {
        'color': [171, 134, 1],
        'isthing': 1,
        'id': 46,
        'name': 'wine glass'
    },
    {
        'color': [109, 63, 54],
        'isthing': 1,
        'id': 47,
        'name': 'cup'
    },
    {
        'color': [207, 138, 255],
        'isthing': 1,
        'id': 48,
        'name': 'fork'
    },
    {
        'color': [151, 0, 95],
        'isthing': 1,
        'id': 49,
        'name': 'knife'
    },
    {
        'color': [9, 80, 61],
        'isthing': 1,
        'id': 50,
        'name': 'spoon'
    },
    {
        'color': [84, 105, 51],
        'isthing': 1,
        'id': 51,
        'name': 'bowl'
    },
    {
        'color': [74, 65, 105],
        'isthing': 1,
        'id': 52,
        'name': 'banana'
    },
    {
        'color': [166, 196, 102],
        'isthing': 1,
        'id': 53,
        'name': 'apple'
    },
    {
        'color': [208, 195, 210],
        'isthing': 1,
        'id': 54,
        'name': 'sandwich'
    },
    {
        'color': [255, 109, 65],
        'isthing': 1,
        'id': 55,
        'name': 'orange'
    },
    {
        'color': [0, 143, 149],
        'isthing': 1,
        'id': 56,
        'name': 'broccoli'
    },
    {
        'color': [179, 0, 194],
        'isthing': 1,
        'id': 57,
        'name': 'carrot'
    },
    {
        'color': [209, 99, 106],
        'isthing': 1,
        'id': 58,
        'name': 'hot dog'
    },
    {
        'color': [5, 121, 0],
        'isthing': 1,
        'id': 59,
        'name': 'pizza'
    },
    {
        'color': [227, 255, 205],
        'isthing': 1,
        'id': 60,
        'name': 'donut'
    },
    {
        'color': [147, 186, 208],
        'isthing': 1,
        'id': 61,
        'name': 'cake'
    },
    {
        'color': [153, 69, 1],
        'isthing': 1,
        'id': 62,
        'name': 'chair'
    },
    {
        'color': [3, 95, 161],
        'isthing': 1,
        'id': 63,
        'name': 'couch'
    },
    {
        'color': [163, 255, 0],
        'isthing': 1,
        'id': 64,
        'name': 'potted plant'
    },
    {
        'color': [119, 0, 170],
        'isthing': 1,
        'id': 65,
        'name': 'bed'
    },
    {
        'color': [0, 182, 199],
        'isthing': 1,
        'id': 67,
        'name': 'dining table'
    },
    {
        'color': [0, 165, 120],
        'isthing': 1,
        'id': 70,
        'name': 'toilet'
    },
    {
        'color': [183, 130, 88],
        'isthing': 1,
        'id': 72,
        'name': 'tv'
    },
    {
        'color': [95, 32, 0],
        'isthing': 1,
        'id': 73,
        'name': 'laptop'
    },
    {
        'color': [130, 114, 135],
        'isthing': 1,
        'id': 74,
        'name': 'mouse'
    },
    {
        'color': [110, 129, 133],
        'isthing': 1,
        'id': 75,
        'name': 'remote'
    },
    {
        'color': [166, 74, 118],
        'isthing': 1,
        'id': 76,
        'name': 'keyboard'
    },
    {
        'color': [219, 142, 185],
        'isthing': 1,
        'id': 77,
        'name': 'cell phone'
    },
    {
        'color': [79, 210, 114],
        'isthing': 1,
        'id': 78,
        'name': 'microwave'
    },
    {
        'color': [178, 90, 62],
        'isthing': 1,
        'id': 79,
        'name': 'oven'
    },
    {
        'color': [65, 70, 15],
        'isthing': 1,
        'id': 80,
        'name': 'toaster'
    },
    {
        'color': [127, 167, 115],
        'isthing': 1,
        'id': 81,
        'name': 'sink'
    },
    {
        'color': [59, 105, 106],
        'isthing': 1,
        'id': 82,
        'name': 'refrigerator'
    },
    {
        'color': [142, 108, 45],
        'isthing': 1,
        'id': 84,
        'name': 'book'
    },
    {
        'color': [196, 172, 0],
        'isthing': 1,
        'id': 85,
        'name': 'clock'
    },
    {
        'color': [95, 54, 80],
        'isthing': 1,
        'id': 86,
        'name': 'vase'
    },
    {
        'color': [128, 76, 255],
        'isthing': 1,
        'id': 87,
        'name': 'scissors'
    },
    {
        'color': [201, 57, 1],
        'isthing': 1,
        'id': 88,
        'name': 'teddy bear'
    },
    {
        'color': [246, 0, 122],
        'isthing': 1,
        'id': 89,
        'name': 'hair drier'
    },
    {
        'color': [191, 162, 208],
        'isthing': 1,
        'id': 90,
        'name': 'toothbrush'
    },
    {
        'color': [255, 255, 128],
        'isthing': 0,
        'id': 92,
        'name': 'banner'
    },
    {
        'color': [147, 211, 203],
        'isthing': 0,
        'id': 93,
        'name': 'blanket'
    },
    {
        'color': [150, 100, 100],
        'isthing': 0,
        'id': 95,
        'name': 'bridge'
    },
    {
        'color': [168, 171, 172],
        'isthing': 0,
        'id': 100,
        'name': 'cardboard'
    },
    {
        'color': [146, 112, 198],
        'isthing': 0,
        'id': 107,
        'name': 'counter'
    },
    {
        'color': [210, 170, 100],
        'isthing': 0,
        'id': 109,
        'name': 'curtain'
    },
    {
        'color': [92, 136, 89],
        'isthing': 0,
        'id': 112,
        'name': 'door-stuff'
    },
    {
        'color': [218, 88, 184],
        'isthing': 0,
        'id': 118,
        'name': 'floor-wood'
    },
    {
        'color': [241, 129, 0],
        'isthing': 0,
        'id': 119,
        'name': 'flower'
    },
    {
        'color': [217, 17, 255],
        'isthing': 0,
        'id': 122,
        'name': 'fruit'
    },
    {
        'color': [124, 74, 181],
        'isthing': 0,
        'id': 125,
        'name': 'gravel'
    },
    {
        'color': [70, 70, 70],
        'isthing': 0,
        'id': 128,
        'name': 'house'
    },
    {
        'color': [255, 228, 255],
        'isthing': 0,
        'id': 130,
        'name': 'light'
    },
    {
        'color': [154, 208, 0],
        'isthing': 0,
        'id': 133,
        'name': 'mirror-stuff'
    },
    {
        'color': [193, 0, 92],
        'isthing': 0,
        'id': 138,
        'name': 'net'
    },
    {
        'color': [76, 91, 113],
        'isthing': 0,
        'id': 141,
        'name': 'pillow'
    },
    {
        'color': [255, 180, 195],
        'isthing': 0,
        'id': 144,
        'name': 'platform'
    },
    {
        'color': [106, 154, 176],
        'isthing': 0,
        'id': 145,
        'name': 'playingfield'
    },
    {
        'color': [230, 150, 140],
        'isthing': 0,
        'id': 147,
        'name': 'railroad'
    },
    {
        'color': [60, 143, 255],
        'isthing': 0,
        'id': 148,
        'name': 'river'
    },
    {
        'color': [128, 64, 128],
        'isthing': 0,
        'id': 149,
        'name': 'road'
    },
    {
        'color': [92, 82, 55],
        'isthing': 0,
        'id': 151,
        'name': 'roof'
    },
    {
        'color': [254, 212, 124],
        'isthing': 0,
        'id': 154,
        'name': 'sand'
    },
    {
        'color': [73, 77, 174],
        'isthing': 0,
        'id': 155,
        'name': 'sea'
    },
    {
        'color': [255, 160, 98],
        'isthing': 0,
        'id': 156,
        'name': 'shelf'
    },
    {
        'color': [255, 255, 255],
        'isthing': 0,
        'id': 159,
        'name': 'snow'
    },
    {
        'color': [104, 84, 109],
        'isthing': 0,
        'id': 161,
        'name': 'stairs'
    },
    {
        'color': [169, 164, 131],
        'isthing': 0,
        'id': 166,
        'name': 'tent'
    },
    {
        'color': [225, 199, 255],
        'isthing': 0,
        'id': 168,
        'name': 'towel'
    },
    {
        'color': [137, 54, 74],
        'isthing': 0,
        'id': 171,
        'name': 'wall-brick'
    },
    {
        'color': [135, 158, 223],
        'isthing': 0,
        'id': 175,
        'name': 'wall-stone'
    },
    {
        'color': [7, 246, 231],
        'isthing': 0,
        'id': 176,
        'name': 'wall-tile'
    },
    {
        'color': [107, 255, 200],
        'isthing': 0,
        'id': 177,
        'name': 'wall-wood'
    },
    {
        'color': [58, 41, 149],
        'isthing': 0,
        'id': 178,
        'name': 'water-other'
    },
    {
        'color': [183, 121, 142],
        'isthing': 0,
        'id': 180,
        'name': 'window-blind'
    },
    {
        'color': [255, 73, 97],
        'isthing': 0,
        'id': 181,
        'name': 'window-other'
    },
    {
        'color': [107, 142, 35],
        'isthing': 0,
        'id': 184,
        'name': 'tree-merged'
    },
    {
        'color': [190, 153, 153],
        'isthing': 0,
        'id': 185,
        'name': 'fence-merged'
    },
    {
        'color': [146, 139, 141],
        'isthing': 0,
        'id': 186,
        'name': 'ceiling-merged'
    },
    {
        'color': [70, 130, 180],
        'isthing': 0,
        'id': 187,
        'name': 'sky-other-merged'
    },
    {
        'color': [134, 199, 156],
        'isthing': 0,
        'id': 188,
        'name': 'cabinet-merged'
    },
    {
        'color': [209, 226, 140],
        'isthing': 0,
        'id': 189,
        'name': 'table-merged'
    },
    {
        'color': [96, 36, 108],
        'isthing': 0,
        'id': 190,
        'name': 'floor-other-merged'
    },
    {
        'color': [96, 96, 96],
        'isthing': 0,
        'id': 191,
        'name': 'pavement-merged'
    },
    {
        'color': [64, 170, 64],
        'isthing': 0,
        'id': 192,
        'name': 'mountain-merged'
    },
    {
        'color': [152, 251, 152],
        'isthing': 0,
        'id': 193,
        'name': 'grass-merged'
    },
    {
        'color': [208, 229, 228],
        'isthing': 0,
        'id': 194,
        'name': 'dirt-merged'
    },
    {
        'color': [206, 186, 171],
        'isthing': 0,
        'id': 195,
        'name': 'paper-merged'
    },
    {
        'color': [152, 161, 64],
        'isthing': 0,
        'id': 196,
        'name': 'food-other-merged'
    },
    {
        'color': [116, 112, 0],
        'isthing': 0,
        'id': 197,
        'name': 'building-other-merged'
    },
    {
        'color': [0, 114, 143],
        'isthing': 0,
        'id': 198,
        'name': 'rock-merged'
    },
    {
        'color': [102, 102, 156],
        'isthing': 0,
        'id': 199,
        'name': 'wall-other-merged'
    },
    {
        'color': [250, 141, 255],
        'isthing': 0,
        'id': 200,
        'name': 'rug-merged'
    },
]

# Update IDs so they are consecutive
for i in range(len(COCO_META)):
  COCO_META[i]['id'] = i + 1
  
DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    'num_classes, label_divisor, thing_list, colormap, class_names')

# Functions for COCO dataset information
def _coco_label_colormap():
  """Creates a label colormap used in COCO segmentation benchmark.

  See more about COCO dataset at https://cocodataset.org/
  Tsung-Yi Lin, et al. "Microsoft COCO: Common Objects in Context." ECCV. 2014.

  Returns:
    A 2-D numpy array with each row being mapped RGB color (in uint8 range).
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  for category in COCO_META:
    colormap[category['id']] = category['color']
  return colormap

def _coco_class_names():
  return ('void',) + tuple([x['name'] for x in COCO_META])

def coco_dataset_information():
  return DatasetInfo(
      num_classes=134,
      label_divisor=256,
      thing_list=tuple(range(1, 81)),
      colormap=_coco_label_colormap(),
      class_names=_coco_class_names())

def perturb_color(color, noise, used_colors, max_trials=50, random_state=None):
  """Perturbs the color with some noise.

  If `used_colors` is not None, we will return the color that has
  not appeared before in it.

  Args:
    color: A numpy array with three elements [R, G, B].
    noise: Integer, specifying the amount of perturbing noise (in uint8 range).
    used_colors: A set, used to keep track of used colors.
    max_trials: An integer, maximum trials to generate random color.
    random_state: An optional np.random.RandomState. If passed, will be used to
      generate random numbers.

  Returns:
    A perturbed color that has not appeared in used_colors.
  """
  if random_state is None:
    random_state = np.random

  for _ in range(max_trials):
    random_color = color + random_state.randint(
        low=-noise, high=noise + 1, size=3)
    random_color = np.clip(random_color, 0, 255)

    if tuple(random_color) not in used_colors:
      used_colors.add(tuple(random_color))
      return random_color

  print('Max trial reached and duplicate color will be used. Please consider '
        'increase noise in `perturb_color()`.')
  return random_color

def color_panoptic_map(panoptic_prediction, dataset_info, perturb_noise):
  """Helper method to colorize output panoptic map.

  Args:
    panoptic_prediction: A 2D numpy array, panoptic prediction from deeplab
      model.
    dataset_info: A DatasetInfo object, dataset associated to the model.
    perturb_noise: Integer, the amount of noise (in uint8 range) added to each
      instance of the same semantic class.

  Returns:
    colored_panoptic_map: A 3D numpy array with last dimension of 3, colored
      panoptic prediction map.
    used_colors: A dictionary mapping semantic_ids to a set of colors used
      in `colored_panoptic_map`.
  """
  if panoptic_prediction.ndim != 2:
    raise ValueError('Expect 2-D panoptic prediction. Got {}'.format(
        panoptic_prediction.shape))

  semantic_map = panoptic_prediction // dataset_info.label_divisor
  instance_map = panoptic_prediction % dataset_info.label_divisor
  height, width = panoptic_prediction.shape
  colored_panoptic_map = np.zeros((height, width, 3), dtype=np.uint8)

  used_colors = collections.defaultdict(set)
  # Use a fixed seed to reproduce the same visualization.
  random_state = np.random.RandomState(0)

  unique_semantic_ids = np.unique(semantic_map)
  for semantic_id in unique_semantic_ids:
    semantic_mask = semantic_map == semantic_id
    if semantic_id in dataset_info.thing_list:
      # For `thing` class, we will add a small amount of random noise to its
      # correspondingly predefined semantic segmentation colormap.
      unique_instance_ids = np.unique(instance_map[semantic_mask])
      for instance_id in unique_instance_ids:
        instance_mask = np.logical_and(semantic_mask,
                                       instance_map == instance_id)
        random_color = perturb_color(
            dataset_info.colormap[semantic_id],
            perturb_noise,
            used_colors[semantic_id],
            random_state=random_state)
        colored_panoptic_map[instance_mask] = random_color
    else:
      # For `stuff` class, we use the defined semantic color.
      colored_panoptic_map[semantic_mask] = dataset_info.colormap[semantic_id]
      used_colors[semantic_id].add(tuple(dataset_info.colormap[semantic_id]))
  return colored_panoptic_map, used_colors

# Functions for handling fixations and images
def get_fixations_on_frames(fixations, image_folder, fps=30):
    """
    Function that returns a list of dictionaries containing images and their corresponding fixation coordinates.

    :param fixations: List of fixations in the form [{"start": , "end": , "x": , "y": }]
    :param image_folder: Folder containing the images extracted from the video
    :param fps: Frames per second of the video (default 30)
    :return: List of dictionaries [{"image": "imgXXXX.jpg", "coordinates": [x, y]}]
    """
    # Filter only image files (jpg, jpeg, png)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # Sort images to ensure order
    total_frames = len(image_files)
    fixation_frames = []

    # Check if there are fixations
    if not fixations:
        return fixation_frames

    # Find the first fixation timestamp to align times
    first_timestamp = fixations[0]['start']  # First fixation timestamp in nanoseconds
    print(f"First fixation timestamp: {first_timestamp}")

    # Loop over each image and try to find the corresponding fixations
    for frame_index, img_name in enumerate(image_files):
        frame_time_sec = frame_index / fps  # Compute time in seconds for this frame
        print(f"Processing image {img_name} - frame time: {frame_time_sec:.4f} sec")

        matching_fixation = None

        # Search for the fixation that matches this time
        for i, fixation in enumerate(fixations):
            # Convert the start timestamp of the fixation to seconds
            fixation_time_sec = (fixation['start'] - first_timestamp) / 1e9  # Convert nanoseconds to seconds
            fixation_end_sec = fixation_time_sec + fixation['duration'] / 1000  # End time in seconds

            # Find the next fixation (if it exists)
            if i + 1 < len(fixations):
                next_fixation_time_sec = (fixations[i + 1]['start'] - first_timestamp) / 1e9
            else:
                next_fixation_time_sec = float('inf')  # No next fixation, so use infinity

            print(f"  Fixation start: {fixation_time_sec:.4f} sec, end: {fixation_end_sec:.4f} sec, "
                f"frame_time_sec: {frame_time_sec:.4f} sec, next_fixation_start: {next_fixation_time_sec:.4f} sec")

            # If the image time is between the start of the current fixation and the start of the next
            if fixation_time_sec <= frame_time_sec < next_fixation_time_sec:
                matching_fixation = fixation
                print(f"  Found a matching fixation for {img_name}")
                break  # If a fixation is found, exit the loop

        # If a fixation is found, add it to the list of frames
        if matching_fixation:
            fixation_frames.append({
                "image": img_name,
                "coordinates": [matching_fixation['x'], matching_fixation['y']]
            })
        else:
            print(f"  No fixation found for {img_name}")

    print(f"Total number of fixations found: {len(fixation_frames)}")
    return fixation_frames
    
def get_image_timestamp(image_filename, fps=30):
    """
    Computes the timestamp of the image based on its file number.

    Args:
        image_filename (str): The image file name (e.g. 'img0001.jpg')
        fps (int): Number of frames per second (default 30)

    Returns:
        float: The timestamp of the image in seconds
    """
    # Extract the image number from the file name
    image_id = int(image_filename.split('img')[1].split('.jpg')[0])
    # Compute the timestamp (in seconds)
    timestamp = (image_id - 1)/ fps  # fps images per second
    return timestamp
    
def match_fixations_to_images(fixations, image_folder, start_time, fps=30, time_window=1/30):
    """
    Associates fixations to images based on their timestamp.

    Args:
        fixations (list): List of fixations with 'start', 'end', 'x', 'y'
        image_folder (str): Folder containing the images
        fps (int): Number of frames per second
        time_window (float): Time window around the image timestamp (in seconds)

    Returns:
        dict: Dictionary with images as keys and associated fixations as values
    """
    # List of images in the folder
    image_files = sorted(os.listdir(image_folder))

    # Dictionary to store fixations associated with each image
    fixations_per_image = {image_filename: [] for image_filename in image_files}

    first_timestamp = start_time / 1e9

    # Associate each fixation to the corresponding image
    for fixation in fixations:
        fixation_start = fixation["start"] / 1e9 - first_timestamp # Convert to seconds
        fixation_end = fixation["end"] / 1e9 - first_timestamp

        # Check each image to see if it matches the fixation
        for image_filename in image_files:
            # Compute the timestamp of the image
            image_timestamp = get_image_timestamp(image_filename, fps=fps)

            # Check if the fixation is in the time window around the image
            if fixation_start <= image_timestamp <= fixation_end or fixation_start <= image_timestamp + time_window <= fixation_end:
                fixations_per_image[image_filename].append(fixation)

    return fixations_per_image
    
def get_label_for_fixation(image_name, fixations, panoptic_pred, dataset_info):
    """
    Returns a label for each fixation in the given image, from the panoptic prediction.

    Arguments:
    - image_name: The image name (e.g. 'img0080.jpg')
    - fixations: A dictionary with the image name as key and a list of fixations as value
    - panoptic_pred: The panoptic prediction for the image (HxW matrix)
    - dataset_info: The object containing dataset information (class names)

    Returns:
    - A list of labels for each fixation in the image.
    """
    # Get the fixations associated with the given image
    image_fixations = fixations.get(image_name, [])
    labels = []

    for fixation in image_fixations:
        # Get the x and y coordinates of the fixation
        x = int(round(fixation['x']))  # Round coordinates
        y = int(round(fixation['y']))

        # Check that the coordinates are within the image
        if 0 <= y < panoptic_pred.shape[0] and 0 <= x < panoptic_pred.shape[1]:
            # Get the panoptic ID at this position
            segment_id = panoptic_pred[y, x]

            # Split the ID into semantic and instance
            semantic_id = segment_id // dataset_info.label_divisor
            instance_id = segment_id % dataset_info.label_divisor

            # Check that the semantic ID is valid
            if semantic_id < len(dataset_info.class_names):
                label = dataset_info.class_names[semantic_id]
            else:
                label = "unknown"  # If the ID is outside the defined classes
        else:
            label = "out_of_bounds"  # If the fixation is outside the image

        # Add the label to the list
        labels.append(label)

    return labels
    
def get_frame_timestamp(video_start_timestamp, frame_index):
    return video_start_timestamp + (frame_index / 30)

# Function to save segmented images with legend
def save_segmented_frame_with_legend(image, panoptic_prediction, output_path, dataset_info):
    """Save the image with panoptic overlay and a visible legend."""

    # Generate the panoptic map
    panoptic_map, used_colors = color_panoptic_map(panoptic_prediction, dataset_info, perturb_noise=60)

    # Merge with the original image (overlay)
    overlay = cv2.addWeighted(image, 0.5, panoptic_map, 0.5, 0)

    # Generate a proper legend
    legend_rows = len(used_colors)
    legend_height = legend_rows * 30  # Increase height for readability
    legend_width = 300  # Fixed width for consistency

    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # White background

    # Add colors and class names
    for i, (semantic_id, colors) in enumerate(sorted(used_colors.items())):
        color = np.array(list(colors)[0], dtype=np.uint8)  # Take a representative color
        cv2.rectangle(legend, (10, i * 30 + 5), (40, (i + 1) * 30 - 5), color.tolist(), -1)
        label = dataset_info.class_names[semantic_id] if semantic_id < dataset_info.num_classes else "ignore"
        cv2.putText(legend, label, (50, (i + 1) * 30 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Resize if necessary
    lh, lw, _ = legend.shape
    h, w, _ = overlay.shape
    if lw > w:
        legend = cv2.resize(legend, (w // 3, legend_height))

    # Set the legend position at the bottom right
    x_offset = w - legend.shape[1] - 10
    y_offset = h - legend.shape[0] - 10
    overlay[y_offset:y_offset+legend.shape[0], x_offset:x_offset+legend.shape[1]] = legend

    # Save the final image
    cv2.imwrite(output_path, overlay)
  
def save_images_with_fixations(fixations, image_folder, output_folder):
    # Create the output_folder if it does not already exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for fixation in fixations:
        # Get the image name from the fixation dictionary
        image_filename = fixation['image']
        image_path = os.path.join(image_folder, image_filename)

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue  # Skip to next image if this one does not exist

        # Read the image
        image = cv2.imread(image_path)

        # Get the x and y coordinates
        x, y = fixation['coordinates']

        # Draw a circle (dot) at position x, y
        cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)  # Blue dot

        # Save the modified image in the output folder with the same name
        output_image_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_image_path, image)
        print(f"Image saved: {output_image_path}")

def add_fixations_to_images(fixations_per_image, input_folder, output_folder):
    """
    Function to add fixations to images and save them in a new folder.

    Args:
        fixations_per_image (dict): Dictionary where the key is the image name and the value is a list of fixations.
        input_folder (str): Folder containing the input images.
        output_folder (str): Folder where the modified images will be saved.
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through images and add fixations
    for image_name in tqdm(os.listdir(input_folder),
                       desc=f"Adding fixations in {output_folder}",
                       unit="img"):
        # If the file is not an image, ignore it
        if not image_name.endswith('.jpg'):
            continue

        # Load the image
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path)

        # Check if the image has associated fixations
        if image_name in fixations_per_image:
            for fixation in fixations_per_image[image_name]:
                # Extract the (x, y) coordinates
                x = int(fixation['x'])
                y = int(fixation['y'])

                # Draw a circle at the fixation location
                cv2.circle(img, (x, y), 15, (0, 0, 255), 3)  # Red circle

        # Save the modified image in the new folder
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO video segmentation")
    parser.add_argument("--result_path", type=str, required=True, help="Save path for the results")
    
    # By default, is_saved = False
    parser.add_argument(
        "--is_saved",
        action="store_true",  # if the option is present, it's True
        help="Save the segmented frames with fixations (False by default)"
    )

    args = parser.parse_args()
    is_saved = args.is_saved
    
    # Paths to folders/files produced by preprocess_neon.py
    TEMP_FOLDER = "../temp"
    FRAMES_FOLDER = os.path.join(TEMP_FOLDER, "frames_undistorted")
    FIXATIONS_FILE = os.path.join(TEMP_FOLDER, "fixations_undistorted.csv")
    START_TIME_FILE = os.path.join(TEMP_FOLDER, "start_time.txt")
    NAME_FILE = os.path.join(TEMP_FOLDER, "name_of_file.txt")
    FIX_OUTPUT_FOLDER = args.result_path # Folder to save fixations associated with labels
    SEG_OUTPUT_FOLDER = os.path.join(TEMP_FOLDER, "segmented_frames")  # Folder to save segmented frames
    SEG_FIX_OUTPUT_FOLDER = os.path.join(TEMP_FOLDER, "seg_fix_frames")

    # Load undistorted frames
    image_files = sorted(glob.glob(os.path.join(FRAMES_FOLDER, "*.jpg")))
    if not image_files:
        raise AssertionError('No image found in frames_undistorted/ folder!')

    # Load undistorted fixations
    fixations = []
    with open(FIXATIONS_FILE, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fixations.append({
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["duration"]),
                "x": float(row["x"]),
                "y": float(row["y"])
            })
            
    # Load start_time
    with open(START_TIME_FILE, "r") as f:
        start_time = float(f.read())
        
    # Load the original video file name
    with open(NAME_FILE, "r") as f:
        name_of_file = f.read().strip()

    # Associate fixations to images
    fixations_per_image = match_fixations_to_images(fixations, FRAMES_FOLDER, start_time)

    # Dictionary to store labels per image
    image_labels_dict = {}

    # Load the model
    LOCAL_MODEL_PATH = "../pretrained_model/convnext_large_kmax_deeplab_coco_train"
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Model folder not found at '{LOCAL_MODEL_PATH}'. "
            "Please download and extract the pretrained model before running this script."
        )
    
    LOADED_MODEL = tf.saved_model.load(LOCAL_MODEL_PATH)

    # Check and create output folders if necessary
    os.makedirs(FIX_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    
    # Create folders for segmented images if necessary
    if is_saved:
        os.makedirs(SEG_OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(SEG_FIX_OUTPUT_FOLDER, exist_ok=True)

    # Get COCO dataset information
    DATASET_INFO = coco_dataset_information()

    for img_path in tqdm(image_files, desc=f"Segmenting frames", unit="img"):
        img_name = os.path.basename(img_path)

        # Load the image
        with tf.io.gfile.GFile(img_path, 'rb') as f:
            image = np.array(Image.open(f))

        panoptic_pred_full = LOADED_MODEL(tf.cast(image, tf.uint8))['panoptic_pred'][0].numpy()
        if is_saved:
            output_path = os.path.join(SEG_OUTPUT_FOLDER, img_name)
            save_segmented_frame_with_legend(image, panoptic_pred_full, output_path, DATASET_INFO)
        
        image_labels_dict[img_name] = get_label_for_fixation(img_name, fixations_per_image, panoptic_pred_full, DATASET_INFO)
        
    # Build a mapping name -> color
    name_to_color = {entry['name']: [c/255.0 for c in entry['color']] for entry in COCO_META}
    csv_filename =os.path.join(FIX_OUTPUT_FOLDER, f"{name_of_file}.csv")

    # Extract all unique classes present
    all_labels = sorted(set(lab for labs in image_labels_dict.values() for lab in labs))
    label_to_y = {label: i for i, label in enumerate(all_labels)}

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "timestamp", "image_name", "label", "y_position", "r", "g", "b"])  # Header

        for i, (img_name, labels) in enumerate(image_labels_dict.items()):
            # Compute the timestamp for this frame
            timestamp_frame = get_frame_timestamp(start_time, i)
            for label in labels:
                y = label_to_y[label]
                color = name_to_color.get(label, [0.5, 0.5, 0.5])  # fallback gray if unknown
                writer.writerow([i, timestamp_frame, img_name, label, y, color[0], color[1], color[2]])

    print(f"CSV exported: {csv_filename}")

    # Save images with fixations
    if is_saved:
        # Save segmented images with fixations
        add_fixations_to_images(fixations_per_image, SEG_OUTPUT_FOLDER, SEG_FIX_OUTPUT_FOLDER)