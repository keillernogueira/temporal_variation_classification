import os
import argparse

import imageio
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 2866205440


def int2color(val):
    if val == 0:
        return (0, 0, 0)
    elif val == 1:
        return (255, 0, 0)
    elif val == 2:
        return (0, 255, 0)
    elif val == 3:
        return (0, 0, 255)
    elif val == 4:
        return (255, 255, 0)
    elif val == 5:
        return (255, 0, 255)
    elif val == 6:
        return (0, 255, 255)
    elif val == 7:
        return (255, 255, 255)
    elif val == 8:
        return (128, 0, 0)
    elif val == 9:
        return (0, 128, 0)


def plot_masks(dataset_path):
    train = imageio.imread(os.path.join(dataset_path, "train.png"))
    test = imageio.imread(os.path.join(dataset_path, "test.png"))

    print(train.shape, np.bincount(train.flatten()), test.shape, np.bincount(test.flatten()))

    h, w = train.shape
    train_m = np.zeros((h, w, 3), dtype=np.uint8)
    test_m = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            train_m[i, j, :] = int2color(train[i, j])
            test_m[i, j, :] = int2color(test[i, j])

    imageio.imwrite(os.path.join(dataset_path, 'train_m.png'), train_m)
    imageio.imwrite(os.path.join(dataset_path, 'test_m.png'), test_m)


def main():
    parser = argparse.ArgumentParser(description='general.py')
    parser.add_argument('--dataset_input_path', type=str, required=True, help='Dataset path.')
    args = parser.parse_args()
    print(args)

    plot_masks(args.dataset_input_path)


if __name__ == "__main__":
    main()
