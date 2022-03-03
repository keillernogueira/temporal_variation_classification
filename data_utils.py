import os

import imageio
import numpy as np
from skimage import transform


def load_image(image_path):  # os.path.join(dataset_path, image_name)
    try:
        img = imageio.imread(image_path)
    except IOError:
        raise IOError("Could not open file: ", image_path)

    return np.asarray(img)


def create_distributions_over_classes(labels, crop_size, stride_size, num_classes, return_all=False):
    classes = [[[] for i in range(0)] for i in range(num_classes + 1)]

    w, h = labels.shape

    for i in range(0, w, stride_size):
        for j in range(0, h, stride_size):
            cur_x = i
            cur_y = j
            patch_class = labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

            if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                cur_x = cur_x - (crop_size - len(patch_class))
                cur_y = cur_y - (crop_size - len(patch_class[0]))
                patch_class = labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
            elif len(patch_class) != crop_size:
                cur_x = cur_x - (crop_size - len(patch_class))
                patch_class = labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
            elif len(patch_class[0]) != crop_size:
                cur_y = cur_y - (crop_size - len(patch_class[0]))
                patch_class = labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

            assert patch_class.shape == (crop_size, crop_size), "Error create_distributions_over_classes: " \
                                                                "Current patch size is " + str(len(patch_class)) + \
                                                                "x" + str(len(patch_class[0]))

            count = np.bincount(patch_class.astype(int).flatten(), minlength=10)
            # print('count', count, count[-1] == crop_size * crop_size, np.argmax(count[:-1]))
            if count[-1] == crop_size * crop_size:
                classes[-1].append((cur_x, cur_y, count))
            else:
                classes[np.argmax(count[:-1])].append((cur_x, cur_y, count))

    for i in range(len(classes)):
        print('Class ' + str(i + 1) + ' has length ' + str(len(classes[i])))

    if return_all is False:
        return np.asarray(classes[0] + classes[1] + classes[2] + classes[3] + classes[4] +
                          classes[5] + classes[6] + classes[7] + classes[8])
    else:
        return np.asarray(classes[0] + classes[1] + classes[2] + classes[3] + classes[4] +
                          classes[5] + classes[6] + classes[7] + classes[8] + classes[9])


def normalize_images(data, _mean, _std):
    for i in range(len(_mean)):
        data[:, :, i] = np.subtract(data[:, :, i], _mean[i])
        data[:, :, i] = np.divide(data[:, :, i], _std[i])


def compute_image_mean(data):
    _mean = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    _std = np.std(np.std(np.std(data, axis=0, ddof=1), axis=0, ddof=1), axis=0, ddof=1)

    return _mean, _std


def dynamically_calculate_mean_and_std(data, distrib, crop_size):
    all_patches = []

    for i in range(len(distrib)):
        cur_x = distrib[i][0]
        cur_y = distrib[i][1]
        patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

        if len(patch[0]) != crop_size or len(patch[1]) != crop_size:
            raise NotImplementedError("Error! Current patch size: " +
                                      str(len(patch)) + "x" + str(len(patch[0])))

        all_patches.append(patch)

    # remaining images
    return compute_image_mean(np.asarray(all_patches))


def create_or_load_statistics(data, distrib, crop_size, stride_size, output_path):
    # create mean, std from training
    if os.path.isfile(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                      str(stride_size) + '_mean.npy')):
        _mean = np.load(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                                     str(stride_size) + '_mean.npy'), allow_pickle=True)
        _std = np.load(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                                    str(stride_size) + '_std.npy'), allow_pickle=True)
    else:
        _mean, _std = dynamically_calculate_mean_and_std(data, distrib, crop_size)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                str(stride_size) + '_mean.npy'), _mean)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                str(stride_size) + '_std.npy'), _std)
    print(_mean, _std)
    return _mean, _std


def data_augmentation(img, label=None):
    rand_fliplr = np.random.random() > 0.50
    rand_flipud = np.random.random() > 0.50
    rand_rotate = np.random.random()

    if rand_fliplr:
        img = np.fliplr(img)
        if label is not None:
            label = np.fliplr(label)
    if rand_flipud:
        img = np.flipud(img)
        if label is not None:
            label = np.flipud(label)

    if rand_rotate < 0.25:
        img = transform.rotate(img, 270, order=1, preserve_range=True)
        if label is not None:
            label = transform.rotate(label, 270, order=0, preserve_range=True)
    elif rand_rotate < 0.50:
        img = transform.rotate(img, 180, order=1, preserve_range=True)
        if label is not None:
            label = transform.rotate(label, 180, order=0, preserve_range=True)
    elif rand_rotate < 0.75:
        img = transform.rotate(img, 90, order=1, preserve_range=True)
        if label is not None:
            label = transform.rotate(label, 90, order=0, preserve_range=True)

    img = img.astype(np.float32)
    if label is not None:
        label = label.astype(np.int64)

    return img, label
