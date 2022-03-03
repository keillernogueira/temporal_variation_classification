import os
import random
import numpy as np
import imageio
import argparse
import matplotlib.pyplot as plt
import torch


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, output_path, best_records, epoch, metric, num_saves=1,
                     patch_acc_loss=None, patch_occur=None, patch_chosen_values=None):
    if len(best_records) < num_saves:
        best_records.append({'epoch': epoch, 'kappa': metric})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        if patch_acc_loss is not None and patch_occur is not None and patch_chosen_values is not None:
            np.save(output_path + 'patch_acc_loss_step_' + str(epoch) + '.npy', patch_acc_loss)
            np.save(output_path + 'patch_occur_step_' + str(epoch) + '.npy', patch_occur)
            np.save(output_path + 'patch_chosen_values_step_' + str(epoch) + '.npy', patch_chosen_values)
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['kappa'] > best_records[i]['kappa']:
                min_index = i

        # check if currect acc is greater than min saved acc
        if metric > best_records[min_index]['kappa']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'kappa': metric}
            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))

            if patch_acc_loss is not None and patch_occur is not None and patch_chosen_values is not None:
                os.remove(os.path.join(output_path, 'patch_acc_loss_step_' + min_step + '.npy'))
                os.remove(os.path.join(output_path, 'patch_occur_step_' + min_step + '.npy'))
                os.remove(os.path.join(output_path, 'patch_chosen_values_step_' + min_step + '.npy'))

                np.save(output_path + 'patch_acc_loss_step_' + str(epoch) + '.npy', patch_acc_loss)
                np.save(output_path + 'patch_occur_step_' + str(epoch) + '.npy', patch_occur)
                np.save(output_path + 'patch_chosen_values_step_' + str(epoch) + '.npy', patch_chosen_values)

    np.save(os.path.join(output_path, 'best_records.npy'), best_records)


def define_multinomial_probs(values, dif_prob=2):
    interval_size = values[-1] - values[0] + 1

    general_prob = 1.0 / float(interval_size)
    max_prob = general_prob * dif_prob  # for values

    probs = np.full(interval_size, (1.0 - max_prob * len(values)) / float(interval_size - len(values)))
    for i in range(len(values)):
        probs[values[i] - values[0]] = max_prob

    return probs


def select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, is_loss_or_acc='acc',
                           patch_chosen_values=None, debug=False):
        patch_occur[np.where(patch_occur == 0)] = 1
        patch_mean = patch_acc_loss / patch_occur
        # print is_loss_or_acc

        if is_loss_or_acc == 'acc':
            argmax_acc = np.argmax(patch_mean)
            if distribution_type == 'multi_fixed':
                cur_patch_val = int(values[argmax_acc])
            elif distribution_type == 'uniform' or distribution_type == 'multinomial':
                cur_patch_val = values[0] + argmax_acc

            if patch_chosen_values is not None:
                patch_chosen_values[int(argmax_acc)] += 1

            if debug is True:
                print('patch_acc_loss', patch_acc_loss)
                print('patch_occur', patch_occur)
                print('patch_mean', patch_mean)
                print('argmax_acc', argmax_acc)

                print('specific', argmax_acc, patch_acc_loss[argmax_acc], patch_occur[argmax_acc], patch_mean[argmax_acc])

        elif is_loss_or_acc == 'loss':
            arg_sort_out = np.argsort(patch_mean)

            if debug is True:
                print('patch_acc_loss', patch_acc_loss)
                print('patch_occur', patch_occur)
                print('patch_mean', patch_mean)
                print('arg_sort_out', arg_sort_out)
            if distribution_type == 'multi_fixed':
                for i in range(len(values)):
                    if patch_occur[arg_sort_out[i]] > 0:
                        cur_patch_val = int(values[arg_sort_out[i]])  # -1*(i+1)
                        if patch_chosen_values is not None:
                            patch_chosen_values[arg_sort_out[i]] += 1
                        if debug is True:
                            print('specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], patch_occur[
                                arg_sort_out[i]], patch_mean[arg_sort_out[i]])
                        break
            elif distribution_type == 'uniform' or distribution_type == 'multinomial':
                for i in range(values[-1] - values[0] + 1):
                    if patch_occur[arg_sort_out[i]] > 0:
                        cur_patch_val = values[0] + arg_sort_out[i]
                        if patch_chosen_values is not None:
                            patch_chosen_values[arg_sort_out[i]] += 1
                        if debug is True:
                            print('specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], patch_occur[
                                arg_sort_out[i]], patch_mean[arg_sort_out[i]])
                        break

        if debug is True:
            print('Current patch size ', cur_patch_val)
            if patch_chosen_values is not None:
                print('Distr of chosen sizes ', patch_chosen_values)

        return cur_patch_val


def create_prediction_map(img_name, prob_img, channels=False):
    if channels is True:
        for i in range(prob_img.shape[-1]):
            # imageio.imwrite(img_name + 'feat_' + str(i) + '.png', prob_img[:, :, i].astype(np.uint8))
            plt.imsave(img_name + 'feat_' + str(i) + '.png', prob_img[:, :, i], cmap=plt.cm.jet)
    else:
        imageio.imwrite(img_name + '.png', prob_img.astype(np.uint8) * 255)
        # img = Image.fromarray(prob_img.astype(np.uint8) * 255)
        # img.save(img_name + ".tif")


def calc_accuracy_by_crop(true_crop, pred_crop, num_classes, track_conf_matrix, masks=None):
    b, h, w = pred_crop.shape

    acc = 0
    local_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
    # count = 0
    for i in range(b):
        for j in range(h):
            for k in range(w):
                if masks is None or (masks is not None and masks[i, j, k]):
                    # count += 1
                    if true_crop[i, j, k] == pred_crop[i, j, k]:
                        acc = acc + 1
                    if track_conf_matrix is not None:
                        track_conf_matrix[true_crop[i, j, k]][pred_crop[i, j, k]] += 1
                    local_conf_matrix[true_crop[i, j, k]][pred_crop[i, j, k]] += 1

    # print count, b*h*w
    return acc, local_conf_matrix


def calc_accuracy_by_class(true_crop, pred_crop, num_classes, track_conf_matrix):
    acc = 0
    local_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
    # count = 0
    for i in range(len(true_crop)):
        if true_crop[i] == pred_crop[i]:
            acc = acc + 1
        track_conf_matrix[true_crop[i]][pred_crop[i]] += 1
        local_conf_matrix[true_crop[i]][pred_crop[i]] += 1

    return acc, local_conf_matrix


def create_cm(true, pred):
    conf_matrix = np.zeros((len(np.unique(true)), len(np.unique(true))), dtype=np.uint32)
    c, h, w = true.shape
    for i in range(c):
        for j in range(h):
            for k in range(w):
                conf_matrix[true[i, j, k]][pred[i, j, k]] += 1

    return conf_matrix


def kappa_with_cm(conf_matrix):
    acc = 0
    marginal = 0
    total = float(np.sum(conf_matrix))
    for i in range(len(conf_matrix)):
        acc += conf_matrix[i][i]
        marginal += np.sum(conf_matrix, 0)[i] * np.sum(conf_matrix, 1)[i]

    kappa = (total * acc - marginal) / (total * total - marginal)
    return kappa


def f1_with_cm(conf_matrix):
    precision = [0] * len(conf_matrix)
    recall = [0] * len(conf_matrix)
    f1 = [0] * len(conf_matrix)
    for i in range(len(conf_matrix)):
        precision[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 0)[i])
        recall[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 1)[i])
        f1[i] = 2 * ((precision[i]*recall[i])/(precision[i]+recall[i]))

    return np.mean(f1)


def jaccard_with_cm(conf_matrix):
    den = float(np.sum(conf_matrix[:, 1]) + np.sum(conf_matrix[1]) - conf_matrix[1][1])
    _sum_iou = conf_matrix[1][1] / den if den != 0 else 0

    return _sum_iou
