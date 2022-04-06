import sys
import datetime
import pathlib
import math

from PIL import Image
from skimage import img_as_float
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, jaccard_score

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from config import *
from utils import *
from data_utils import load_image

from dataloader import DataLoader
from networks.factory import model_factory

Image.MAX_IMAGE_PIXELS = None


def test(test_loader, model, epoch):
    # Setting network for evaluation mode.
    model.eval()

    track_cm = np.zeros((test_loader.dataset.num_classes, test_loader.dataset.num_classes))
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labels = data[0], data[1]

            # Casting to cuda variables.
            inps = Variable(inps).cuda()

            # Forwarding.
            outs = model(inps)

            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)
            prds = soft_outs.cpu().data.numpy().argmax(axis=1).flatten()
            labels = labels.flatten()

            # filtering out pixels
            coord = np.where(labels != test_loader.dataset.num_classes)
            labels = labels[coord]
            prds = prds[coord]

            track_cm += confusion_matrix(labels, prds, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        acc = (track_cm[0][0] + track_cm[1][1]) / np.sum(track_cm)
        f1_s = f1_with_cm(track_cm)
        kappa = kappa_with_cm(track_cm)
        jaccard = jaccard_with_cm(track_cm)

        _sum = 0.0
        for k in range(len(track_cm)):
            _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
        nacc = _sum / float(test_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " F1 Score= " + "{:.4f}".format(f1_s) +
              " Kappa= " + "{:.4f}".format(kappa) +
              " Jaccard= " + "{:.4f}".format(jaccard) +
              " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc, f1_s, kappa, track_cm


def train(train_loader, model, criterion, optimizer, epoch):
    # Setting network for training mode.
    model.train()

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining images, labels and paths for batch.
        inps, labels = data[0], data[1]

        # if the current batch does not have samples from all classes
        # print('out i', i, len(np.unique(labels.flatten())))
        # if len(np.unique(labels.flatten())) < 10:
        #     print('in i', i, len(np.unique(labels.flatten())))
        #     continue

        # Casting tensors to cuda.
        inps = Variable(inps).cuda()
        labs = Variable(labels).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = model(inps)

        # Computing loss.
        loss = criterion(outs, labs)

        if math.isnan(loss):
            print('-------------------------NaN-----------------------------------------------')
            print(inps.shape, labels.shape, outs.shape, np.bincount(labels.flatten()))
            print(np.min(inps.cpu().data.numpy()), np.max(inps.cpu().data.numpy()),
                  np.isnan(inps.cpu().data.numpy()).any())
            print(np.min(labels.cpu().data.numpy()), np.max(labels.cpu().data.numpy()),
                  np.isnan(labels.cpu().data.numpy()).any())
            print(np.min(outs.cpu().data.numpy()), np.max(outs.cpu().data.numpy()),
                  np.isnan(outs.cpu().data.numpy()).any())
            print('-------------------------NaN-----------------------------------------------')
            raise AssertionError

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            soft_outs = F.softmax(outs, dim=1)
            # Obtaining predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1).flatten()

            labels = labels.cpu().data.numpy().flatten()

            # filtering out pixels
            coord = np.where(labels != train_loader.dataset.num_classes)
            labels = labels[coord]
            prds = prds[coord]

            acc = accuracy_score(labels, prds)
            conf_m = confusion_matrix(labels, prds, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            f1_s = f1_score(labels, prds, average='weighted')

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i + 1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(train_loader.dataset.num_classes)) +
                  " F1 Score= " + "{:.4f}".format(f1_s) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )
            sys.stdout.flush()


# CUDA_VISIBLE_DEVICES=0 python main.py --operation Train --output_path outputs/ --dataset biologists \
# --dataset_input_path /home/kno/biologists_datasets/MDPI/ --dataset_input_name CED-SX260-2016-09-25-MOS-GEO.tif
# --num_classes 9 --model_name segnet --reference_crop_size 256 --reference_stride_crop 200

def main():
    parser = argparse.ArgumentParser(description='main')
    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation [Options: Train | Test | Test_Full]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset_input_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--image_name', type=str, required=True, help='Dataset name.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=True, help='Stride size')

    # model options
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['segnet', 'deeplab', 'fcnwideresnet', 'ddcn', 'pixelwise'], help='Model to evaluate')
    parser.add_argument('--model_path', type=str, default=None, help='Model path.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=500, help='Number of epochs')
    parser.add_argument('--loss_weight', type=float, nargs='+',
                        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], help='Weight Loss.')

    # dynamic dilated convnet options
    parser.add_argument('--distribution_type', type=str, default=None,
                        help='Distribution type [Options: single_fixed, uniform, multi_fixed, multinomial]')
    parser.add_argument('--values', type=int, nargs='+', default=None, help='Values considered in the distribution.')
    parser.add_argument('--update_type', type=str, default='acc', help='Update type [Options: loss, acc]')

    parser.add_argument('--weight_sampler', type=str2bool, default=False, help='Use weight sampler for loader?')
    args = parser.parse_args()
    print(args)

    # Making sure output directory is created.
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # reading the input image
    # read this image here because it is going to be used for both train and test
    data = img_as_float(load_image(os.path.join(args.dataset_input_path, args.image_name)))
    print('image ', np.min(data), np.max(data), data.shape, data[0, 0, 0], type(data[0, 0, 0]))

    if args.operation == 'Train':
        print('---- training data ----')
        train_set = DataLoader('Train', data, args.dataset_input_path, args.crop_size, args.stride_crop,
                               args.output_path, args.model_name)
        print('---- testing data ----')
        test_set = DataLoader('Test', data, args.dataset_input_path, args.crop_size, args.stride_crop,
                              args.output_path, args.model_name)

        if args.weight_sampler is False:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
        else:
            class_loader_weights = 1. / np.bincount(train_set.gen_classes)
            samples_weights = class_loader_weights[train_set.gen_classes]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights),
                                                                     replacement=True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       num_workers=NUM_WORKERS, drop_last=False, sampler=sampler)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # Setting network architecture.
        model = model_factory(args.model_name, train_set.num_channels, train_set.num_classes).cuda()

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.loss_weight),
                                        ignore_index=train_set.num_classes).cuda()

        # Setting optimizer.
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        curr_epoch = 1
        best_records = []
        if args.model_path is not None:
            print('Loading model ' + args.model_path)
            best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
            model.load_state_dict(torch.load(args.model_path))
            # optimizer.load_state_dict(torch.load(args.model_path.replace("model", "opt")))
            curr_epoch += int(os.path.basename(args.model_path)[:-4].split('_')[-1])
            for i in range(curr_epoch):
                scheduler.step()
        model.cuda()

        # Iterating over epochs.
        print('---- training ----')
        for epoch in range(curr_epoch, args.epoch_num + 1):
            # Training function.
            train(train_loader, model, criterion, optimizer, epoch)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, f1_s, kappa, track_cm = test(test_loader, model, epoch)
                save_best_models(model, args.output_path, best_records, epoch, kappa)
                # patch_acc_loss=None, patch_occur=None, patch_chosen_values=None
            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing data ----')
        test_set = DataLoader('Test', data, args.dataset_input_path, args.crop_size, args.stride_crop,
                              args.output_path, args.model_name)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # Setting network architecture.
        model = model_factory(args.model_name, test_set.num_channels, test_set.num_classes).cuda()

        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['kappa'] < best_records[i]['kappa']:
                index = i
        epoch = int(best_records[index]['epoch'])
        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_' + str(epoch) + '.pth')))
        model.cuda()

        test(test_loader, model, epoch)
    else:
        raise NotImplementedError("Process " + args.operation + "not found!")


if __name__ == "__main__":
    main()
