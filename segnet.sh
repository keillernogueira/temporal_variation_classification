#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python main.py --operation Train --output_path segnet/64/2016/ --dataset_input_path /home/kno/datasets/MDPI/ --image_name CED-SX260-2016-09-25-MOS-GEO.tif --model_name segnet --crop_size 64 --stride_crop 32 --batch_size 128 > segnet/64/2016/out.txt

CUDA_VISIBLE_DEVICES=3 python main.py --operation Train --output_path segnet/64/2017/ --dataset_input_path /home/kno/datasets/MDPI/ --image_name CED-SX260-2017-01-05-MOS-GEO.tif --model_name segnet --crop_size 64 --stride_crop 32 --batch_size 128 > segnet/64/2017/out.txt

CUDA_VISIBLE_DEVICES=3 python main.py --operation Train --output_path segnet/64/stack/ --dataset_input_path /home/kno/datasets/MDPI/ --image_name stack_CED_2016_09_25_2017_01_05.tif --model_name segnet --crop_size 64 --stride_crop 32 --batch_size 128 > segnet/64/stack/out.txt

##
CUDA_VISIBLE_DEVICES=3 python main.py --operation Train --output_path segnet/128/2016/ --dataset_input_path /home/kno/datasets/MDPI/ --image_name CED-SX260-2016-09-25-MOS-GEO.tif --model_name segnet --crop_size 128 --stride_crop 64 --batch_size 64 > segnet/128/2016/out.txt

CUDA_VISIBLE_DEVICES=3 python main.py --operation Train --output_path segnet/128/2017/ --dataset_input_path /home/kno/datasets/MDPI/ --image_name CED-SX260-2017-01-05-MOS-GEO.tif --model_name segnet --crop_size 128 --stride_crop 64 --batch_size 64 > segnet/128/2017/out.txt

CUDA_VISIBLE_DEVICES=3 python main.py --operation Train --output_path segnet/128/stack/ --dataset_input_path /home/kno/datasets/MDPI/ --image_name stack_CED_2016_09_25_2017_01_05.tif --model_name segnet --crop_size 128 --stride_crop 64 --batch_size 64 > segnet/128/stack/out.txt


