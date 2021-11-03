#!/bin/bash
tipath=./images/train/images
tmpath=./images/train/label
ifpath=./images/test

lbd=$3
save_path=./runs/2DUnet_baseline_${lbd}_${1}_loss_aux_$2/

CUDA_VISIBLE_DEVICES=0     python train.py     \
--output_dir          ${save_path}             \
--batch_size          2                        \
--epochs              500                      \
--lr                  1e-3                    \
--print_freq          2                       \
--train_img_folder    ${tipath}               \
--train_mask_folder   ${tmpath}               \
--test_img_folder     ${ifpath}               \
--loss                $1                      \
--aux_loss            $2                      \
--lbd                 ${lbd}                  \
