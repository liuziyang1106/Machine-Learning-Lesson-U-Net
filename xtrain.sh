#!/bin/bash
tipath=./images/train/images
tmpath=./images/train/label
ifpath=./images/test

lbd=0
save_path=./runs/2DUnet_baseline_${lbd}_BCE_loss/

CUDA_VISIBLE_DEVICES=0     python train.py     \
--output_dir          ${save_path}             \
--batch_size          4                        \
--epochs              500                      \
--lr                  1e-3                     \
--print_freq          2                       \
--train_img_folder    ${tipath}               \
--train_mask_folder   ${tmpath}               \
--test_img_folder     ${ifpath}               \
--loss                BCE                     \
--aux_loss            dice                    \
--lbd                 ${lbd}                  \
