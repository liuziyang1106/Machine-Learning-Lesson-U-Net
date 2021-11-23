#!/bin/bash
## AMP + DDP ###################################################################

bash xtrain.sh bce     bce  0 unet
wait
bash xtrain.sh dice    bce  0 unet
wait
bash xtrain.sh l2      bce  0 unet
wait

bash xtrain.sh bce     dice  1  unet
wait
bash xtrain.sh bce     dice  2 unet
wait
bash xtrain.sh bce     dice  4 unet
wait
bash xtrain.sh bce     dice  8 unet
wait
bash xtrain.sh bce     dice  10 unet
wait


bash xtrain.sh dice     l2  1 unet
wait
bash xtrain.sh dice     l2  2 unet
wait
bash xtrain.sh dice     l2  4 unet
wait
bash xtrain.sh dice     l2  8 unet
wait
bash xtrain.sh dice     l2  10 unet
wait


bash xtrain.sh bce     bce  0 seunet
wait
bash xtrain.sh dice    bce  0 seunet
wait
bash xtrain.sh l2      bce  0 seunet
wait

bash xtrain.sh bce     dice  1 seunet
wait
bash xtrain.sh bce     dice  2 seunet
wait
bash xtrain.sh bce     dice  4 seunet
wait
bash xtrain.sh bce     dice  8 seunet
wait
bash xtrain.sh bce     dice  10 seunet
wait


bash xtrain.sh dice     l2  1 seunet
wait
bash xtrain.sh dice     l2  2 seunet
wait
bash xtrain.sh dice     l2  4 seunet
wait
bash xtrain.sh dice     l2  8 seunet
wait
bash xtrain.sh dice     l2  10 seunet
wait

