#!/bin/bash
## AMP + DDP ###################################################################

bash xtrain.sh bce     bce  0
wait
bash xtrain.sh dice    bce  0
wait
bash xtrain.sh l2      bce  0
wait

bash xtrain.sh bce     dice  1
wait
bash xtrain.sh bce     dice  2
wait
bash xtrain.sh bce     dice  4
wait
bash xtrain.sh bce     dice  8
wait
bash xtrain.sh bce     dice  10
wait


bash xtrain.sh bce     l2  1
wait
bash xtrain.sh bce     l2  2
wait
bash xtrain.sh bce     l2  4
wait
bash xtrain.sh bce     l2  8
wait
bash xtrain.sh bce     l2  10
wait

bash xtrain.sh dice     l2  1
wait
bash xtrain.sh dice     l2  2
wait
bash xtrain.sh dice     l2  4
wait
bash xtrain.sh dice     l2  8
wait
bash xtrain.sh dice     l2  10
wait


