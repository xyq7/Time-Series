#!/usr/bin/env bash

ENV=pair
MODEL=lstm

python train.py --env $ENV --model $MODEL --T 50
python train.py --env $ENV --model $MODEL --T 40
python train.py --env $ENV --model $MODEL --T 30
python train.py --env $ENV --model $MODEL --T 20