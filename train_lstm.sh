#!/usr/bin/env bash

ENV=pair
MODEL=lstm

python -m train --env $ENV --model $MODEL --T 100 &
python -m train --env $ENV --model $MODEL --T 50  &
python -m train --env $ENV --model $MODEL --T 20  &
python -m train --env $ENV --model $MODEL --T 10  &