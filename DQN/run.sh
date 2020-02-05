#!/bin/bash
python run_experiment.py \
    -e 'BreakoutNoFrameskip-v4' \
    -t 3e6 \
    -start_eps 1 \
    -end_eps 0.1 \
    -n_iter_decay 1e6 \
    -refresh_freq 5000 \
    -batch_size 16