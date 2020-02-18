#!/usr/bin/env bash
export PYTHONPATH="$PYTHONPATH:./"

ENVS=('BreakoutNoFrameskip-v4')
# 'CartPole-v1'
# 'MountainCar-v0'
# Acrobot-v1
for i in "${ENVS[@]}"
do
   # echo "$i"
   xvfb-run -a python rl_experiments/DQN/run_experiment.py \
      -e "$i" \
      -t 25000000 \
      -start_eps 1 \
      -end_eps 0.1 \
      -eps_iters 1000000 \
      -refresh_freq 10000 \
      -eval 50000 \
      -batch_size 32 \
      -prioritized False \
      -o ./runs/"$i"
done
