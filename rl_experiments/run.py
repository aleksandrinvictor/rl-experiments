from rl_experiments.common.utils import make_env, get_env_type

from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
from omegaconf import DictConfig
from importlib import import_module

import torch
import click
import logging
import os
import random
import numpy as np
import hydra
import inspect


OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'rmsprop': torch.optim.RMSprop
}


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig) -> None:

    # Init Tensorboard writer
    writer = SummaryWriter(cfg.common.output_path)

    # Set seed
    random.seed(cfg.common.seed)
    np.random.seed(cfg.common.seed)
    torch.manual_seed(cfg.common.seed)

    logger.info('Starting experiment:')
    logger.info(f'Algorithm: {cfg.algorithm.name}')
    logger.info(f'Environment: {cfg.env.name}')
    logger.info(f'Seed: {cfg.common.seed}')

    # Setup environment
    env = make_env(cfg.env.name, cfg.common.seed)
    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    logger.info(f'State shape: {state_shape}')
    logger.info(f'n actions: {n_actions}')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    models_module = import_module('rl_experiments.common.models')
    models = dict(inspect.getmembers(models_module, inspect.isclass))

    model = models[cfg.model.type](state_shape, n_actions).to(device)

    optimizer_name = str(cfg.optimizer.pop('name'))
    optimizer = OPTIMIZERS[optimizer_name](model.parameters(), **cfg.optimizer)

    logger.info(f'optimizer: {optimizer_name}')
    logger.info(f'lr: {cfg.optimizer.lr}')

    # checkpoint = None
    # if cfg.common.resume:
    #     checkpoint = torch.load(cfg.common.checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    alg_module = import_module(
        '.'.join(['rl_experiments', cfg.algorithm.name]))
    cfg.algorithm.pop('name')
    agent = alg_module.Agent(
        model=model, optimizer=optimizer, device=device, **cfg.algorithm)

    alg_module.train(env=env, agent=agent, writer=writer, **cfg.train_spec)

    writer.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
