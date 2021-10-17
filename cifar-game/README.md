# ae-comm: CIFAR Game
This repository contains implementation of game environment and models used for `CIFAR Game`
in NeurIPS 2021 submission ***"Learning to Ground Multi-Agent Communication with Autoencoders"***.

## Requirements
Prerequisite: Python 3.6 or higher

To install requirements:
```setup
pip install -r requirements.txt
mkdir data
```

Download content of [this folder](https://drive.google.com/drive/folders/1COJttDgQluEfR9tP8Z2fMF48bDaqQakS?usp=sharing) and unzip to `./data`

## Training
To train the models in the paper, run the following commands:
```train
# no-comm (baseline)
python train.py --set comm_type 0 num_workers 8 --gpu 0

# rl-comm (baseline)
python train.py --set comm_type 2 env_cfg.comm_size 10 num_workers 8 --gpu 0

# ae-rl-comm (baseline)
python train.py --set comm_type 1 env_cfg.comm_size 10 ae_fc_size 10 num_workers 8 --gpu 0

# ae-comm
python train.py --set comm_type 1 aux_loss 'a' env_cfg.comm_size 10 num_workers 8 --gpu 0
```
Tensorboard logs and checkpoints generated during training are saved in `./runs` by default.

## Code layout

| Code          | Detail |
| :-------------: |:-------------:|
| actor_critic/master.py | A3C master weight and optimizer |
| actor_critic/worker.py | asynchronous worker for no-comm / ae-comm / ae-rl-comm |
| actor_critic/worker_pg.py | asynchronous worker for rl-comm |
| actor_critic/evaluator.py | separate worker to compute and log results |
| :-------------: |:-------------:|
| envs/game_environment.py | definition of CIFAR Game environment | 
| :-------------: |:-------------:|
| model/cifar.py | base for all agent models | 

## License
All content in this repository is licensed under the MIT license.