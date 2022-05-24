from time import sleep
from game import Daino
from collections import deque
import numpy as np
from game import SmallCactus, LargeCactus, Ptero
import torch
import random
from model import Linear_QNet, QTrainer
import torch.nn as nn

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

game = Daino()

n_games = 0
epsilon = 0  # randomness
gamma = 0.9  # discount rate
memory = deque(maxlen=MAX_MEMORY)
model = Linear_QNet(5, 256, 4)
trainer = QTrainer(model, lr=LR, gamma=gamma)


def get_state(game):
    next = 0
    if game.obstacles[0] != None:
        obs_distance = game.obstacles[0].rect.x - game.dino.dino_rect.x
        next = 0
    else:
        obs_distance = game.w
    if game.obstacles[1] != None:
        second_obs_distance = game.obstacles[1].rect.x - game.dino.dino_rect.x
        if second_obs_distance < obs_distance:
            obs_distance = second_obs_distance
            next = 1

    state = [isinstance(game.obstacles[next], SmallCactus),
             max(0, obs_distance), game.game_speed,
             isinstance(game.obstacles[next], LargeCactus),
             isinstance(game.obstacles[next], Ptero)]  # game.obstacles[next].rect.y == 170]

    return np.array(state, dtype=int)


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def train_long_memory():
    if len(memory) > BATCH_SIZE:
        mini_sample = random.sample(
            memory, BATCH_SIZE)  # List of tuples
    else:
        mini_sample = memory

    states, actions, rewards, next_states, dones = zip(*mini_sample)
    trainer.train_step(states, actions, rewards, next_states, dones)


def train_short_memory(state, action, reward, next_state, done):
    trainer.train_step(state, action, reward, next_state, done)


def get_action(state):
    # random moves: tradeoff exploration / exploitation
    epsilon = 100 - n_games
    final_move = [0, 0, 0, 0]
    if random.randint(0, 200) < epsilon:
        move = random.randint(0, 3)
        final_move[move] = 1
    else:
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = model(state0)

        pred_probab = nn.Softmax(dim=0)(prediction)

        move = pred_probab.argmax(0)

        #move = torch.argmax(prediction).item()
        final_move[move] = 1

    return final_move


while True:

    state_old = get_state(game)

    if not game.dino.is_jumping:
        # get move
        final_move = get_action(state_old)
    else:
        game_over, score, reward = game.play_step([0, 0, 0, 1])

        if game_over:
            # train long memory / experience replay, plot results
            game.reset()
            n_games += 1
            print(f"Game n: {n_games}")
            train_long_memory()

        continue

    # perform move and get new state
    game_over, score, reward = game.play_step(final_move)
    state_new = get_state(game)

    # train short memory
    # train_short_memory(state_old, final_move, reward, state_new, game_over)
    remember(state_old, final_move, reward, state_new, game_over)

    if game_over:
        # train long memory / experience replay, plot results
        game.reset()
        n_games += 1
        print(f"Game n: {n_games}")
        train_long_memory()
