from time import sleep
from game import Daino
from collections import deque
import numpy as np
from game import SmallCactus, LargeCactus, Ptero
import torch
import random
from model import Linear_QNet, QTrainer
import torch.nn as nn
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

game = Daino()

n_games = 0
epsilon = 0  # randomness
gamma = 0.9  # discount rate
memory = deque(maxlen=MAX_MEMORY)

model_folder_path = './model'
file_name = 'model.pth'
file_name = os.path.join(model_folder_path, file_name)
model = torch.load(file_name)
model.eval()
#trainer = QTrainer(model, model, lr=LR, gamma=gamma)


def get_state(game):
    min_distance = 800
    

    isSmallCactus = False
    isLargeCactus = False
    isPtero = False
    ptero_low = 0

    for obstacle in game.obstacles:
        # print(obstacle)
        if obstacle and obstacle.rect.x < min_distance:
            min_distance = obstacle.rect.x
            isSmallCactus = isinstance(obstacle, SmallCactus)
            isLargeCactus = isinstance(obstacle, LargeCactus)
            isPtero = isinstance(obstacle, Ptero)
            if isPtero:
                ptero_low = 1 if obstacle.rect.y == 210 else 0

    min_distance = min_distance//10

    state = [max(min_distance, 0), game.game_speed, isSmallCactus, isLargeCactus, isPtero, ptero_low]

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
    final_move = [0,0,0,0]
    state0 = torch.tensor(state, dtype=torch.float)
    prediction = model(state0)
    pred_probab = nn.Softmax(dim=0)(prediction)

    move = pred_probab.argmax(0)

	# move = torch.argmax(prediction).item()
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
        continue

    # perform move and get new state
    game_over, score, reward = game.play_step(final_move)
    state_new = get_state(game)

    if game_over:
        # train long memory / experience replay, plot results
        game.reset()
        n_games += 1
        print(f"Game n: {n_games}")
