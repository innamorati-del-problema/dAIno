from time import sleep
from game import Daino
from collections import deque

from game import SmallCactus, LargeCactus, Ptero

MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001

game = Daino()


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

    return (obs_distance, game.game_speed, isinstance(game.obstacles[next], SmallCactus), isinstance(game.obstacles[next], LargeCactus), isinstance(game.obstacles[next], Ptero))


while True:

    game_over, score = game.play_step()

    ds = get_state(game)
    print(ds)

    if game_over:
        print(score)
        game.reset()
