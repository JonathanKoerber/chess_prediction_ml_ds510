import chess
import tensorflow as tf
from variable_settings import *
from board_conversion import *
from q_network import *
from rewards import * 

    
model = Q_model()
model_target = Q_model()

class ChessEnv():
    def __init__(self):
        self.board = chess.Board()
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = {
            'white' : [],
            'black' : [],
        }
        self.done_history = []
        self.episode_reward_history = []
        self.move_counter = 1
        self.fast_counter = 0
        self.pgn = ''
        self.pgns = []
        pass