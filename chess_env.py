import gym
from gym import spaces
import chess
import chess.engine
import random
import numpy as np




class ChessEnv:
    def __init__(self):
        super(ChessEnv, self).__init__()
        # create board yay
        self.board = chess.Board()
        # observation space is the board
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 6), dtype=np.uint8)
    
        # action space 
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))

    def reset(self):
        self.board.reset()
        return self._get_observation()
    
    def _get_observation(self):
        # convert the board to a 8x8x6 matrix
        observation = np.zeros((8, 8, 6), dtype=int)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                observation[chess.square_rank(square)][chess.square_file(square)][piece.piece_type - 1] = piece.color
        return observation
    
    def step(self, action):
        #convert the action inde to a chess move
        leagal_moves = list(self.board.legal_moves)
        if 0 <= action < len(leagal_moves):
            move = leagal_moves[action]
        else:
            observation = self._get_observation()
            reward = -1
            done = False
            info = {"illegal_move": True}
            return observation, reward, done, info
        self.board.push(move)
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self.board.is_game_over()
        info = {}

        return observation, reward, done, info
    
    def _calculate_reward(self):
        # todo implement a better reward function
        if self.board.is_checkmate():
            return 1
        if self.board.is_stalemate():
            return 0
        return 0.5
        
    def score_move(self):
        chess.engine.SimpleEngine.ponder(self.board)
    
    def make_radom_move(self):
        move = random.choice(list(self.board.legal_moves))
        self.board.push(move)

    
    def make_move(self, move):
        leagal_moves =  self.board.leagal_moves()
        if move in leagal_moves:
            self.board.push_uci(move)
    
    def random_board(max_depth=100):
        board = chess.Board()
        depth = random.randrange(0, max_depth)

        for _ in range(depth):
            moves = list(board.legal_moves)
            ran_move = random.choice(moves)
            board.push(ran_move)
            if board.is_game_over():
                break
        return board
    

# env = ChessEnv()
# observation = env.reset()
# for _ in range(10):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if done:
#         break

# print("Observation:", observation)
# print("Reward:", reward)
# print("Done:", done)
# print("Info:", info)

