import chess
import numpy as np


class BitBoardEnv:
    def __init__(self):
        self.done = False
        self.board = chess.Board()
        bit_board = self.board_to_biboards()
        self.state = self.bitboards_to_array(bit_board)
        
        self.atacked_squares = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: []
        }
            
    # this show witch matrix the peice is on
    bb_key = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11
    }
    # wich square a piece is on
    file_key = {
        #file
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
        6: 'g',
        7: 'h',
    }

    def reset(self):
        self.done = False
        self.board = chess.Board()
        bit_board = self.board_to_biboards()
        self.state = self.bitboards_to_array(bit_board)
        return self.bitboard_to_input(self.state)

    def board_to_biboards(self):
        black, white = self.board.occupied_co[chess.BLACK], self.board.occupied_co[chess.WHITE]
        
        bitboards = np.array([
            black & self.board.pawns,
            black & self.board.knights,
            black & self.board.bishops,
            black & self.board.rooks,
            black & self.board.queens,
            black & self.board.kings,
            white & self.board.pawns,
            white & self.board.knights,
            white & self.board.bishops,
            white & self.board.rooks,
            white & self.board.queens,
            white & self.board.kings,
        ], dtype=np.uint64)
    
        return bitboards
    


    def bitboard_to_array(self, bb: int) -> np.ndarray:
        s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
        b = (bb >> s).astype(np.uint8)
        b = np.unpackbits(b, bitorder="little")
        return b.reshape(8, 8)

    def bitboards_to_array(self, bb: np.ndarray) -> np.ndarray:
        bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
        s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
        b = (bb >> s).astype(np.uint8)
        b = np.unpackbits(b, bitorder="little")
        return b.reshape(-1, 8, 8)

    def board_value(self) -> tuple:
        #sums the value of the pieces on the board 
        # return the value of the black and white pieces
        # as tuple (black, white)
        b_value, w_value = 0, 0
        # pawns
        b_value += np.sum(self.state[0])
        w_value += np.sum(self.state[6])
        # knights and bishops
        b_value += (np.sum(self.state[1]) + np.sum(self.state[2]))*2
        w_value += (np.sum(self.state[7]) + np.sum(self.state[8]))*2
        # rooks
        b_value += np.sum(self.state[3])*5
        w_value += np.sum(self.state[9])*5
        # queens
        b_value += np.sum(self.state[4])*10
        w_value += np.sum(self.state[10])*10
        # kings
        b_value += np.sum(self.state[5])*100
        w_value += np.sum(self.state[11])*100

        return b_value, w_value
    
    def bitboard_to_input(self, bitboard: np.ndarray) -> np.ndarray:
        # Prepares the bitboard for input
        input_channels = [bitboard.astype(np.float32) for bitboard in bitboard]  # Use 'bitboard' instead of 'self.state'
        input_array = np.stack(input_channels, axis=0)  # Use axis=0 to stack along the first dimension
    
        return input_array
    
    def calculate_reward(self):
        board_value = self.board_value()
        if self.board.turn:
            return board_value[1] - board_value[0]
        else:
            return board_value[0] - board_value[1]
    
    def step(self, action):
        
        bitboard = self.board_to_biboards() 
        self.state = self.bitboards_to_array(bitboard)
        next_input = self.bitboard_to_input(self.state)
        reward = self.calculate_reward()
        done = self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition() or self.board.is_insufficient_material() or self.board.is_stalemate() or self.board.is_checkmate() or self.board.is_game_over()
        done = self.board.is_game_over()

        return next_input, reward, done
    
    def make_move(self, action):
    
        self.board.push(action)
        bitboard = self.board_to_biboards()
        self.state = self.bitboards_to_array(bitboard)
        return self.bitboard_to_input(self.state)
    
  
    