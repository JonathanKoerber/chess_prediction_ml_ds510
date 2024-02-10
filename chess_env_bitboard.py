import chess
import numpy as np


class BitBoardEnv:
    def __init__(self):
        self.board = chess.Board()
        self.black, self.white = self.board.occupied_co[chess.BLACK], self.board.occupied_co[chess.WHITE]
        self.bitboards = np.array([
            self.black & self.board.pawns,
            self.black & self.board.knights,
            self.black & self.board.bishops,
            self.black & self.board.rooks,
            self.black & self.board.queens,
            self.black & self.board.kings,
            self.white & self.board.pawns,
            self.white & self.board.knights,
            self.white & self.board.bishops,
            self.white & self.board.rooks,
            self.white & self.board.queens,
            self.white & self.board.kings,
        ], dtype=np.uint64)

        self.bit_chess = self.bitboards_to_array(self.bitboards)
        
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
        pass

    def bitboard_to_array(bb: int) -> np.ndarray:
        s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
        b = (bb >> s).astype(np.uint8)
        b = np.unpackbits(b, bitorder="little")
        return b.reshape(8, 8)

    def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
        bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
        s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
        b = (bb >> s).astype(np.uint8)
        b = np.unpackbits(b, bitorder="little")
        return b.reshape(-1, 8, 8)


    