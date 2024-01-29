import chess

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        
        # self.action_space = chess.Move.uci(self.board.legal_moves)
        # self.observation_space = self.board.fen()

    def legal_moves(self):
        return list self.board.legal_moves
    
    def make_move(self, move):
        if self.board.is_fivefold_repetition():
            return -1
        leagal_moves = leagal_moves()
        if move in leagal_moves:
            self.board.push_uci(move)