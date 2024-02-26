import chess
import numpy as np

def board_to_bitboards(board: chess.Board) -> np.ndarray:
        black, white = board.occupied_co[chess.BLACK], board.occupied_co[chess.WHITE]
        
        bitboards = np.array([
            black & board.pawns,
            black & board.knights,
            black & board.bishops,
            black & board.rooks,
            black & board.queens,
            black & board.kings,
            white & board.pawns,
            white & board.knights,
            white & board.bishops,
            white & board.rooks,
            white & board.queens,
            white & board.kings,
        ], dtype=np.uint64)
        
        return bitboards

def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)

def process_chess_move(chess_move):
    # Define a dictionary to map chess ranks and files to numerical values
    rank_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
    file_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    # Extract source and destination squares from the chess move
    source_square, dest_square = chess_move[:2], chess_move[2:]

    # Convert chess move into a numerical representation
    source_rank, source_file = rank_mapping[source_square[1]], file_mapping[source_square[0]]
    dest_rank, dest_file = rank_mapping[dest_square[1]], file_mapping[dest_square[0]]

    # Represent the move as a NumPy array
    move_array = np.array([source_rank, source_file, dest_rank, dest_file])

    return move_array

def bitboard_to_array(bb: int) -> np.ndarray:
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(8, 8)



def encode_move(move_uci, bitboard):
    rank_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
    file_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    source_square, dest_square = move_uci[:2], move_uci[2:]
    
    source_rank, source_file = rank_mapping[source_square[1]], file_mapping[source_square[0]]
    dest_rank, dest_file = rank_mapping[dest_square[1]], file_mapping[dest_square[0]]
    if len(move_uci) == 5:
        move_uci = move_uci[:4]
    
    piece_channel = bitboard[:, :, int(move_uci[-1])-1]
    
    encoded_move = np.array([source_rank, source_file, dest_rank, dest_file, piece_channel[source_rank, source_file]])

    return encoded_move

def piece_to_channel(move, bitboard):
    # Map piece type to the corresponding channel (plane)
    piece_mapping = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                     'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11}
    
    return piece_mapping[move[-1]]

def moves_from_file(file_path):
    
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            lines = file.readlines()
    np.random.shuffle(data)
    for line in lines:
        board_state, move = line.split("[MOVESEOP]")
        cb = chess.Board(board_state)
        bb = board_to_bitboards(cb)
        ab = bitboards_to_array(bb)

        move = encode_move(move, ab)
        data.append((ab, move))
    return data

def split_data(data, train_size=0.8):
    # Calculate the number of samples for the training set
    train_samples = int(len(data) * train_size)
    
    # Split the data into training and validation sets
    train_data = data[:train_samples]
    val_data = data[train_samples:]
    
    return train_data, val_data


def generate_batches(data, batch_size):
    # Shuffle the data
    np.random.shuffle(data)
    
    # Calculate the number of batches
    num_batches = len(data) // batch_size
    
    # Generate the batches
    for i in range(0, len(data),  batch_size):
        batch = data[i:i + batch_size]
        x_batch, y_batch = zip(*batch)
        yield np.array(x_batch), np.array(y_batch)