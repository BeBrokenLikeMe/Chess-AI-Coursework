import copy
import time
from chessmaker.chess.base import (Board, Player, Position)

# Visualise a bitboard

def print_bitboard(bitboard):
    bitboard = bin(bitboard)[2:].zfill(25)
    print("\n      A B C D E ")
    print("    ____________ ")

    for i in range(5):
        # Need to print upside down and back to front to ensure correct formatting
        temp = [' ', 5 - i, "|"] + [*(bitboard[-5:])[::-1]]

        # Replace 0 with . for readability
        temp = [i if i != '0' else '.' for i in temp]

        print(*temp, sep = " ")
        bitboard = bitboard[:-5]

    print("\n")


def enumerate_global_board_square_names():
    # Enumerates all the board square names and make them global
    global  a1, b1, c1, d1, e1, \
            a2, b2, c2, d2, e2, \
            a3, b3, c3, d3, e3, \
            a4, b4, c4, d4, e4, \
            a5, b5, c5, d5, e5

    a1, b1, c1, d1, e1, \
    a2, b2, c2, d2, e2, \
    a3, b3, c3, d3, e3, \
    a4, b4, c4, d4, e4, \
    a5, b5, c5, d5, e5 = range(25)

def enumerate_global_piece_names():
    # Enumerates piece names to match the indexes of board.

    global Wpawn, Bpawn, Wknight, Bknight, Wbishop, Bbishop, Wqueen, Bqueen, Wright,\
        Bright, Wking, Bking, white, black, all, enpassant, player_to_move

    Wpawn, Bpawn, Wknight, Bknight, Wbishop, Bbishop, Wqueen, Bqueen, Wright, \
        Bright, Wking, Bking, white, black, all, enpassant, player_to_move = range(17)

    global piece_names
    piece_names = ['P', 'p', 'N', 'n', 'B', 'b', 'Q', 'q', 'R', 'r', 'K', 'k']

def get_bit(bitboard, square) -> int:
    return 1 if (bitboard & (1<<square)) else 0

def set_bit(bitboard, square) -> int:
    return bitboard | (1<<square)

def remove_bit(bitboard, square) -> int:
    return bitboard ^ (1<<square if get_bit(bitboard, square) else 0)

def ctz(bitboard) -> int: # Count trailing zeroes
    return (bitboard & -bitboard).bit_length() - 1

def update_bitboards(B):
    # union of other bitboards
    B[white] = B[Wpawn] | B[Wright] | B[Wknight] | B[Wbishop] | B[Wking] | B[Wqueen]
    B[black] = B[Bpawn] | B[Bright] | B[Bknight] | B[Bbishop] | B[Bking] | B[Bqueen]
    B[all] = B[white] | B[black]


def position_to_index(x, y) -> int:
    return (y * 5) + x

def index_to_position(index) -> (int, int):
    return index % 5, index // 5

def piece_type_on_square(bitboard, sq):

    if (bitboard[Wpawn]  | bitboard[Bpawn]) & (1 << sq):
        return "pawn"

    if (bitboard[Wknight]  | bitboard[Bknight]) & (1 << sq):
        return "knight"

    if (bitboard[Wbishop] | bitboard[Bbishop]) & (1 << sq):
        return "bishop"

    if (bitboard[Wqueen] | bitboard[Bqueen]) & (1 << sq):
        return "queen"

    if (bitboard[Wright] | bitboard[Bright]) & (1 << sq):
        return "right"

    if (bitboard[Wking] | bitboard[Bking]) & (1 << sq):
        return "king"

    return None

def piece_colour_on_square(bitboard, sq):
    if bitboard[white] & (1 << sq):
        return "white"
    if bitboard[black] & (1 << sq):
        return "black"
    return None

def new_board():

    return [
        992, 1015808,    # White pawns, Black pawns
        16, 1048576,     # White knight, Black knight
        2, 8388608,      # White bishop, Black bishop
        8, 2097152,      # White queen, Black queen
        1,16777216,      # White right, Black right
        4, 4194304,      # White king, Black king
        1023, 33521664,  # White pieces, Black pieces
        33522687,        # All pieces
        0,               # En passant pieces
        0                # Player to move
    ]


def print_board(Bitboard):
    output = [' ' for i in range(25)]

    # can the first 12 bitboards for pieces
    for i in range(12):
        bitboard = Bitboard[i]

        while bitboard:
            square = ctz(bitboard)
            bitboard = remove_bit(bitboard, square)
            output[square] = piece_names[i]

    print("\n      A B C D E ")
    print("    ____________ ")

    for i in range(5):
        # need to print board upside down and back to front to ensure correct formatting
        row = output[20 - 5 * i: 25 - 5 * i]
        temp = [' ', 5 - i, "|"] + row
        print(*temp, sep = " ")

    print("\n")


def get_knight_moves(bitboard, piece) -> list [(int, int)]:
    knight_moves = []
    pos_x, pos_y = index_to_position(piece)

    if piece_colour_on_square(bitboard, piece) == "white":
        own = bitboard[white]
    else:
        own = bitboard[black]

    directions = [
        (2, 1), (2, -1),
        (1, 2), (1, -2),
        (-1, 2), (-1, -2),
        (-2, 1), (-2, -1),
    ]

    for dx, dy in directions:
        new_x = pos_x + dx
        new_y = pos_y + dy

        if not (0 <= new_x < 5 and 0 <= new_y < 5):
            continue

        new_square = position_to_index(pos_x + dx, pos_y + dy)

        if not (own & (1 << new_square)):
            knight_moves.append((piece, new_square))
        else:
            continue

    return knight_moves


def get_rook_moves(bitboard, piece) -> list [(int, int)]:
    rook_moves = []
    pos_x, pos_y = index_to_position(piece)

    if piece_colour_on_square(bitboard, piece) == "white":
        own = bitboard[white]
        opp = bitboard[black]
    else:
        own = bitboard[black]
        opp = bitboard[white]

    directions = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
    ]

    for dx, dy in directions:

        for i in range(1, 5):
            new_x = pos_x + i*dx
            new_y = pos_y + i*dy

            if not (0 <= new_x < 5 and 0 <= new_y < 5):
                break

            new_square = position_to_index(new_x, new_y)

            if own & (1 << new_square):
                break

            if opp & (1 << new_square):
                rook_moves.append((piece, new_square))
                break

            rook_moves.append((piece, new_square))

    return rook_moves



def get_bishop_moves(bitboard, piece) -> list [(int, int)]:
    bishop_moves = []
    pos_x, pos_y = index_to_position(piece)

    if piece_colour_on_square(bitboard, piece) == "white":
        own = bitboard[white]
        opp = bitboard[black]

    else:
        own = bitboard[black]
        opp = bitboard[white]

    directions = [
        (1, 1), (-1, -1),
        (1, -1), (-1, 1)
    ]

    for dx, dy in directions:
        for i in range(1, 5):
            new_x = pos_x + i*dx
            new_y = pos_y + i*dy

            if not (0 <= new_x < 5 and 0 <= new_y < 5):
                break

            new_square = position_to_index(new_x, new_y)

            if own & (1 << new_square):
                break

            if opp & (1 << new_square):
                bishop_moves.append((piece, new_square))
                break

            bishop_moves.append((piece, new_square))

    return bishop_moves

def get_queen_moves(bitboard, piece) -> list [(int, int)]:

    rook_moves = get_rook_moves(bitboard, piece)
    bishop_moves = get_bishop_moves(bitboard, piece)

    queen_moves = rook_moves + bishop_moves

    return queen_moves

def get_right_moves(bitboard, piece) -> list [(int, int)]:

    rook_moves = get_rook_moves(bitboard, piece)
    knight_moves = get_knight_moves(bitboard, piece)

    right_moves = rook_moves + knight_moves

    return right_moves


def get_pawn_moves_and_captures(bitboard, piece) -> (list[(int, int)], list[(int, int)]):
    pawn_moves = []
    pawn_captures = []
    pos_x, pos_y = index_to_position(piece)

    white_to_move = piece_colour_on_square(bitboard, piece) == "white"

    if white_to_move:
        own = bitboard[white]
        opp = bitboard[black]
        forward = 1
        start_rank = 1
    else:
        own = bitboard[black]
        opp = bitboard[white]
        forward = -1
        start_rank = 3

    new_x = pos_x
    new_y = pos_y + forward

    if 0 <= new_x < 5 and 0 <= new_y < 5:
        sq = position_to_index(new_x, new_y)

        if not (own & (1 << sq)) and not (opp & (1 << sq)):
            pawn_moves.append((piece, sq))

            if pos_y == start_rank:
                new_y2 = pos_y + 2 * forward
                if 0 <= new_y2 < 5:
                    sq2 = position_to_index(new_x, new_y2)
                    if not (own & (1 << sq2)) and not (opp & (1 << sq2)):
                        pawn_moves.append((piece, sq2))

    captures = [(1, forward), (-1, forward)]
    ep_bitboard = bitboard[enpassant]

    for dx, dy in captures:
        new_x = pos_x + dx
        new_y = pos_y + dy

        if not (0 <= new_x < 5 and 0 <= new_y < 5):
            continue

        sq = position_to_index(new_x, new_y)

        if opp & (1 << sq):
            pawn_captures.append((piece, sq))

        elif ep_bitboard & (1 << sq):
            pawn_captures.append((piece, sq))

    return pawn_moves, pawn_captures

def get_king_moves(bitboard, piece) -> list [(int, int)]:
    king_moves = []
    pos_x, pos_y = index_to_position(piece)

    if piece_colour_on_square(bitboard, piece) == "white":
        own = bitboard[white]
        opp = bitboard[black]
    else:
        own = bitboard[black]
        opp = bitboard[white]

    directions = [
        (1, 1), (1, -1), (1, 0),
        (0, 1), (0, -1),
        (-1, 1), (-1, -1), (-1, 0)
    ]

    for dx, dy in directions:
        new_x = pos_x + dx
        new_y = pos_y + dy

        if not (0 <= new_x < 5 and 0 <= new_y < 5):
            continue

        new_square = position_to_index(new_x, new_y)

        if own & (1 << new_square):
            continue

        if opp & (1 << new_square):
            king_moves.append((piece, new_square))
            continue

        king_moves.append((piece, new_square))

    return king_moves

def get_pseudo_legal_moves(bitboard, piece) -> list [(int, int)]:

    piece_type = piece_type_on_square(bitboard, piece)

    match piece_type:
        case "knight":
            knight_moves = get_knight_moves(bitboard, piece)
            return knight_moves

        case "bishop":
            bishop_moves = get_bishop_moves(bitboard, piece)
            return bishop_moves

        case "right":
            right_moves = get_right_moves(bitboard, piece)
            return right_moves

        case "pawn":
            pawn_moves, pawn_captures = get_pawn_moves_and_captures(bitboard, piece)
            pawn_moves_and_captures = pawn_moves + pawn_captures
            return pawn_moves_and_captures

        case "queen":
            queen_moves = get_queen_moves(bitboard, piece)
            return queen_moves

        case "king":
            king_moves = get_king_moves(bitboard, piece)
            return king_moves

        case _:
            print("No piece on square specified")

def test_for_check(bitboard) -> bool:

    moves = []

    white_to_move = (bitboard[player_to_move] == 0)

    if white_to_move:
        king_bb = bitboard[Wking]
        opponent_pieces_bb = bitboard[black]
    else:
        king_bb = bitboard[Bking]
        opponent_pieces_bb = bitboard[white]

    if king_bb == 0:
        return False

    king_sq = ctz(king_bb)

    for sq in range(25):
        if opponent_pieces_bb & (1 << sq):
            ptype = piece_type_on_square(bitboard, sq)

            if ptype == "pawn":
                _, temp =  get_pawn_moves_and_captures(bitboard, sq)
                moves += temp
            elif ptype == "knight":
                moves += get_knight_moves(bitboard, sq)
            elif ptype == "bishop":
                moves += get_bishop_moves(bitboard, sq)
            elif ptype == "right":
                moves += get_right_moves(bitboard, sq)
            elif ptype == "queen":
                moves += get_queen_moves(bitboard, sq)
            elif ptype == "king":
                moves += get_king_moves(bitboard, sq)
            else:
                continue

    for (_, target) in moves:
        if target == king_sq:
            return True

    return False

def make_temporary_move(bitboard, start_sq, end_sq):
    new = bitboard.copy()

    ptype = piece_type_on_square(new, start_sq)
    pcol  = piece_colour_on_square(new, start_sq)

    ep_square_bit = new[enpassant]
    new[enpassant] = 0

    white_map = {
        "pawn": Wpawn, "knight": Wknight, "bishop": Wbishop,
        "queen": Wqueen, "right": Wright, "king": Wking
    }
    black_map = {
        "pawn": Bpawn, "knight": Bknight, "bishop": Bbishop,
        "queen": Bqueen, "right": Bright, "king": Bking
    }

    piece_idx = white_map[ptype] if pcol == "white" else black_map[ptype]

    new[piece_idx] = remove_bit(new[piece_idx], start_sq)

    opp_map = black_map if pcol == "white" else white_map

    for name, idx in opp_map.items():
        if new[idx] & (1 << end_sq):
            new[idx] = remove_bit(new[idx], end_sq)
            break

    new[piece_idx] = set_bit(new[piece_idx], end_sq)

    if ptype == "pawn":
        if pcol == "white" and 20 <= end_sq <= 24:
            new[Wpawn] = remove_bit(new[Wpawn], end_sq)
            new[Wqueen] = set_bit(new[Wqueen], end_sq)

        elif pcol == "black" and 0 <= end_sq <= 4:
            new[Bpawn] = remove_bit(new[Bpawn], end_sq)
            new[Bqueen] = set_bit(new[Bqueen], end_sq)

        # Check for enpassant
        if abs(start_sq - end_sq) == 10:
            skipped_sq = (start_sq + end_sq) // 2
            new[enpassant] = (1 << skipped_sq)

    update_bitboards(new)

    return new


def filter_pseudo_legal_moves(bitboard, pseudo_moves):
    legal = []

    for start_sq, end_sq in pseudo_moves:

        test_bb = make_temporary_move(bitboard, start_sq, end_sq)

        if not test_for_check(test_bb):
            legal.append((start_sq, end_sq))

    return legal


def list_legal_moves(bitboard, piece) -> list[(int, int)]:

    pseudo_moves = get_pseudo_legal_moves(bitboard, piece)
    legal_moves = filter_pseudo_legal_moves(bitboard, pseudo_moves)

    return legal_moves


def get_all_legal_moves(bitboard):
    moves = []
    # Determine which color is moving
    current_colour = "white" if bitboard[player_to_move] == 0 else "black"


    for sq in range(25):
        if piece_colour_on_square(bitboard, sq) == current_colour:
            piece_moves = list_legal_moves(bitboard, sq)
            moves.extend(piece_moves)

    return moves

def get_result(bitboard) -> str:

    moves = []
    target_colour = 'white' if bitboard[player_to_move] == 0 else 'black'

    for sq in range(25):
        if target_colour == piece_colour_on_square(bitboard, sq):
            moves += list_legal_moves(bitboard, sq)
    if not moves:
        if test_for_check(bitboard):
            return 'checkmate'
        else:
            return 'stalemate'
    else:
        return ''


def calculate_material(bitboard) -> (int, int):
    black_material_total = 0
    white_material_total = 0

    material_map = {
        "pawn" : 10,
        "knight" : 40,
        "bishop" : 30,
        "right" : 80,
        "queen" : 80,
        "king" : 0
    }

    for sq in range(25):
        piece_type = piece_type_on_square(bitboard, sq)

        if bitboard[white] & (1 << sq):
            white_material_total += material_map[piece_type]

        elif bitboard[black] & (1 << sq):
            black_material_total += material_map[piece_type]

    return white_material_total, black_material_total


def chessmaker_board_to_bitboard(chessmaker_board: Board, chessmaker_player_to_move="white"):

    # This will convert the chessmaker board to a bitboard.

    bitboard = [0] * 17
    key = None

    for square in chessmaker_board:

        if square.piece is None:
            continue

        player_white = square.piece.player.name.lower() == "white"
        piece_on_square = square.piece.name
        square_position = square.position
        square_index = position_to_index(square_position.x,  4 - square_position.y)

        if piece_on_square.lower() == "knight":
            key = Wknight if player_white else Bknight
        elif piece_on_square.lower() == "queen":
            key = Wqueen if player_white else Bqueen
        elif piece_on_square.lower() ==  "bishop":
            key = Wbishop if player_white else Bbishop
        elif piece_on_square.lower() == "right":
            key = Wright if player_white else Bright
        elif piece_on_square.lower() == "pawn":
            key = Wpawn if player_white else Bpawn
        elif piece_on_square.lower() == "king":
            key = Wking if player_white else Bking

        bitboard[key] |= set_bit(bitboard[key], square_index)

    bitboard[player_to_move] = 0 if chessmaker_player_to_move == "white" else 1

    update_bitboards(bitboard)

    return bitboard

def chessmaker_player_to_player(player : Player) -> str:
    return player.name.lower()


def get_chessmaker_move(board, start_sq, end_sq):
    sx, sy = index_to_position(start_sq)
    start_pos = Position(sx, 4 - sy)

    ex, ey = index_to_position(end_sq)
    target_pos = Position(ex, 4 - ey)
    real_piece = board[start_pos].piece

    if not real_piece:
        return None, None

    move_options = real_piece.get_move_options()

    chosen_option = None
    for opt in move_options:
        if opt.position == target_pos:
            chosen_option = opt
            break

    return real_piece, chosen_option

def get_piece_value(piece_type):
    values = {
        "pawn": 100,
        "knight": 300,
        "bishop": 300,
        "right": 500,
        "queen": 900,
        "king": 10000
    }
    return values.get(piece_type, 0)

def order_moves(bitboard, moves, prioritize_move=None):
    scored_moves = []

    for move in moves:
        start_sq, end_sq = move
        score = 0

        if prioritize_move and move == prioritize_move:
            score = 20_000_000

        else:
            victim_type = piece_type_on_square(bitboard, end_sq)

            if victim_type:
                attacker_type = piece_type_on_square(bitboard, start_sq)

                victim_val = get_piece_value(victim_type)
                attacker_val = get_piece_value(attacker_type)

                score = 1_000_000 + (victim_val * 10) - attacker_val

        scored_moves.append((score, move))

    scored_moves.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in scored_moves]


def agent(board, player, var):
    enumerate_global_board_square_names()
    enumerate_global_piece_names()

    bitboard_player = chessmaker_player_to_player(player)
    bitboard = chessmaker_board_to_bitboard(board, bitboard_player)

    time_limit = 13 # Seconds
    current_depth = var[0]

    max_depth = 8

    start_sq, end_sq = iterate(bitboard, max_depth, time_limit)

    piece, move_opt = get_chessmaker_move(board, start_sq, end_sq)

    return piece, move_opt


def iterate(bitboard, var, time_limit):
    max_depth = int(var) if var is not None else 1
    start_time = time.time()

    best_move_so_far = (None, None)

    for depth in range(1, max_depth + 1):

        try:

            if time_limit is not None and (time.time() - start_time) > time_limit:
                raise SearchTimeout

            depth_start = time.time()
            it_alpha = -float("inf")
            it_beta = float("inf")


            board_clone = copy.deepcopy(bitboard)

            score, (start_sq, end_sq) = select_move(
                board_clone, depth, it_alpha, it_beta,
                start_time, time_limit, prioritize_move=best_move_so_far)

            if start_sq is None:
                print(f"Depth {depth} timed out.")
                break

            best_move_so_far = (start_sq, end_sq)

            depth_time = time.time() - depth_start
            print(f"Time is {(time.time() - start_time) :4f}s after depth {depth}")

            if score is not None and abs(score) >= 1_000_000:
                break

        except SearchTimeout:
            print(f"Depth {depth} timed out. Getting best move.")
            return best_move_so_far

    return best_move_so_far


def select_move(bitboard, depth, it_alpha, it_beta, start_time, time_limit, prioritize_move=None):
    if bitboard[player_to_move] == 0:   #If it is white's moves
        return max_value(
            bitboard, it_alpha, it_beta, depth,
            start_time, time_limit, prioritize_move=prioritize_move)
    else:                               #If it is black's move
        return min_value(
            bitboard, it_alpha, it_beta, depth,
            start_time, time_limit, prioritize_move=prioritize_move)


def max_value(bitboard, alpha, beta, depth, start_time, time_limit, prioritize_move=None):
    bc = base_case(bitboard, depth, start_time, time_limit)

    if bc is not None:
        return bc, (None, None)

    best_move = (None, None)
    v = -float("inf")
    legal_moves = get_all_legal_moves(bitboard)
    legal_moves = order_moves(bitboard, legal_moves, prioritize_move=prioritize_move)

    for start_sq, end_sq in legal_moves:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            raise SearchTimeout

        new_board = make_temporary_move(bitboard, start_sq, end_sq)

        new_board[player_to_move] = 1

        score, _ = min_value(new_board, alpha, beta, depth - 1, start_time, time_limit)

        if score > v:
            v = score
            best_move = (start_sq, end_sq)

        if v >= beta:
            return v, best_move

        alpha = max(alpha, v)

    return v, best_move

def min_value(bitboard, alpha, beta, depth, start_time, time_limit, prioritize_move=None, is_root=False):
    bc = base_case(bitboard, depth, start_time, time_limit)

    if bc is not None:
        return bc, (None, None)

    best_move = (None, None)
    v = float("inf")
    legal_moves = get_all_legal_moves(bitboard)
    legal_moves = order_moves(bitboard, legal_moves, prioritize_move=prioritize_move)

    for start_sq, end_sq in legal_moves:

        if time_limit is not None and (time.time() - start_time) > time_limit:
            raise SearchTimeout

        new_board = make_temporary_move(bitboard, start_sq, end_sq)

        new_board[player_to_move] = 0

        score, _ = max_value(new_board, alpha, beta, depth - 1, start_time, time_limit)

        if score < v:
            v = score
            best_move = (start_sq, end_sq)

        if v <= alpha:
            return v, best_move

        beta = min(beta, v)

    return v, best_move


def base_case(bitboard, depth, start_time, time_limit):

    if time_limit is not None and (time.time() - start_time) > time_limit:
        raise SearchTimeout

    result = get_result(bitboard)

    if result == 'checkmate' or result == 'stalemate':
        if bitboard[player_to_move] == 0:
            return -1_000_000 - depth
        else:
            return 1_000_000 + depth

    if depth == 0:
        return evaluate(bitboard)

    return None

def evaluate(bitboard):
    white_material, black_material = calculate_material(bitboard)
    wknight_mobility, bknight_mobility = calculate_knight_mobility(bitboard)

    return (white_material + wknight_mobility) - (black_material + bknight_mobility)

def calculate_knight_mobility(bitboard):
    black_mobility = 0
    white_mobility = 0

    knight_mobility_table = [
        -1, 0, 0, 0, -1,
        0, 2, 2, 2, 0,
        0, 3, 5, 3, 0,
        0, 2, 2, 2, 0,
        -1, 0, 0, 0, -1,
    ]

    for sq in range(25):
        piece_type = piece_type_on_square(bitboard, sq)
        piece_colour = piece_colour_on_square(bitboard, sq)

        if piece_type == "knight":
            if piece_colour == 'white':
                white_mobility += knight_mobility_table[sq]
            else:
                black_mobility += knight_mobility_table[sq]

    return white_mobility, black_mobility


class SearchTimeout(Exception):
    pass
