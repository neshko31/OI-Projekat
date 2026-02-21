import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  Konstante igre
# ─────────────────────────────────────────────
ROW_COUNT = 6
COLUMN_COUNT = 6
CONNECTIONS_NEEDED = 4

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

# ─────────────────────────────────────────────
#  Logika table
# ─────────────────────────────────────────────
def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def get_valid_locations(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Horizontalno
    for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
        for r in range(ROW_COUNT):
            if all(board[r][c + i] == piece for i in range(CONNECTIONS_NEEDED)):
                return True
    # Vertikalno
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
            if all(board[r + i][c] == piece for i in range(CONNECTIONS_NEEDED)):
                return True
    # Dijagonala (pozitivno orijentisana)
    for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
        for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
            if all(board[r + i][c + i] == piece for i in range(CONNECTIONS_NEEDED)):
                return True
    # Dijagonala (negativno orijentisana)
    for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
        for r in range(CONNECTIONS_NEEDED - 1, ROW_COUNT):
            if all(board[r - i][c + i] == piece for i in range(CONNECTIONS_NEEDED)):
                return True
    return False

# Koristi se za proveru u algoritmima da se vidi da li je došlo do kraja igre (igrač pobedio, AI pobedio, nerešeno)
def is_terminal_node(board):
    return (winning_move(board, PLAYER_PIECE) or
            winning_move(board, AI_PIECE) or
            len(get_valid_locations(board)) == 0)

# ─────────────────────────────────────────────
#  Heuristika (bodovanje)
# ─────────────────────────────────────────────
def evaluate_window(window, piece):
    score = 0
    opp_piece = AI_PIECE if piece == PLAYER_PIECE else PLAYER_PIECE

    if window.count(piece) == 4:
        score += 100_000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 500
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 100
    
    if window.count(opp_piece) == 4:
        score -= 80_000
    elif window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 8000
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 80

    return score

def score_position(board, piece):
    score = 0

    # Centar
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    score += 10 * center_array.count(piece)
    if COLUMN_COUNT % 2 == 0:
        center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2 + 1])]
        score += 10 * center_array.count(piece)

    # Horizontalno
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Vertikalno
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Dijagonale
    for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
        for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
        for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

# ─────────────────────────────────────────────
#  Alpha-Beta Pruning
# ─────────────────────────────────────────────
def alphabeta(board, depth, alpha, beta, maximizing_player):
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)

    if terminal:
        if winning_move(board, AI_PIECE):
            return None, 10_000_000
        elif winning_move(board, PLAYER_PIECE):
            return None, -10_000_000
        else:
            return None, 0
    if depth == 0:
        return None, score_position(board, AI_PIECE)

    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = alphabeta(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = alphabeta(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

# ─────────────────────────────────────────────
#  Minimax
# ─────────────────────────────────────────────
def minimax(board, depth, maximizing_player):
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)

    if terminal:
        if winning_move(board, AI_PIECE):
            return None, 10_000_000
        elif winning_move(board, PLAYER_PIECE):
            return None, -10_000_000
        else:
            return None, 0
    if depth == 0:
        return None, score_position(board, AI_PIECE)

    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, False)[1]
            if new_score > value:
                value = new_score
                column = col
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, True)[1]
            if new_score < value:
                value = new_score
                column = col
        return column, value

# ─────────────────────────────────────────────
#  Logika koju koristi protivnik agenta
# ─────────────────────────────────────────────
def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -np.inf
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
    
    return best_col

# ─────────────────────────────────────────────
#  Funkcija za igranje igre između 2 bota
# ─────────────────────────────────────────────
def play_game(ai_depth, algo_name='minimax'):
    turn = random.randint(PLAYER, AI)
    game_over = False
    board = create_board()
    ai_move_times = []
    game_result = 0 # Moguce vrednosti: 0 - nereseno, 1 - agent pobeda, -1 - poraz

    while not game_over:
        if turn == PLAYER and not game_over:
            col = pick_best_move(board, PLAYER_PIECE)
            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)
                if winning_move(board, PLAYER_PIECE):
                    game_over = True
                    game_result = -1
                if len(get_valid_locations(board)) == 0:
                    game_over = True
                    game_result = 0
                turn += 1
                turn = turn % 2
        
        if turn == AI and not game_over:
            t_start = time.perf_counter()
            if algo_name == 'minimax':
                col, _ = minimax(board, ai_depth, True)
            else:
                col, _ = alphabeta(board, ai_depth, -math.inf, math.inf, True)
            if col is None:
                valid_cols = get_valid_locations(board)
                if valid_cols:
                    col = random.choice(valid_cols)
            t_end = time.perf_counter()
            elapsed = t_end - t_start
            ai_move_times.append(elapsed)
            
            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)
                if winning_move(board, AI_PIECE):
                    game_over = True
                    game_result = 1
                if len(get_valid_locations(board)) == 0:
                    game_over = True
                    game_result = 0
                turn += 1
                turn = turn % 2
    
    return np.mean(ai_move_times), min(ai_move_times), max(ai_move_times), np.median(ai_move_times), len(ai_move_times), game_result

# ─────────────────────────────────────────────
#  Pokretanje koda za agentove algoritme i iscrtavanje
# ─────────────────────────────────────────────
# Potrebne konstante
NUMBER_OF_GAMES = 50
ROUNDING_CONST = 6
random.seed(17 + 13)

# ============================================================
# Grafici za Minimax algoritam
# ============================================================

# Dubine za Minimax algoritam
minimax_depth = [1,2,3,4,5,6]

# Ovde cemo cuvati rezultate utakmica za minimax algoritam u sledecem formatu
# dubina: niz tuples koji sadrzi (prosek, najbrze vreme, najsporije vreme, medijana, broj poteza, zavrsetak igre)
# Velicina niza predstavlja NUMBER_OF_GAMES
minimax_dict = dict()
minimax_graph_min = []
minimax_graph_max = []
minimax_graph_median = []
minimax_graph_mean = []

for mm_depth in minimax_depth:
    minimax_dict[mm_depth] = []
    for game_num in range(NUMBER_OF_GAMES):
        avg_time, min_time, max_time, med_time, num_of_moves, game_result = play_game(mm_depth)
        minimax_dict[mm_depth].append((avg_time, min_time, max_time, med_time, num_of_moves, game_result))
    avg_times, min_times, max_times, med_times, moves, results = zip(*minimax_dict[mm_depth])

    print(f"----- Minimax, depth = {mm_depth} -----")
    print("Avg: ", round(np.mean(avg_times), ROUNDING_CONST))
    print("Min: ", round(min(min_times), ROUNDING_CONST))
    print("Max: ", round(max(max_times), ROUNDING_CONST))
    print("Med: ", round(np.median(med_times), ROUNDING_CONST))
    print("Num moves: ", round(np.mean(moves), ROUNDING_CONST))
    print("Win: ", results.count(1))
    print("Loss: ", results.count(-1))
    print("Tie: ", results.count(0))
    print("Games played: ", len(avg_times))
    print()

    minimax_graph_min.append(min(min_times))
    minimax_graph_max.append(max(max_times))
    minimax_graph_median.append(np.median(med_times))
    minimax_graph_mean.append(np.mean(avg_times))

# Log skala za Minimax
plt.figure(figsize=(10, 6))

plt.plot(minimax_depth, minimax_graph_min, marker='o', label='Najbrži potez')
plt.plot(minimax_depth, minimax_graph_max, marker='o', label='Najsporiji potez')
plt.plot(minimax_depth, minimax_graph_mean, marker='o', label='Prosečno trajanje poteza')
plt.plot(minimax_depth, minimax_graph_median, marker='o', label='Medijalno trajanje poteza')

plt.xlabel('Dubina pretrage')
plt.ylabel('Vreme trajanja poteza (s)')
plt.yscale('log')
plt.title('Performanse Minimax algoritma po dubini pretrage (logaritamska skala)')
plt.xticks(minimax_depth)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/Minimax-Log.png')
plt.show()

# Obicna skala za Minimax
plt.figure(figsize=(10, 6))

plt.plot(minimax_depth, minimax_graph_min, marker='o', label='Najbrži potez')
plt.plot(minimax_depth, minimax_graph_max, marker='o', label='Najsporiji potez')
plt.plot(minimax_depth, minimax_graph_mean, marker='o', label='Prosečno trajanje poteza')
plt.plot(minimax_depth, minimax_graph_median, marker='o', label='Medijalno trajanje poteza')

plt.xlabel('Dubina pretrage')
plt.ylabel('Vreme trajanja poteza (s)')
plt.title('Performanse Minimax algoritma po dubini pretrage')
plt.xticks(minimax_depth)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/Minimax.png')
plt.show()

# ============================================================
# Grafici za Alpha-beta pruning algoritam
# ============================================================

# Dubine za Alpha-beta pruning algoritam
alphabeta_depth = [1,2,3,4,5,6]

# Ovde cemo cuvati rezultate utakmica za Alpha-beta pruning algoritam u sledecem formatu
# dubina: niz tuples koji sadrzi (prosek, najbrze vreme, najsporije vreme, medijana, broj poteza, zavrsetak igre)
# Velicina niza predstavlja NUMBER_OF_GAMES
alphabeta_dict = dict()
alphabeta_graph_min = []
alphabeta_graph_max = []
alphabeta_graph_median = []
alphabeta_graph_mean = []

for ab_depth in alphabeta_depth:
    alphabeta_dict[ab_depth] = []
    for game_num in range(NUMBER_OF_GAMES):
        avg_time, min_time, max_time, med_time, num_of_moves, game_result = play_game(ab_depth, 'alpha-beta')
        alphabeta_dict[ab_depth].append((avg_time, min_time, max_time, med_time, num_of_moves, game_result))
    avg_times, min_times, max_times, med_times, moves, results = zip(*alphabeta_dict[ab_depth])

    print(f"----- Alpha-beta pruning, depth = {ab_depth} -----")
    print("Avg: ", round(np.mean(avg_times), ROUNDING_CONST))
    print("Min: ", round(min(min_times), ROUNDING_CONST))
    print("Max: ", round(max(max_times), ROUNDING_CONST))
    print("Med: ", round(np.median(med_times), ROUNDING_CONST))
    print("Num moves: ", round(np.mean(moves), ROUNDING_CONST))
    print("Win: ", results.count(1))
    print("Loss: ", results.count(-1))
    print("Tie: ", results.count(0))
    print("Games played: ", len(avg_times))
    print()

    alphabeta_graph_min.append(min(min_times))
    alphabeta_graph_max.append(max(max_times))
    alphabeta_graph_median.append(np.median(med_times))
    alphabeta_graph_mean.append(np.mean(avg_times))

# Log skala za Alpha-beta pruning
plt.figure(figsize=(10, 6))

plt.plot(alphabeta_depth, alphabeta_graph_min, marker='o', label='Najbrži potez')
plt.plot(alphabeta_depth, alphabeta_graph_max, marker='o', label='Najsporiji potez')
plt.plot(alphabeta_depth, alphabeta_graph_mean, marker='o', label='Prosečno trajanje poteza')
plt.plot(alphabeta_depth, alphabeta_graph_median, marker='o', label='Medijalno trajanje poteza')

plt.xlabel('Dubina pretrage')
plt.ylabel('Vreme trajanja poteza (s)')
plt.yscale('log')
plt.title('Performanse Alpha-beta pruning algoritma po dubini pretrage (logaritamska skala)')
plt.xticks(alphabeta_depth)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/Alpha-beta-Log.png')
plt.show()

# Obicna skala za Alpha-beta pruning
plt.figure(figsize=(10, 6))

plt.plot(alphabeta_depth, alphabeta_graph_min, marker='o', label='Najbrži potez')
plt.plot(alphabeta_depth, alphabeta_graph_max, marker='o', label='Najsporiji potez')
plt.plot(alphabeta_depth, alphabeta_graph_mean, marker='o', label='Prosečno trajanje poteza')
plt.plot(alphabeta_depth, alphabeta_graph_median, marker='o', label='Medijalno trajanje poteza')

plt.xlabel('Dubina pretrage')
plt.ylabel('Vreme trajanja poteza (s)')
plt.title('Performanse Alpha-beta pruning algoritma po dubini pretrage')
plt.xticks(alphabeta_depth)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/Alpha-beta.png')
plt.show()

# ============================================================
# Uporedno poređenje Minimax vs Alpha-beta pruning
# ============================================================

# Zajednički niz dubina (pretpostavljamo da su identični)
common_depths = minimax_depth

# --- Uporedni graf: normalna skala ---
plt.figure(figsize=(10, 6))

plt.plot(common_depths, minimax_graph_mean, marker='o', color='steelblue', linewidth=2, label='Minimax - prosečno vreme')
plt.plot(common_depths, alphabeta_graph_mean, marker='s', color='tomato', linewidth=2, label='Alpha-beta pruning - prosečno vreme')

# Anotacija svake tačke sa vrednostima
for d, mm, ab in zip(common_depths, minimax_graph_mean, alphabeta_graph_mean):
    plt.annotate(f'{mm:.4f}s', xy=(d, mm),
                 xytext=(4, 6), textcoords='offset points',
                 fontsize=8, color='steelblue')
    plt.annotate(f'{ab:.4f}s', xy=(d, ab),
                 xytext=(4, -14), textcoords='offset points',
                 fontsize=8, color='tomato')

plt.xlabel('Dubina pretrage')
plt.ylabel('Prosečno vreme trajanja poteza (s)')
plt.title('Minimax vs Alpha-beta pruning - prosečno vreme trajanja poteza po dubini')
plt.xticks(common_depths)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/Comparison.png')
plt.show()

# --- Uporedni graf: logaritamska skala ---
plt.figure(figsize=(10, 6))

plt.plot(common_depths, minimax_graph_mean, marker='o', color='steelblue', linewidth=2, label='Minimax - prosečno vreme')
plt.plot(common_depths, alphabeta_graph_mean, marker='s', color='tomato', linewidth=2, label='Alpha-beta pruning - prosečno vreme')

# Anotacija svake tačke sa vrednostima
for d, mm, ab in zip(common_depths, minimax_graph_mean, alphabeta_graph_mean):
    plt.annotate(f'{mm:.4f}s', xy=(d, mm),
                 xytext=(4, 6), textcoords='offset points',
                 fontsize=8, color='steelblue')
    plt.annotate(f'{ab:.4f}s', xy=(d, ab),
                 xytext=(4, -14), textcoords='offset points',
                 fontsize=8, color='tomato')

plt.xlabel('Dubina pretrage')
plt.ylabel('Prosečno vreme trajanja poteza (s)')
plt.yscale('log')
plt.title('Minimax vs Alpha-beta pruning - prosečno vreme trajanja poteza po dubini (logaritamska skala)')
plt.xticks(common_depths)
plt.legend(loc='best')
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('./graphs/Comparison-Log.png')
plt.show()