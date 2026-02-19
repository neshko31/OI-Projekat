import numpy as np
import pygame
import sys
import math
import random

ROW_COUNT = 6
COLUMN_COUNT = 6
CONNECTIONS_NEEDED = 4

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

BLUE = (0,0,255)
BLACK = (255,255,255)
RED = (255,0,0)
YELLOW = (255,255,0)

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

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # treba refaktorisati da samo gleda gde je poslednji deo stavljen
    # Prvo gledamo horizontalne lokacije
    for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Zatim gledamo vertikalne lokacije
    for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
        for c in range(COLUMN_COUNT):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Zatim proveravamo sporedne dijagonale (pozitivno orijentisane u odnosu na ispis)
    for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
        for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    # Konacno proveravamo glavne diagonale (negativno orijentisane u odnosu na ispis)
    for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
        for r in range(CONNECTIONS_NEEDED - 1, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def evaluate_window(window, piece):
    score = 0
    
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE
    else:
        opp_piece = PLAYER_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1: # ovo nije bas 3 in a row :/
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1: # ovo nije bas 3 in a row :/
        score -= 4

    return score

def score_position(board, piece):
    score = 0

    # Score center kolona
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += 3 * center_count

    # Score horizontalno
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score vertikalno
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)
            
    # Score pozitivno orijentisane dijagonale, u odnosu na ispis
    for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
        for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Score negativno orijentisane dijagonale, u odnosu na ispis
    for r in range(ROW_COUNT - CONNECTIONS_NEEDED + 1):
        for c in range(COLUMN_COUNT - CONNECTIONS_NEEDED + 1):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def alphabeta(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)
    if terminal:
        if winning_move(board, AI_PIECE):
            return (None, 10000)
        elif winning_move(board, PLAYER_PIECE):
            return (None, -10000)
        else:
            return (None, 0)
    elif depth == 0:
        return (None, score_position(board, AI_PIECE))
    
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = alphabeta(b_copy, depth-1, alpha, beta, False)[1]
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
            new_score = alphabeta(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def minimax(board, depth, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)
    if terminal:
        if winning_move(board, AI_PIECE):
            return (None, 10000)
        elif winning_move(board, PLAYER_PIECE):
            return (None, -10000)
        else:
            return (None, 0)
    elif depth == 0:
        return (None, score_position(board, AI_PIECE))
    
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, False)[1]
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
            new_score = minimax(b_copy, depth-1, True)[1]
            if new_score < value:
                value = new_score
                column = col
        return column, value

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

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


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, YELLOW, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE/2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE/2), height - int(r * SQUARESIZE + SQUARESIZE/2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, BLUE, (int(c * SQUARESIZE + SQUARESIZE/2), height - int(r * SQUARESIZE  + SQUARESIZE/2)), RADIUS)
    pygame.display.update()

game_over = False
board = create_board()
turn = 0

pygame.init()

SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

turn = random.randint(PLAYER, AI)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
        pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            # Pitaj igraca 1 za unos
            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)

                    if winning_move(board, PLAYER_PIECE):
                        label = myfont.render("Igrac 1 je pobednik!", 1, RED)
                        screen.blit(label, (40,10))
                        game_over = True
                    
                    turn += 1
                    turn = turn % 2
                    draw_board(board)

        ## Pitaj igraca 2 za unos
    if turn == AI and not game_over:
        # col = random.randint(0, COLUMN_COUNT - 1)
        # col = pick_best_move(board, AI_PIECE)
        # col, minimax_score = minimax(board, 1, True)
        col, score = alphabeta(board, 1, -math.inf, math.inf, True)
        if col is not None and is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)
            if winning_move(board, AI_PIECE):
                label = myfont.render("Igrac 2 je pobednik!", 1, BLUE)
                screen.blit(label, (40,10))
                game_over = True
        
            draw_board(board)
            turn += 1
            turn = turn % 2
   
    if game_over:
        pygame.time.wait(3000)