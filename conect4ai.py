import numpy as np
import pygame
import sys
import math
import random
import time

# ─────────────────────────────────────────────
#  Konstante igre
# ─────────────────────────────────────────────
ROW_COUNT = 6
COLUMN_COUNT = 6
CONNECTIONS_NEEDED = 4
WINDOW_LENGTH = 4

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

# ─────────────────────────────────────────────
#  Boje (pygame RGB)
# ─────────────────────────────────────────────
C_BOARD      = (30, 100, 200)      # plava tabla
C_EMPTY      = (20, 20, 30)        # tamna rupa
C_RED        = (220, 50, 50)       # crvena boja disa
C_BLUE       = (0, 0, 230)         # plava boja diska
C_BG         = (15, 15, 25)        # pozadina menija
C_WHITE      = (240, 240, 240)     # bela boja 
C_GRAY       = (130, 130, 150)     # siva boja
C_HIGHLIGHT  = (80, 180, 255)      # higlight opcija
C_TIMER_BG   = (30, 30, 45)        # pozadinski tajmer

# ─────────────────────────────────────────────
#  Logika table
# ─────────────────────────────────────────────
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

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
#  Pygame crtanje table
# ─────────────────────────────────────────────
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 6)


def get_piece_color(piece, player_color):
    # Vrati boju diska na osnovu toga koje je boje igrač odabrao.
    player_rgb = C_RED if player_color == "red" else C_BLUE
    ai_rgb     = C_BLUE if player_color == "red" else C_RED
    if piece == PLAYER_PIECE:
        return player_rgb
    elif piece == AI_PIECE:
        return ai_rgb
    return C_EMPTY


def draw_board(screen, board, player_color, height):
    # Crtanje Table
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            # Za svako polje prvo crtamo pravougaonik
            pygame.draw.rect(screen, C_BOARD, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            # Za svako polje unutar pravougaonika crtamo krug
            pygame.draw.circle(screen, C_EMPTY, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    # Crtanje Diskova
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            piece = board[r][c]
            if piece != EMPTY:
                color = get_piece_color(piece, player_color)
                # Crtamo boje tamo gde je postavljen disk, u zavisnosti od toga koji je igrač i kako je igrač birao određujemo boju
                pygame.draw.circle(screen, color, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    pygame.display.update()


def draw_hover(screen, posx, turn, player_color):
    # Crta disk koji prati miš u gornjem redu.
    pygame.draw.rect(screen, C_BG, (0, 0, COLUMN_COUNT * SQUARESIZE, SQUARESIZE))
    if turn == PLAYER:
        color = C_RED if player_color == "red" else C_BLUE
    else:
        color = C_BLUE if player_color == "red" else C_RED
    pygame.draw.circle(screen, color, (posx, SQUARESIZE // 2), RADIUS)


def draw_timer_panel(screen, algo_name, difficulty_name, last_ai_time, move_times, width, height):
    # Prikazuje informacije o algoritmu i vremenima poteza u donjem delu ekrana.
    # Biće korišćeno za dokumentaciju.
    panel_y = (ROW_COUNT + 1) * SQUARESIZE
    panel_h = height - panel_y
    pygame.draw.rect(screen, C_TIMER_BG, (0, panel_y, width, panel_h))

    font_s = pygame.font.SysFont("monospace", 18)
    font_b = pygame.font.SysFont("monospace", 20, bold=True)

    algo_text = font_b.render(f"Algoritam: {algo_name}  |  Težina: {difficulty_name}", True, C_HIGHLIGHT)
    screen.blit(algo_text, (10, panel_y + 8))

    if last_ai_time is not None:
        t_text = font_s.render(f"Poslednji potez AI: {last_ai_time:.4f}s", True, C_WHITE)
        screen.blit(t_text, (10, panel_y + 35))

    if move_times:
        avg = sum(move_times) / len(move_times)
        avg_text = font_s.render(f"Prosek ({len(move_times)} poteza): {avg:.4f}s  |  Max: {max(move_times):.4f}s", True, C_GRAY)
        screen.blit(avg_text, (10, panel_y + 58))

    pygame.display.update()

# ─────────────────────────────────────────────
#  Meni ekran – helper widget-i
# ─────────────────────────────────────────────
def draw_button(screen, rect, text, font, selected=False, hover=False):
    color = C_HIGHLIGHT if selected else (60, 60, 90) if hover else (40, 40, 65)
    text_color = C_BG if selected else C_WHITE
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, C_HIGHLIGHT if selected else C_GRAY, rect, 2, border_radius=8)
    label = font.render(text, True, text_color)
    label_rect = label.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2))
    screen.blit(label, label_rect)


def show_menu(screen):
    # Prikazuje meni i vraća (algorithm, depth, player_color, player_goes_first).
    width  = COLUMN_COUNT * SQUARESIZE
    height = 580

    font_title  = pygame.font.SysFont("monospace", 32, bold=True)
    font_label  = pygame.font.SysFont("monospace", 20, bold=True)
    font_btn    = pygame.font.SysFont("monospace", 18)
    font_hint   = pygame.font.SysFont("monospace", 14)

    # Stanje menija
    sel_algo       = 1   # 0 = minimax, 1 = alphabeta
    sel_difficulty = 1   # 0 = lak, 1 = srednji, 2 = težak
    sel_color      = 0   # 0 = crvena, 1 = plava

    # Opcije koje korisnik može da bira
    difficulties = [("Lak", 1), ("Srednji", 3), ("Težak", 5)]
    algorithms   = ["Minimax", "Alpha-Beta Pruning"]
    colors       = [("Crvena", C_RED), ("Plava", C_BLUE)]

    clock = pygame.time.Clock()
    running = True
    mouse_pos = (0, 0)

    while running:
        screen.fill(C_BG)
        mouse_pos = pygame.mouse.get_pos()

        # Naslov
        title = font_title.render("CONNECT FOUR", True, C_HIGHLIGHT)
        screen.blit(title, (width // 2 - title.get_width() // 2, 20))

        # --- Algoritam ---
        algo_label = font_label.render("Algoritam:", True, C_WHITE)
        screen.blit(algo_label, (20, 80))
        algo_rects = []
        for i, name in enumerate(algorithms):
            r = pygame.Rect(20 + i * (width // 2 - 10), 108, width // 2 - 20, 44)
            algo_rects.append(r)
            hover = r.collidepoint(mouse_pos)
            draw_button(screen, r, name, font_btn, selected=(sel_algo == i), hover=hover)

        # --- Težina ---
        diff_label = font_label.render("Težina:", True, C_WHITE)
        screen.blit(diff_label, (20, 175))
        diff_rects = []
        btn_w = (width - 40) // 3
        for i, (name, _) in enumerate(difficulties):
            r = pygame.Rect(20 + i * btn_w, 203, btn_w - 8, 44)
            diff_rects.append(r)
            hover = r.collidepoint(mouse_pos)
            draw_button(screen, r, name, font_btn, selected=(sel_difficulty == i), hover=hover)

        hint = font_hint.render("Veća dubina = jači AI, dubine su redom 1, 3 i 5.", True, C_GRAY)
        screen.blit(hint, (20, 252))

        # --- Boja igrača ---
        color_label = font_label.render("Tvoja boja (Crveni prvi igra):", True, C_WHITE)
        screen.blit(color_label, (20, 290))
        color_rects = []
        for i, (cname, crgb) in enumerate(colors):
            r = pygame.Rect(20 + i * 210, 318, 180, 50)
            color_rects.append(r)
            hover = r.collidepoint(mouse_pos)
            pygame.draw.rect(screen, crgb if sel_color == i else (40, 40, 65), r, border_radius=8)
            pygame.draw.rect(screen, C_HIGHLIGHT if sel_color == i else C_GRAY, r, 2, border_radius=8)
            pygame.draw.circle(screen, crgb, (r.x + 30, r.y + 25), 16)
            txt = font_btn.render(cname, True, C_BG if sel_color == i else C_WHITE)
            screen.blit(txt, (r.x + 50, r.y + 15))

        # --- Start dugme ---
        start_rect = pygame.Rect(width // 2 - 100, 510, 200, 54)
        hover = start_rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, C_HIGHLIGHT if hover else (50, 150, 80), start_rect, border_radius=12)
        start_txt = font_label.render("IGRAJ", True, C_BG)
        screen.blit(start_txt, (start_rect.x + start_rect.w // 2 - start_txt.get_width() // 2,
                                start_rect.y + 14))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                for i, r in enumerate(algo_rects):
                    if r.collidepoint(pos):
                        sel_algo = i

                for i, r in enumerate(diff_rects):
                    if r.collidepoint(pos):
                        sel_difficulty = i

                for i, r in enumerate(color_rects):
                    if r.collidepoint(pos):
                        sel_color = i

                if start_rect.collidepoint(pos):
                    algo     = algorithms[sel_algo]
                    depth    = difficulties[sel_difficulty][1]
                    diff_name = difficulties[sel_difficulty][0]
                    p_color  = "red" if sel_color == 0 else "blue"
                    p_first  = sel_color == 0
                    return algo, depth, diff_name, p_color, p_first

        clock.tick(60)

# ─────────────────────────────────────────────
#  Kraj igre ekran
# ─────────────────────────────────────────────
def show_end_screen(screen, message, move_times, algo_name, diff_name, width, height):
    font_big  = pygame.font.SysFont("monospace", 36, bold=True)
    font_med  = pygame.font.SysFont("monospace", 20)
    font_s    = pygame.font.SysFont("monospace", 16)

    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    box = pygame.Rect(40, height // 2 - 160, width - 80, 320)
    pygame.draw.rect(screen, C_BG, box, border_radius=14)
    pygame.draw.rect(screen, C_HIGHLIGHT, box, 2, border_radius=14)

    msg_surf = font_big.render(message, True, C_HIGHLIGHT)
    screen.blit(msg_surf, (box.centerx - msg_surf.get_width() // 2, box.y + 20))

    y = box.y + 80
    info_lines = [
        f"Algoritam: {algo_name}",
        f"Težina: {diff_name}",
    ]
    if move_times:
        avg = sum(move_times) / len(move_times)
        info_lines += [
            f"Ukupno AI poteza: {len(move_times)}",
            f"Prosečno vreme poteza: {avg:.4f}s",
            f"Najbrži potez: {min(move_times):.4f}s",
            f"Najsporiji potez: {max(move_times):.4f}s",
            f"Medijana trajanja poteza: {np.median(move_times):.4f}s"
        ]
    else:
        info_lines.append("AI nije odigrao ni jedan potez.")

    for line in info_lines:
        surf = font_med.render(line, True, C_WHITE)
        screen.blit(surf, (box.x + 20, y))
        y += 28

    hint = font_s.render("Bilo koji taster za novi meč ili ESC za izlaz...", True, C_GRAY)
    screen.blit(hint, (box.centerx - hint.get_width() // 2, box.bottom - 35))

    pygame.display.update()
    pygame.time.wait(500)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                waiting = False

# ─────────────────────────────────────────────
#  Glavna petlja igre
# ─────────────────────────────────────────────
def run_game(screen, algo_name, depth, diff_name, player_color, player_goes_first):
    width  = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT + 2) * SQUARESIZE  # +1 hover red, +1 panel za tajmer

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Connect Four - {algo_name} - {diff_name}")

    board = create_board()
    game_over = False
    last_ai_time = None
    move_times   = []          # lista svih AI vremena u ovoj partiji

    turn = PLAYER if player_goes_first else AI

    font_end = pygame.font.SysFont("monospace", 48, bold=True)

    draw_board(screen, board, player_color, (ROW_COUNT + 1) * SQUARESIZE)
    draw_timer_panel(screen, algo_name, diff_name, last_ai_time, move_times, width, height)

    clock = pygame.time.Clock()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, C_BG, (0, 0, width, SQUARESIZE))
                if turn == PLAYER:
                    draw_hover(screen, event.pos[0], turn, player_color)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if turn == PLAYER:
                    pygame.draw.rect(screen, C_BG, (0, 0, width, SQUARESIZE))
                    col = int(math.floor(event.pos[0] / SQUARESIZE))

                    if 0 <= col < COLUMN_COUNT and is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, PLAYER_PIECE)

                        if winning_move(board, PLAYER_PIECE):
                            draw_board(screen, board, player_color, (ROW_COUNT + 1) * SQUARESIZE)
                            draw_timer_panel(screen, algo_name, diff_name, last_ai_time, move_times, width, height)
                            show_end_screen(screen, "Pobedio si!", move_times, algo_name, diff_name, width, height)
                            return

                        if len(get_valid_locations(board)) == 0:
                            draw_board(screen, board, player_color, (ROW_COUNT + 1) * SQUARESIZE)
                            draw_timer_panel(screen, algo_name, diff_name, last_ai_time, move_times, width, height)
                            show_end_screen(screen, "Nerešeno!", move_times, algo_name, diff_name, width, height)
                            return

                        turn = AI
                        draw_board(screen, board, player_color, (ROW_COUNT + 1) * SQUARESIZE)
                        draw_timer_panel(screen, algo_name, diff_name, last_ai_time, move_times, width, height)

        # ── AI potez ──
        if turn == AI and not game_over:
            t_start = time.perf_counter()

            if algo_name == "Minimax":
                col, _ = minimax(board, depth, True)
            else:  # Alpha-Beta Pruning
                col, _ = alphabeta(board, depth, -math.inf, math.inf, True)

            if col is None:
                valid_cols = get_valid_locations(board)
                if valid_cols:
                    col = random.choice(valid_cols)

            t_end = time.perf_counter()
            elapsed = t_end - t_start
            last_ai_time = elapsed
            move_times.append(elapsed)

            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    draw_board(screen, board, player_color, (ROW_COUNT + 1) * SQUARESIZE)
                    draw_timer_panel(screen, algo_name, diff_name, last_ai_time, move_times, width, height)
                    show_end_screen(screen, "AI je pobedio!", move_times, algo_name, diff_name, width, height)
                    return

                if len(get_valid_locations(board)) == 0:
                    draw_board(screen, board, player_color, (ROW_COUNT + 1) * SQUARESIZE)
                    draw_timer_panel(screen, algo_name, diff_name, last_ai_time, move_times, width, height)
                    show_end_screen(screen, "Nerešeno!", move_times, algo_name, diff_name, width, height)
                    return

            turn = PLAYER
            draw_board(screen, board, player_color, (ROW_COUNT + 1) * SQUARESIZE)
            draw_timer_panel(screen, algo_name, diff_name, last_ai_time, move_times, width, height)

        clock.tick(60)

# ─────────────────────────────────────────────
#  Ulazna tačka
# ─────────────────────────────────────────────
def main():
    pygame.init()
    width  = COLUMN_COUNT * SQUARESIZE
    height = 580
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect Four")

    while True:
        screen = pygame.display.set_mode((width, height))
        algo_name, depth, diff_name, player_color, player_goes_first = show_menu(screen)
        run_game(screen, algo_name, depth, diff_name, player_color, player_goes_first)

if __name__ == "__main__":
    main()