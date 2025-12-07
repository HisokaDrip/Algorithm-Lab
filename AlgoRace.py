"""
SUII
"""

import pygame
import math
import random
import sys  # Added for safe exiting
from queue import PriorityQueue
from collections import deque

# --- CONSTANTS & CONFIGURATION ---
WIDTH = 1000
HEIGHT = 800
ROWS = 50  # Results in a 50x50 grid
GRID_WIDTH = 800  # The grid takes up the left side
UI_WIDTH = 200  # UI panel on the right
WIN_TITLE = "Algo-Racer: 50+ Algorithm Visualizer"

# Colors (R, G, B)
RED = (255, 65, 54)
GREEN = (46, 204, 64)
BLUE = (0, 116, 217)
YELLOW = (255, 220, 0)
WHITE = (255, 255, 255)
BLACK = (17, 17, 17)
PURPLE = (177, 13, 201)
ORANGE = (255, 133, 27)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
DARK_GREY = (50, 50, 50)

# Pygame Setup
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(WIN_TITLE)
FONT = pygame.font.SysFont('arial', 16)
HEADER_FONT = pygame.font.SysFont('arial', 20, bold=True)


class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbor = []
        self.width = width
        self.total_rows = total_rows
        self.weight = 1

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE
        self.weight = 1

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        if not self.is_start() and not self.is_end():
            self.color = RED

    def make_open(self):
        if not self.is_start() and not self.is_end():
            self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        if not self.is_start() and not self.is_end():
            self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid, diagonal=False):
        self.neighbor = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbor.append(grid[self.row + 1][self.col])
        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbor.append(grid[self.row - 1][self.col])
        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbor.append(grid[self.row][self.col + 1])
        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbor.append(grid[self.row][self.col - 1])

        # Simple implementation for Diagonal (Set 2 refinement)
        if diagonal:
            # Down-Right
            if self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and not grid[self.row + 1][
                self.col + 1].is_barrier():
                self.neighbor.append(grid[self.row + 1][self.col + 1])
            # Up-Right
            if self.row > 0 and self.col < self.total_rows - 1 and not grid[self.row - 1][self.col + 1].is_barrier():
                self.neighbor.append(grid[self.row - 1][self.col + 1])
            # Down-Left
            if self.row < self.total_rows - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_barrier():
                self.neighbor.append(grid[self.row + 1][self.col - 1])
            # Up-Left
            if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_barrier():
                self.neighbor.append(grid[self.row - 1][self.col - 1])

    def __lt__(self, other):
        return False


class GridManager:
    def __init__(self, rows, width):
        self.rows = rows
        self.width = width
        self.grid = self.make_grid()
        self.start_node = None
        self.end_node = None

    def make_grid(self):
        grid = []
        gap = self.width // self.rows
        for i in range(self.rows):
            grid.append([])
            for j in range(self.rows):
                node = Node(i, j, gap, self.rows)
                grid[i].append(node)
        return grid

    def draw_grid_lines(self, win):
        gap = self.width // self.rows
        for i in range(self.rows):
            pygame.draw.line(win, GREY, (0, i * gap), (self.width, i * gap))
            for j in range(self.rows):
                pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, self.width))

    def draw(self, win):
        for row in self.grid:
            for node in row:
                node.draw(win)
        self.draw_grid_lines(win)
        pygame.display.update()

    def get_clicked_pos(self, pos):
        gap = self.width // self.rows
        y, x = pos
        row = y // gap
        col = x // gap
        return row, col

    def clear_path(self):
        for row in self.grid:
            for node in row:
                if node.is_open() or node.is_closed() or node.color == PURPLE:
                    node.reset()

    def clear_all(self):
        self.start_node = None
        self.end_node = None
        self.grid = self.make_grid()


# --- ALGORITHM ENGINE ---

class Heuristics:
    @staticmethod
    def null_h(p1, p2):
        return 0

    @staticmethod
    def manhattan(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)

    @staticmethod
    def euclidean(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def chebyshev(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return max(abs(x1 - x2), abs(y1 - y2))

    @staticmethod
    def octile(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    TYPES = {
        "Dijkstra (Null)": null_h,
        "A* (Manhattan)": manhattan,
        "A* (Euclidean)": euclidean,
        "A* (Chebyshev)": chebyshev,
        "A* (Octile)": octile,
    }


def reconstruct_path(came_from, current, draw_func):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw_func()


def algorithm_astar_generic(draw_func, grid, start, end, heuristic_func, is_greedy=False):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic_func(start.get_pos(), end.get_pos())
    open_set_hash = {start}

    while not open_set.empty():
        # --- FIX: Safe Exit ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()  # Stop script immediately
        # ----------------------

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            return True

        for neighbor in current.neighbor:
            temp_g_score = g_score[current] + neighbor.weight

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                h_score = heuristic_func(neighbor.get_pos(), end.get_pos())

                if is_greedy:
                    f_score[neighbor] = h_score
                else:
                    f_score[neighbor] = temp_g_score + h_score

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw_func()

        if current != start:
            current.make_closed()

    return False


def algorithm_bfs(draw_func, grid, start, end):
    queue = deque([start])
    visited = {start}
    came_from = {}

    while queue:
        # --- FIX: Safe Exit ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # ----------------------

        current = queue.popleft()

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            return True

        for neighbor in current.neighbor:
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()

        draw_func()
        if current != start:
            current.make_closed()

    return False


def algorithm_dfs(draw_func, grid, start, end):
    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        # --- FIX: Safe Exit ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # ----------------------

        current = stack.pop()

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            return True

        neighbors = current.neighbor[:]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()

        draw_func()
        if current != start:
            current.make_closed()

    return False


def algorithm_bidirectional(draw_func, grid, start, end):
    start_q = deque([start])
    end_q = deque([end])
    start_visited = {start: None}
    end_visited = {end: None}

    while start_q and end_q:
        # --- FIX: Safe Exit ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # ----------------------

        # 1. Expand from Start
        if start_q:
            curr_start = start_q.popleft()
            for neighbor in curr_start.neighbor:
                if neighbor not in start_visited:
                    start_visited[neighbor] = curr_start
                    start_q.append(neighbor)
                    neighbor.make_open()
                    if neighbor in end_visited:
                        construct_bidirectional_path(start_visited, end_visited, neighbor, draw_func)
                        return True

        # 2. Expand from End
        if end_q:
            curr_end = end_q.popleft()
            for neighbor in curr_end.neighbor:
                if neighbor not in end_visited:
                    end_visited[neighbor] = curr_end
                    end_q.append(neighbor)
                    neighbor.color = (100, 255, 100)  # Distinct color for backwards search
                    if neighbor in start_visited:
                        construct_bidirectional_path(start_visited, end_visited, neighbor, draw_func)
                        return True

        draw_func()
        if curr_start != start: curr_start.make_closed()
        if curr_end != end: curr_end.make_closed()

    return False


def construct_bidirectional_path(start_parents, end_parents, meeting_node, draw_func):
    curr = meeting_node
    while curr in start_parents and start_parents[curr] is not None:
        curr = start_parents[curr]
        curr.make_path()
        draw_func()
    curr = meeting_node
    while curr in end_parents and end_parents[curr] is not None:
        curr = end_parents[curr]
        curr.make_path()
        draw_func()


# --- UI & MAIN LOOP ---

ALGO_TYPES = ["A* Search", "Greedy Best-First", "Dijkstra", "BFS", "DFS", "Bidirectional"]
HEURISTIC_NAMES = ["Manhattan", "Euclidean", "Chebyshev", "Octile", "Null"]


def draw_ui(win, current_algo, current_heuristic, diagonal, state_text):
    pygame.draw.rect(win, DARK_GREY, (GRID_WIDTH, 0, UI_WIDTH, HEIGHT))

    # Title
    title = HEADER_FONT.render("ALGO-RACER", True, WHITE)
    win.blit(title, (GRID_WIDTH + 10, 20))
    pygame.draw.line(win, WHITE, (GRID_WIDTH, 50), (WIDTH, 50))

    controls = [
        ("Mouse Left", "Draw Wall/Start/End"),
        ("Mouse Right", "Erase"),
        ("SPACE", "Start Race"),
        ("C", "Clear Path"),
        ("R", "Reset Grid"),
        ("M", "Random Maze"),
        ("----------------", ""),
        ("A", f"Algo: {current_algo}"),
        ("H", f"Heur: {current_heuristic}"),
        ("D", f"Diag: {'ON' if diagonal else 'OFF'}"),
    ]

    y_offset = 70
    for key, desc in controls:
        if key == "----------------":
            pygame.draw.line(win, GREY, (GRID_WIDTH, y_offset + 10), (WIDTH, y_offset + 10))
            y_offset += 20
            continue
        key_surf = FONT.render(f"[{key}]", True, YELLOW)
        win.blit(key_surf, (GRID_WIDTH + 10, y_offset))
        desc_surf = FONT.render(desc, True, WHITE)
        win.blit(desc_surf, (GRID_WIDTH + 10, y_offset + 20))
        y_offset += 50

    status_color = GREEN if "Finished" in state_text or "Found" in state_text else RED
    if state_text == "Running...": status_color = ORANGE
    if state_text == "Ready": status_color = BLUE

    status_surf = HEADER_FONT.render(f"Status: {state_text}", True, status_color)
    win.blit(status_surf, (GRID_WIDTH + 10, HEIGHT - 80))

    pygame.draw.line(win, GREY, (GRID_WIDTH, HEIGHT - 45), (WIDTH, HEIGHT - 45))
    credit_surf = FONT.render("Created by", True, WHITE)
    win.blit(credit_surf, (GRID_WIDTH + 10, HEIGHT - 30))
    mazy_surf = HEADER_FONT.render("Mazy", True, YELLOW)
    win.blit(mazy_surf, (GRID_WIDTH + 90, HEIGHT - 32))


def generate_random_walls(grid_manager, prob=0.3):
    grid_manager.clear_path()
    for row in grid_manager.grid:
        for node in row:
            if not node.is_start() and not node.is_end():
                if random.random() < prob:
                    node.make_barrier()


def main():
    grid_manager = GridManager(ROWS, GRID_WIDTH)
    grid = grid_manager.grid

    start = None
    end = None
    run = True
    started = False

    algo_idx = 0
    heur_idx = 0
    diagonal = False
    status_msg = "Ready"

    while run:
        grid_manager.draw(WIN)
        current_algo_name = ALGO_TYPES[algo_idx]
        current_heur_name = HEURISTIC_NAMES[heur_idx]
        if current_algo_name in ["Dijkstra", "BFS", "DFS"]:
            current_heur_name = "N/A"

        draw_ui(WIN, current_algo_name, current_heur_name, diagonal, status_msg)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:  # Left
                pos = pygame.mouse.get_pos()
                if pos[0] < GRID_WIDTH:
                    row, col = grid_manager.get_clicked_pos(pos)
                    node = grid[row][col]
                    if not start and node != end:
                        start = node
                        start.make_start()
                        grid_manager.start_node = start
                    elif not end and node != start:
                        end = node
                        end.make_end()
                        grid_manager.end_node = end
                    elif node != end and node != start:
                        node.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # Right
                pos = pygame.mouse.get_pos()
                if pos[0] < GRID_WIDTH:
                    row, col = grid_manager.get_clicked_pos(pos)
                    node = grid[row][col]
                    node.reset()
                    if node == start:
                        start = None
                    elif node == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    started = True
                    status_msg = "Running..."
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid, diagonal)

                    draw_lambda = lambda: (grid_manager.draw(WIN),
                                           draw_ui(WIN, current_algo_name, current_heur_name, diagonal, "Running..."),
                                           pygame.display.update())

                    algo_type = ALGO_TYPES[algo_idx]
                    heuristic_func = Heuristics.TYPES.get(f"A* ({HEURISTIC_NAMES[heur_idx]})", Heuristics.manhattan)

                    # Map heuristic choice specifically for Generic call
                    h_choice = HEURISTIC_NAMES[heur_idx]
                    if h_choice == "Manhattan":
                        heuristic_func = Heuristics.manhattan
                    elif h_choice == "Euclidean":
                        heuristic_func = Heuristics.euclidean
                    elif h_choice == "Chebyshev":
                        heuristic_func = Heuristics.chebyshev
                    elif h_choice == "Octile":
                        heuristic_func = Heuristics.octile
                    else:
                        heuristic_func = Heuristics.null_h

                    found = False
                    if algo_type == "A* Search":
                        found = algorithm_astar_generic(draw_lambda, grid, start, end, heuristic_func, is_greedy=False)
                    elif algo_type == "Greedy Best-First":
                        found = algorithm_astar_generic(draw_lambda, grid, start, end, heuristic_func, is_greedy=True)
                    elif algo_type == "Dijkstra":
                        found = algorithm_astar_generic(draw_lambda, grid, start, end, Heuristics.null_h,
                                                        is_greedy=False)
                    elif algo_type == "BFS":
                        found = algorithm_bfs(draw_lambda, grid, start, end)
                    elif algo_type == "DFS":
                        found = algorithm_dfs(draw_lambda, grid, start, end)
                    elif algo_type == "Bidirectional":
                        found = algorithm_bidirectional(draw_lambda, grid, start, end)

                    status_msg = "Path Found!" if found else "No Path!"
                    started = False

                if event.key == pygame.K_c:
                    grid_manager.clear_path()
                    status_msg = "Ready"

                if event.key == pygame.K_r:
                    grid_manager.clear_all()
                    grid = grid_manager.grid
                    start = None
                    end = None
                    status_msg = "Ready"

                if event.key == pygame.K_m:
                    generate_random_walls(grid_manager)

                if event.key == pygame.K_a:
                    algo_idx = (algo_idx + 1) % len(ALGO_TYPES)

                if event.key == pygame.K_h:
                    heur_idx = (heur_idx + 1) % len(HEURISTIC_NAMES)

                if event.key == pygame.K_d:
                    diagonal = not diagonal

    pygame.quit()
    sys.exit()


if __name__ == "__main__":

    main()
