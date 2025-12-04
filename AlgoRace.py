"""
ALGO-RACER: Ultimate Pathfinding Visualizer
Set 1: Core Engine, Constants, and Node/Grid Structure
"""

import pygame
import math
import random
from queue import PriorityQueue, Queue
from collections import deque
import time

# --- CONSTANTS & CONFIGURATION ---
WIDTH = 1000
HEIGHT = 800
ROWS = 50  # Results in a 50x50 grid (2500 nodes)
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
    """
    Represents a single cell in the grid.
    Holds state: Start, End, Barrier, Open, Closed, Path.
    """

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbor = []
        self.width = width
        self.total_rows = total_rows
        self.weight = 1  # For weighted algorithms

    def get_pos(self):
        return self.row, self.col

    # --- State Checks ---
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

    # --- State Setters ---
    def reset(self):
        self.color = WHITE
        self.weight = 1

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        # Don't overwrite start/end colors
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
        """
        Populates the self.neighbor list based on adjacent cells.
        Supports switching between 4-way and 8-way (diagonal) movement.
        """
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

        if diagonal:
            # Add diagonal checks here (omitted for brevity in Set 1, can enable later)
            pass

    def __lt__(self, other):
        # Necessary for PriorityQueue comparison
        return False


class GridManager:
    """
    Manages the 2D array of Nodes and handles Grid operations.
    """

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
        # Draw nodes
        for row in self.grid:
            for node in row:
                node.draw(win)
        # Draw lines
        self.draw_grid_lines(win)
        pygame.display.update()

    def get_clicked_pos(self, pos):
        """Translates mouse pixel coordinates to grid row/col"""
        gap = self.width // self.rows
        y, x = pos
        row = y // gap
        col = x // gap
        return row, col

    def clear_path(self):
        """Keeps barriers/start/end but clears the visualization colors"""
        for row in self.grid:
            for node in row:
                if node.is_open() or node.is_closed() or node.color == PURPLE:
                    node.reset()

    def clear_all(self):
        """Resets entire board"""
        self.start_node = None
        self.end_node = None
        self.grid = self.make_grid()

    def update_all_neighbors(self):
        for row in self.grid:
            for node in row:
                node.update_neighbors(self.grid)


"""
ALGO-RACER: Set 2
The Algorithm Engine
Contains: Heuristics, Path Reconstruction, and Core Solvers
"""
import pygame
import math
from queue import PriorityQueue, Queue
from collections import deque


class Heuristics:
    """
    The math brains. Changing these creates different algorithm variants.
    """

    @staticmethod
    def null_h(p1, p2):
        """Returns 0. Turns A* into Dijkstra."""
        return 0

    @staticmethod
    def manhattan(p1, p2):
        """L1 Norm: Great for 4-way grid movement."""
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)

    @staticmethod
    def euclidean(p1, p2):
        """L2 Norm: Shortest straight line (hypotenuse)."""
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def chebyshev(p1, p2):
        """L-Infinity Norm: Best for 8-way movement (King's moves)."""
        x1, y1 = p1
        x2, y2 = p2
        return max(abs(x1 - x2), abs(y1 - y2))

    @staticmethod
    def octile(p1, p2):
        """More precise for 8-way movement with diagonal costs."""
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    # --- Heuristic Registry ---
    # This dictionary helps us count towards our 50 variations
    TYPES = {
        "Dijkstra (Null)": null_h,
        "A* (Manhattan)": manhattan,
        "A* (Euclidean)": euclidean,
        "A* (Chebyshev)": chebyshev,
        "A* (Octile)": octile,
    }


def reconstruct_path(came_from, current, draw_func):
    """Backtracks from End Node to Start Node to draw the final path."""
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw_func()


def algorithm_astar_generic(draw_func, grid, start, end, heuristic_func, is_greedy=False):
    """
    The Master Solver.
    - If heuristic_func is 'null_h', this behaves as DIJKSTRA.
    - If is_greedy is True, this behaves as GREEDY BEST-FIRST SEARCH.
    - Otherwise, it is standard A*.
    """
    count = 0
    open_set = PriorityQueue()
    # (f_score, count, node)
    open_set.put((0, count, start))
    came_from = {}

    # g_score: cost from start to current
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    # f_score: g_score + h_score (heuristic)
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic_func(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        # Allow user to quit mid-algorithm
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()  # Retain color
            return True

        for neighbor in current.neighbor:
            # Assuming distance between neighbors is 1
            temp_g_score = g_score[current] + neighbor.weight

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score

                # Calculate H
                h_score = heuristic_func(neighbor.get_pos(), end.get_pos())

                # Logic Switch: Greedy ignores G, A* uses G+H
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
    """Breadth-First Search: Guaranteed shortest path in unweighted grid."""
    queue = deque([start])
    visited = {start}
    came_from = {}

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

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
    """Depth-First Search: Not shortest path, but fast exploration."""
    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = stack.pop()

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            return True

        # Process neighbors
        # Randomizing neighbor order creates "Random DFS" variant
        neighbors = current.neighbor[:]
        # random.shuffle(neighbors) # Uncomment for Randomized DFS

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
    """
    Bidirectional Search: Meets in the middle.
    Uses two BFS queues running simultaneously.
    """
    start_q = deque([start])
    end_q = deque([end])

    start_visited = {start: None}  # Node: Parent
    end_visited = {end: None}

    while start_q and end_q:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # 1. Expand from Start
        curr_start = start_q.popleft()
        for neighbor in curr_start.neighbor:
            if neighbor not in start_visited:
                start_visited[neighbor] = curr_start
                start_q.append(neighbor)
                neighbor.make_open()

                # Check collision with End frontier
                if neighbor in end_visited:
                    # Path found! Construct both halves
                    construct_bidirectional_path(start_visited, end_visited, neighbor, draw_func)
                    return True

        # 2. Expand from End
        curr_end = end_q.popleft()
        for neighbor in curr_end.neighbor:
            if neighbor not in end_visited:
                end_visited[neighbor] = curr_end
                end_q.append(neighbor)
                # Visual distinction for end-search
                neighbor.color = (100, 255, 100)

                # Check collision with Start frontier
                if neighbor in start_visited:
                    construct_bidirectional_path(start_visited, end_visited, neighbor, draw_func)
                    return True

        draw_func()
        if curr_start != start: curr_start.make_closed()
        if curr_end != end: curr_end.make_closed()

    return False


def construct_bidirectional_path(start_parents, end_parents, meeting_node, draw_func):
    # Trace back to start
    curr = meeting_node
    while curr in start_parents and start_parents[curr] is not None:
        curr = start_parents[curr]
        curr.make_path()
        draw_func()

    # Trace back to end
    curr = meeting_node
    while curr in end_parents and end_parents[curr] is not None:
        curr = end_parents[curr]
        curr.make_path()
        draw_func()


"""
ALGO-RACER: Set 3
The Main Loop & User Interface
"""

# --- Configuration Lists for the "50 Algo" Combinations ---
ALGO_TYPES = ["A* Search", "Greedy Best-First", "Dijkstra", "BFS", "DFS", "Bidirectional"]
HEURISTIC_NAMES = ["Manhattan", "Euclidean", "Chebyshev", "Octile", "Null"]


def draw_ui(win, current_algo, current_heuristic, diagonal, state_text):
    """
    Draws the control panel on the right side of the screen.
    """
    # UI Background
    pygame.draw.rect(win, DARK_GREY, (GRID_WIDTH, 0, UI_WIDTH, HEIGHT))

    # Title
    title = HEADER_FONT.render("ALGO-RACER", True, WHITE)
    win.blit(title, (GRID_WIDTH + 10, 20))

    # Separator
    pygame.draw.line(win, WHITE, (GRID_WIDTH, 50), (WIDTH, 50))

    # Controls Text
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

        # Render Key (Yellow)
        key_surf = FONT.render(f"[{key}]", True, YELLOW)
        win.blit(key_surf, (GRID_WIDTH + 10, y_offset))

        # Render Description (White)
        desc_surf = FONT.render(desc, True, WHITE)
        win.blit(desc_surf, (GRID_WIDTH + 10, y_offset + 20))
        y_offset += 50

    # Status Indicator
    status_color = GREEN if "Finished" in state_text else RED
    if state_text == "Running...": status_color = ORANGE
    if state_text == "Ready": status_color = BLUE

    status_surf = HEADER_FONT.render(f"Status: {state_text}", True, status_color)
    win.blit(status_surf, (GRID_WIDTH + 10, HEIGHT - 80))  # Moved up slightly to make room for name

    # --- CREDIT SECTION ---
    pygame.draw.line(win, GREY, (GRID_WIDTH, HEIGHT - 45), (WIDTH, HEIGHT - 45))

    credit_surf = FONT.render("Created by", True, WHITE)
    win.blit(credit_surf, (GRID_WIDTH + 10, HEIGHT - 30))

    mazy_surf = HEADER_FONT.render("Mazy", True, YELLOW)
    win.blit(mazy_surf, (GRID_WIDTH + 90, HEIGHT - 32))

def generate_random_walls(grid, prob=0.3):
    """Randomly fills grid with walls for quick testing."""
    for row in grid.grid:
        for node in row:
            if not node.is_start() and not node.is_end():
                if random.random() < prob:
                    node.make_barrier()


def main():
    # Initialize Grid
    grid_manager = GridManager(ROWS, GRID_WIDTH)
    grid = grid_manager.grid

    # Default State
    start = None
    end = None
    run = True
    started = False

    # Algorithm State Indices
    algo_idx = 0  # Index for ALGO_TYPES
    heur_idx = 0  # Index for HEURISTIC_NAMES
    diagonal = False  # Toggle 8-way movement

    status_msg = "Ready"

    while run:
        # Drawing Logic
        grid_manager.draw(WIN)

        # Draw UI Overlay
        current_algo_name = ALGO_TYPES[algo_idx]
        current_heur_name = HEURISTIC_NAMES[heur_idx]

        # If Algo is Dijkstra/BFS/DFS, heuristic doesn't apply visually
        if current_algo_name in ["Dijkstra", "BFS", "DFS"]:
            current_heur_name = "N/A"

        draw_ui(WIN, current_algo_name, current_heur_name, diagonal, status_msg)
        pygame.display.update()

        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # --- INPUT HANDLING ---
            if started:
                # Disable input while algorithm is running
                continue

            # Mouse Input (Left Click)
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if pos[0] < GRID_WIDTH:  # Ensure we clicked inside grid
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

            # Mouse Input (Right Click - Erase)
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                if pos[0] < GRID_WIDTH:
                    row, col = grid_manager.get_clicked_pos(pos)
                    node = grid[row][col]
                    node.reset()
                    if node == start:
                        start = None
                    elif node == end:
                        end = None

            # Keyboard Input
            if event.type == pygame.KEYDOWN:

                # START ALGORITHM (SPACE)
                if event.key == pygame.K_SPACE and start and end:
                    started = True
                    status_msg = "Running..."

                    # Update neighbors based on current diagonal setting
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid, diagonal)

                    # Lambda function for drawing during algo execution
                    draw_lambda = lambda: (grid_manager.draw(WIN),
                                           draw_ui(WIN, current_algo_name, current_heur_name, diagonal, "Running..."),
                                           pygame.display.update())

                    # --- ALGORITHM ROUTER ---
                    algo_type = ALGO_TYPES[algo_idx]
                    heuristic_func = Heuristics.TYPES.get(f"A* ({HEURISTIC_NAMES[heur_idx]})", Heuristics.manhattan)

                    found = False

                    if algo_type == "A* Search":
                        # Standard A*
                        found = algorithm_astar_generic(draw_lambda, grid, start, end, heuristic_func, is_greedy=False)

                    elif algo_type == "Greedy Best-First":
                        # Greedy (ignores G cost)
                        found = algorithm_astar_generic(draw_lambda, grid, start, end, heuristic_func, is_greedy=True)

                    elif algo_type == "Dijkstra":
                        # Dijkstra is just A* with Null heuristic
                        found = algorithm_astar_generic(draw_lambda, grid, start, end, Heuristics.null_h,
                                                        is_greedy=False)

                    elif algo_type == "BFS":
                        found = algorithm_bfs(draw_lambda, grid, start, end)

                    elif algo_type == "DFS":
                        found = algorithm_dfs(draw_lambda, grid, start, end)

                    elif algo_type == "Bidirectional":
                        found = algorithm_bidirectional(draw_lambda, grid, start, end)

                    if found:
                        status_msg = "Path Found!"
                    else:
                        status_msg = "No Path!"

                    started = False

                # CLEAR PATH (C) - Keeps walls
                if event.key == pygame.K_c:
                    grid_manager.clear_path()
                    status_msg = "Ready"

                # RESET GRID (R) - Deletes walls
                if event.key == pygame.K_r:
                    grid_manager.clear_all()
                    grid = grid_manager.grid
                    start = None
                    end = None
                    status_msg = "Ready"

                # RANDOM MAZE (M)
                if event.key == pygame.K_m:
                    grid_manager.clear_path()
                    generate_random_walls(grid_manager)

                # TOGGLE ALGORITHM (A)
                if event.key == pygame.K_a:
                    algo_idx = (algo_idx + 1) % len(ALGO_TYPES)

                # TOGGLE HEURISTIC (H)
                if event.key == pygame.K_h:
                    heur_idx = (heur_idx + 1) % len(HEURISTIC_NAMES)

                # TOGGLE DIAGONAL (D)
                if event.key == pygame.K_d:
                    diagonal = not diagonal

    pygame.quit()


if __name__ == "__main__":
    main()