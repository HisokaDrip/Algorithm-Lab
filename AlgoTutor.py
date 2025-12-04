"""
ALGO-TUTOR: The Educational Companion (V2 - Fixed)
Created by Mazy
Features: Fullscreen, Clickable Sidebar, Interactive Math, and Simulations.
"""

import pygame
import math
import random

# --- CONSTANTS & THEME ---
WIDTH, HEIGHT = 1280, 720
BG_COLOR = (30, 30, 30)
SIDEBAR_COLOR = (45, 45, 48)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (0, 122, 204)  # Blue
HIGHLIGHT_COLOR = (255, 215, 0) # Gold
MATH_COLOR = (255, 100, 100)  # Red
WHITE = (255, 255, 255)
GREEN = (50, 205, 50)
PURPLE = (147, 112, 219)

pygame.init()
pygame.display.set_caption("Algo-Tutor: Interactive Algorithm Learning")

# Fonts
def get_fonts():
    return {
        'TITLE': pygame.font.SysFont('consolas', 28, bold=True),
        'HEADER': pygame.font.SysFont('consolas', 20, bold=True),
        'BODY': pygame.font.SysFont('calibri', 20),
        'MATH': pygame.font.SysFont('times new roman', 22, italic=True),
        'SMALL': pygame.font.SysFont('calibri', 14)
    }

FONTS = get_fonts()

# --- CONTENT LIBRARY ---
LESSONS = {
    "intro": {
        "title": "Welcome to Pathfinding",
        "type": "text",
        "content": [
            "What is Pathfinding?",
            "Pathfinding is the computational process of finding the shortest",
            "or most efficient route between two points.",
            "",
            "Why do we need different algorithms?",
            "1. Efficiency: Some are fast but not accurate (Greedy).",
            "2. Accuracy: Some guarantee the shortest path (Dijkstra).",
            "3. Logic: Some use 'Heuristics' (guesses) to speed up math.",
            "",
            "Press 'Next' or click a topic on the left to begin."
        ]
    },
    "bfs": {
        "title": "Breadth-First Search (BFS)",
        "type": "mini_sim",
        "sim_type": "flood",
        "content": [
            "Concept: The Flood Fill",
            "BFS explores neighbors equally in all directions, like water",
            "spilling out of a bucket. It moves layer by layer.",
            "",
            "Data Structure: Queue (FIFO)",
            "First In, First Out.",
            "",
            "Pros: Guarantees shortest path in unweighted grids.",
            "Cons: Slow. It explores EVERYTHING.",
        ]
    },
    "dfs": {
        "title": "Depth-First Search (DFS)",
        "type": "mini_sim",
        "sim_type": "snake",
        "content": [
            "Concept: The Maze Runner",
            "DFS picks one direction and keeps going until it hits a wall.",
            "Then it backtracks.",
            "",
            "Data Structure: Stack (LIFO)",
            "Last In, First Out.",
            "",
            "Pros: Memory efficient. Good for maze generation.",
            "Cons: DOES NOT guarantee shortest path.",
        ]
    },
    "dijkstra": {
        "title": "Dijkstra's Algorithm",
        "type": "mini_sim",
        "sim_type": "dijkstra",
        "content": [
            "The Grandfather of Pathfinding.",
            "Dijkstra accounts for 'Weights' (Terrain Cost).",
            "",
            "It prioritizes exploring cheaper paths first.",
            "Mathematically, it guarantees the shortest path.",
            "",
            "Visual:",
            "Notice how it explores in a circle (like BFS),",
            "but checks every single node carefully."
        ]
    },
    "astar": {
        "title": "A* (A-Star) Search",
        "type": "mini_sim",
        "sim_type": "astar",
        "content": [
            "The Gold Standard.",
            "A* combines Dijkstra (math) with Greedy (guess).",
            "",
            "Formula: F = G + H",
            "G = Cost from Start",
            "H = Heuristic estimate to End",
            "",
            "Visual:",
            "Notice how it 'aims' towards the corner?",
            "It doesn't waste time looking in the wrong direction."
        ]
    },
    "manhattan": {
        "title": "Heuristic: Manhattan",
        "type": "interactive",
        "math_mode": "manhattan",
        "content": [
            "The 'Taxicab' Geometry.",
            "Used when you can only move Up, Down, Left, Right.",
            "Like a taxi driving on city blocks.",
            "",
            "Formula: H = |x1 - x2| + |y1 - y2|",
            "",
            "Interactive:",
            "Drag the nodes on the right.",
            "See the L-shape path?"
        ]
    },
    "euclidean": {
        "title": "Heuristic: Euclidean",
        "type": "interactive",
        "math_mode": "euclidean",
        "content": [
            "The 'As the Bird Flies' Geometry.",
            "Used when you can move in any direction.",
            "",
            "Formula: H = sqrt((x1-x2)^2 + (y1-y2)^2)",
            "",
            "Interactive:",
            "This creates a straight line (hypotenuse).",
        ]
    },
    "chebyshev": {
        "title": "Heuristic: Chebyshev",
        "type": "interactive",
        "math_mode": "chebyshev",
        "content": [
            "The 'Chess King' Geometry.",
            "Used when 8-way movement is allowed.",
            "",
            "Formula: H = max(|x1-x2|, |y1-y2|)",
            "",
            "The cost is determined by the longest side."
        ]
    }
}

# --- COMPONENTS ---

class TextRenderer:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.scroll_y = 0
        self.lines = []

    def set_content(self, raw_lines):
        self.lines = raw_lines
        self.scroll_y = 0

    def resize(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, win):
        win.set_clip(self.rect)
        current_y = self.rect.y - self.scroll_y

        for line in self.lines:
            color = TEXT_COLOR
            font = FONTS['BODY']

            if "Formula:" in line or "=" in line:
                color = MATH_COLOR
                font = FONTS['MATH']
            elif "Concept:" in line or "Data Structure:" in line or "Visual:" in line or "Interactive:" in line:
                color = HIGHLIGHT_COLOR
                font = FONTS['HEADER']

            text_surf = font.render(line, True, color)

            if current_y + text_surf.get_height() > self.rect.y and current_y < self.rect.bottom:
                win.blit(text_surf, (self.rect.x, current_y))

            current_y += 30 # Line height

        win.set_clip(None)

    def scroll(self, direction):
        self.scroll_y += direction * 20
        self.scroll_y = max(0, self.scroll_y)

class InteractiveSandbox:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.p1 = [x + 100, y + 100]
        self.p2 = [x + 300, y + 200]
        self.dragging = None
        self.mode = "manhattan"

    def resize(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.p1 = [x + 50, y + 50]
        self.p2 = [x + w - 50, y + h - 50]

    def handle_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if math.hypot(mx - self.p1[0], my - self.p1[1]) < 20: self.dragging = 1
            elif math.hypot(mx - self.p2[0], my - self.p2[1]) < 20: self.dragging = 2
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = None
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx, my = pygame.mouse.get_pos()
            mx = max(self.rect.left + 20, min(self.rect.right - 20, mx))
            my = max(self.rect.top + 20, min(self.rect.bottom - 20, my))
            if self.dragging == 1: self.p1 = [mx, my]
            else: self.p2 = [mx, my]

    def set_mode(self, mode):
        self.mode = mode

    def draw(self, win):
        pygame.draw.rect(win, (20, 20, 25), self.rect)
        pygame.draw.rect(win, (100, 100, 100), self.rect, 2)

        x1, y1 = self.p1
        x2, y2 = self.p2
        result = 0

        if self.mode == "manhattan":
            pygame.draw.line(win, MATH_COLOR, (x1, y1), (x2, y1), 3)
            pygame.draw.line(win, MATH_COLOR, (x2, y1), (x2, y2), 3)
            result = abs(x1 - x2) + abs(y1 - y2)
        elif self.mode == "euclidean":
            pygame.draw.line(win, MATH_COLOR, (x1, y1), (x2, y2), 3)
            result = int(math.hypot(x1 - x2, y1 - y2))
        elif self.mode == "chebyshev":
            w, h = abs(x1-x2), abs(y1-y2)
            pygame.draw.rect(win, (50,50,50), (min(x1,x2), min(y1,y2), w, h), 1)
            pygame.draw.line(win, MATH_COLOR, (x1,y1), (x2,y2), 1)
            result = max(w, h)

        pygame.draw.circle(win, ACCENT_COLOR, (int(x1), int(y1)), 12)
        pygame.draw.circle(win, HIGHLIGHT_COLOR, (int(x2), int(y2)), 12)

        txt = FONTS['TITLE'].render(f"Result: {int(result)}", True, MATH_COLOR)
        win.blit(txt, (self.rect.x + 20, self.rect.bottom - 40))

        hint = FONTS['BODY'].render("Drag the dots!", True, (150, 150, 150))
        win.blit(hint, (self.rect.x + 20, self.rect.y + 10))

class MiniSimulation:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.grid_size = 10
        self.sim_type = "flood"
        self.timer = 0
        self.step = 0

    def resize(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def update(self):
        self.timer += 1
        if self.timer > 8:
            self.step += 1
            self.timer = 0
            if self.step > 20: self.step = 0

    def set_mode(self, mode):
        self.sim_type = mode
        self.step = 0

    def draw(self, win):
        cell_w = self.rect.width // self.grid_size
        cell_h = self.rect.height // self.grid_size

        start_row, start_col = 0, 0
        end_row, end_col = self.grid_size-1, self.grid_size-1

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = self.rect.x + c * cell_w
                y = self.rect.y + r * cell_h

                color = (40, 40, 45)

                if self.sim_type == "flood":
                    dist = abs(r - start_row) + abs(c - start_col)
                    if dist <= self.step: color = ACCENT_COLOR

                elif self.sim_type == "snake":
                    idx = r * self.grid_size + c
                    if idx <= self.step * 3: color = PURPLE

                elif self.sim_type == "dijkstra":
                    dist = math.sqrt((r-start_row)**2 + (c-start_col)**2)
                    if dist <= self.step * 0.8: color = GREEN

                elif self.sim_type == "astar":
                    dist_start = abs(r - start_row) + abs(c - start_col)
                    if dist_start <= self.step:
                         if abs(r - c) < 3: color = HIGHLIGHT_COLOR
                         else: color = (100, 100, 50)

                pygame.draw.rect(win, color, (x, y, cell_w-2, cell_h-2))

        caption = FONTS['HEADER'].render(f"Simulating: {self.sim_type.upper()}", True, WHITE)
        win.blit(caption, (self.rect.x, self.rect.y - 30))

class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.hover = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.hover: self.callback()

    def draw(self, win):
        color = ACCENT_COLOR if self.hover else (60, 60, 65)
        pygame.draw.rect(win, color, self.rect, border_radius=5)
        text = FONTS['HEADER'].render(self.text, True, WHITE)
        win.blit(text, (self.rect.centerx - text.get_width()//2, self.rect.centery - text.get_height()//2))

class TutorApp:
    def __init__(self):
        self.run = True
        self.clock = pygame.time.Clock()
        self.w, self.h = WIDTH, HEIGHT

        # --- FIX: Define self.win here instead of using global WIN ---
        self.win = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)

        self.lesson_keys = list(LESSONS.keys())
        self.current_index = 0

        self.text_panel = TextRenderer(320, 100, 800, 300)
        self.sandbox = InteractiveSandbox(320, 420, 800, 250)
        self.mini_sim = MiniSimulation(320, 420, 800, 250)

        self.btn_prev = Button(20, 600, 120, 40, "< Prev", self.prev_lesson)
        self.btn_next = Button(160, 600, 120, 40, "Next >", self.next_lesson)

        self.sidebar_click_rects = []
        self.load_lesson()

    def resize_layout(self):
        sb_w = 300
        content_w = self.w - sb_w - 40

        self.text_panel.resize(sb_w + 20, 100, content_w, self.h * 0.4)

        sim_y = self.h * 0.55
        sim_h = self.h * 0.40
        self.sandbox.resize(sb_w + 20, sim_y, content_w, sim_h)
        self.mini_sim.resize(sb_w + 20, sim_y, content_w, sim_h)

        self.btn_prev.rect.topleft = (20, self.h - 120)
        self.btn_next.rect.topleft = (160, self.h - 120)

    def next_lesson(self):
        if self.current_index < len(self.lesson_keys) - 1:
            self.current_index += 1
            self.load_lesson()

    def prev_lesson(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_lesson()

    def jump_to_lesson(self, index):
        self.current_index = index
        self.load_lesson()

    def load_lesson(self):
        key = self.lesson_keys[self.current_index]
        data = LESSONS[key]
        self.text_panel.set_content(data["content"])

        if data["type"] == "interactive":
            self.sandbox.set_mode(data["math_mode"])
        elif data["type"] == "mini_sim":
            self.mini_sim.set_mode(data["sim_type"])

    def draw_sidebar(self, win):
        sb_w = 300
        pygame.draw.rect(win, SIDEBAR_COLOR, (0, 0, sb_w, self.h))
        pygame.draw.line(win, (60,60,60), (sb_w, 0), (sb_w, self.h))

        title = FONTS['TITLE'].render("Algo-Tutor", True, HIGHLIGHT_COLOR)
        win.blit(title, (20, 20))

        y = 80
        self.sidebar_click_rects = []

        for i, key in enumerate(self.lesson_keys):
            color = WHITE if i == self.current_index else (150, 150, 150)
            prefix = "> " if i == self.current_index else "  "
            label = FONTS['HEADER'].render(f"{prefix}{LESSONS[key]['title']}", True, color)

            if label.get_width() > 280:
                label = pygame.transform.scale(label, (280, 20))

            rect = win.blit(label, (10, y))
            self.sidebar_click_rects.append((rect, i))

            y += 40

        self.btn_prev.draw(win)
        self.btn_next.draw(win)

        pygame.draw.line(win, (60, 60, 60), (10, self.h - 60), (290, self.h - 60))
        by_text = FONTS['SMALL'].render("App by", True, (150, 150, 150))
        win.blit(by_text, (20, self.h - 45))
        name_text = FONTS['HEADER'].render("Mazy", True, HIGHLIGHT_COLOR)
        win.blit(name_text, (70, self.h - 48))

    def main_loop(self):
        self.resize_layout()

        while self.run:
            self.clock.tick(60)
            # --- FIX: Use self.win ---
            self.win.fill(BG_COLOR)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        is_full = self.win.get_flags() & pygame.FULLSCREEN
                        if is_full:
                            self.win = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
                        else:
                            self.win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

                        # Update size refs
                        self.w, self.h = self.win.get_size()
                        self.resize_layout()

                if event.type == pygame.VIDEORESIZE:
                    if not (self.win.get_flags() & pygame.FULLSCREEN):
                        self.w, self.h = event.w, event.h
                        self.win = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
                        self.resize_layout()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.pos[0] < 300:
                        for rect, index in self.sidebar_click_rects:
                            if rect.collidepoint(event.pos):
                                self.jump_to_lesson(index)

                self.btn_next.handle_event(event)
                self.btn_prev.handle_event(event)

                key = self.lesson_keys[self.current_index]
                if LESSONS[key]["type"] == "interactive":
                    self.sandbox.handle_input(event)
                if event.type == pygame.MOUSEWHEEL:
                    self.text_panel.scroll(event.y * -1)

            # --- FIX: Draw to self.win ---
            self.draw_sidebar(self.win)

            key = self.lesson_keys[self.current_index]
            header = FONTS['TITLE'].render(LESSONS[key]["title"], True, ACCENT_COLOR)
            self.win.blit(header, (320, 30))
            pygame.draw.line(self.win, (80, 80, 80), (320, 70), (self.w - 20, 70))

            self.text_panel.draw(self.win)

            if LESSONS[key]["type"] == "interactive":
                self.sandbox.draw(self.win)
            elif LESSONS[key]["type"] == "mini_sim":
                self.mini_sim.update()
                self.mini_sim.draw(self.win)

            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    app = TutorApp()
    app.main_loop()