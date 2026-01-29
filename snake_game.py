#!/usr/bin/env python3
"""
Advanced Graphics Snake Game
A visually stunning snake game with modern graphics effects

Features:
- Smooth snake movement with interpolation
- Particle effects when eating food
- Glowing snake head and body gradient
- Animated pulsing food
- Floating background stars
- Screen shake on death
- Trail effects behind snake
- Wave distortion effects

Run with: python snake_game.py
For headless testing: SDL_VIDEODRIVER=dummy python snake_game.py --headless
"""

import os
import pygame
import random
import math
import sys
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Check for headless mode
HEADLESS = '--headless' in sys.argv or os.environ.get('SDL_VIDEODRIVER') == 'dummy'

# Set SDL driver for headless mode before pygame init
if HEADLESS:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Initialize Pygame
pygame.init()
if not HEADLESS:
    pygame.mixer.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE
FPS = 60
SNAKE_SPEED = 10  # Moves per second

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BG = (10, 12, 20)

# Snake gradient colors (neon green theme)
SNAKE_HEAD_COLOR = (80, 255, 120)
SNAKE_HEAD_GLOW = (50, 200, 100)
SNAKE_BODY_START = (60, 220, 100)
SNAKE_BODY_END = (30, 120, 60)

# Food colors (neon red/pink)
FOOD_COLOR = (255, 70, 100)
FOOD_GLOW = (255, 120, 150)
FOOD_INNER = (255, 200, 220)

# Special food colors
BONUS_FOOD_COLOR = (255, 200, 50)
BONUS_FOOD_GLOW = (255, 230, 100)

# UI Colors
SCORE_COLOR = (180, 180, 220)
GAME_OVER_COLOR = (255, 80, 100)
TITLE_COLOR = (100, 200, 255)


@dataclass
class Particle:
    """Particle for visual effects"""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    size: float
    particle_type: str = "circle"  # circle, spark, trail

    def update(self, dt: float) -> bool:
        """Update particle, return False if dead"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

        # Different physics for different types
        if self.particle_type == "spark":
            self.vy += 300 * dt  # Gravity
            self.vx *= 0.95
        elif self.particle_type == "trail":
            self.vx *= 0.9
            self.vy *= 0.9
        else:
            self.vx *= 0.98
            self.vy *= 0.98

        return self.life > 0

    def draw(self, surface: pygame.Surface):
        """Draw particle with fade effect"""
        alpha = self.life / self.max_life
        size = max(1, int(self.size * alpha))

        if size > 0:
            if self.particle_type == "spark":
                # Draw line spark
                color = tuple(int(c * alpha) for c in self.color)
                end_x = int(self.x - self.vx * 0.02)
                end_y = int(self.y - self.vy * 0.02)
                pygame.draw.line(surface, color, (int(self.x), int(self.y)), (end_x, end_y), max(1, size // 2))
            else:
                color = tuple(int(c * alpha) for c in self.color)
                pygame.draw.circle(surface, color, (int(self.x), int(self.y)), size)


class ParticleSystem:
    """Manages particle effects"""

    def __init__(self):
        self.particles: List[Particle] = []

    def emit(self, x: float, y: float, count: int, color: Tuple[int, int, int],
             particle_type: str = "circle", speed_range: Tuple[float, float] = (50, 200)):
        """Emit particles at position"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(speed_range[0], speed_range[1])

            # Vary color slightly
            varied_color = tuple(
                max(0, min(255, c + random.randint(-20, 20))) for c in color
            )

            self.particles.append(Particle(
                x=x + random.uniform(-5, 5),
                y=y + random.uniform(-5, 5),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=random.uniform(0.3, 1.0),
                max_life=1.0,
                color=varied_color,
                size=random.uniform(2, 8),
                particle_type=particle_type
            ))

    def emit_sparks(self, x: float, y: float, count: int, color: Tuple[int, int, int]):
        """Emit spark particles (for special effects)"""
        self.emit(x, y, count, color, "spark", (100, 400))

    def emit_trail(self, x: float, y: float, color: Tuple[int, int, int]):
        """Emit trail particle"""
        self.particles.append(Particle(
            x=x + random.uniform(-2, 2),
            y=y + random.uniform(-2, 2),
            vx=random.uniform(-10, 10),
            vy=random.uniform(-10, 10),
            life=0.5,
            max_life=0.5,
            color=color,
            size=random.uniform(3, 6),
            particle_type="trail"
        ))

    def update(self, dt: float):
        """Update all particles"""
        self.particles = [p for p in self.particles if p.update(dt)]

    def draw(self, surface: pygame.Surface):
        """Draw all particles"""
        for particle in self.particles:
            particle.draw(surface)


class ScreenShake:
    """Screen shake effect manager"""

    def __init__(self):
        self.shake_amount = 0
        self.shake_decay = 10

    def shake(self, amount: float):
        """Trigger screen shake"""
        self.shake_amount = amount

    def update(self, dt: float):
        """Update shake"""
        if self.shake_amount > 0:
            self.shake_amount = max(0, self.shake_amount - self.shake_decay * dt)

    def get_offset(self) -> Tuple[int, int]:
        """Get current shake offset"""
        if self.shake_amount > 0.5:
            return (
                random.randint(int(-self.shake_amount), int(self.shake_amount)),
                random.randint(int(-self.shake_amount), int(self.shake_amount))
            )
        return (0, 0)


class BackgroundEffect:
    """Animated background with floating particles and grid"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Stars
        self.stars = []
        for _ in range(80):
            self.stars.append({
                'x': random.randint(0, width),
                'y': random.randint(0, height),
                'speed': random.uniform(5, 25),
                'size': random.uniform(0.5, 2.5),
                'brightness': random.uniform(0.2, 1.0),
                'twinkle_speed': random.uniform(1, 4)
            })

        # Nebula blobs (background color variations)
        self.nebulas = []
        for _ in range(5):
            self.nebulas.append({
                'x': random.randint(0, width),
                'y': random.randint(0, height),
                'size': random.randint(100, 200),
                'color': random.choice([
                    (30, 20, 50),
                    (20, 30, 50),
                    (40, 20, 40),
                    (25, 35, 45)
                ]),
                'alpha': random.randint(20, 40)
            })

        self.grid_alpha = 25
        self.time = 0

    def update(self, dt: float):
        """Update background animation"""
        self.time += dt
        for star in self.stars:
            star['y'] += star['speed'] * dt
            if star['y'] > self.height:
                star['y'] = 0
                star['x'] = random.randint(0, self.width)

    def draw(self, surface: pygame.Surface):
        """Draw animated background"""
        # Fill with dark background
        surface.fill(DARK_BG)

        # Draw nebula blobs
        for nebula in self.nebulas:
            nebula_surface = pygame.Surface((nebula['size'] * 2, nebula['size'] * 2), pygame.SRCALPHA)
            for i in range(3):
                size = nebula['size'] - i * 30
                if size > 0:
                    alpha = nebula['alpha'] - i * 10
                    pygame.draw.circle(
                        nebula_surface,
                        (*nebula['color'], max(0, alpha)),
                        (nebula['size'], nebula['size']),
                        size
                    )
            surface.blit(nebula_surface, (nebula['x'] - nebula['size'], nebula['y'] - nebula['size']))

        # Draw subtle grid
        grid_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        grid_color = (40, 50, 70, self.grid_alpha)

        for x in range(0, self.width, GRID_SIZE):
            pygame.draw.line(grid_surface, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, GRID_SIZE):
            pygame.draw.line(grid_surface, grid_color, (0, y), (self.width, y))

        surface.blit(grid_surface, (0, 0))

        # Draw floating stars with twinkling
        for star in self.stars:
            twinkle = 0.6 + 0.4 * math.sin(self.time * star['twinkle_speed'] + star['x'] * 0.1)
            brightness = star['brightness'] * twinkle
            gray = int(150 * brightness)

            # Some stars have slight color tint
            if star['brightness'] > 0.7:
                color = (gray, gray, int(gray * 1.1))  # Slight blue tint
            else:
                color = (gray, gray, gray)

            pos = (int(star['x']), int(star['y']))
            size = max(1, int(star['size'] * (0.8 + 0.2 * twinkle)))
            pygame.draw.circle(surface, color, pos, size)


class Snake:
    """Snake with smooth movement and visual effects"""

    def __init__(self, start_pos: Tuple[int, int]):
        self.segments: deque = deque([start_pos])
        self.direction = (1, 0)
        self.next_direction = (1, 0)
        self.grow_pending = 0
        self.move_timer = 0
        self.move_interval = 1.0 / SNAKE_SPEED

        # Visual interpolation
        self.interpolation = 0.0

        # Trail history for smooth rendering
        self.position_history: List[Tuple[float, float]] = []

    def set_direction(self, direction: Tuple[int, int]):
        """Set next direction (prevents 180 degree turns)"""
        if (direction[0] != -self.direction[0] or direction[1] != -self.direction[1]):
            self.next_direction = direction

    def update(self, dt: float) -> bool:
        """Update snake, return True if moved"""
        self.move_timer += dt
        self.interpolation = min(1.0, self.move_timer / self.move_interval)

        if self.move_timer >= self.move_interval:
            self.move_timer = 0
            self.interpolation = 0
            self.direction = self.next_direction

            # Calculate new head position
            head = self.segments[0]
            new_head = (
                (head[0] + self.direction[0]) % GRID_WIDTH,
                (head[1] + self.direction[1]) % GRID_HEIGHT
            )

            self.segments.appendleft(new_head)

            # Save position for trail
            hx = new_head[0] * GRID_SIZE + GRID_SIZE // 2
            hy = new_head[1] * GRID_SIZE + GRID_SIZE // 2
            self.position_history.append((hx, hy))
            if len(self.position_history) > 50:
                self.position_history.pop(0)

            if self.grow_pending > 0:
                self.grow_pending -= 1
            else:
                self.segments.pop()

            return True
        return False

    def grow(self, amount: int = 1):
        """Schedule growth"""
        self.grow_pending += amount

    def check_self_collision(self) -> bool:
        """Check if snake collided with itself"""
        head = self.segments[0]
        return head in list(self.segments)[1:]

    def get_head_pos(self) -> Tuple[int, int]:
        """Get current head grid position"""
        return self.segments[0]

    def get_interpolated_head_pos(self) -> Tuple[float, float]:
        """Get interpolated head screen position"""
        head = self.segments[0]
        x = head[0] * GRID_SIZE + GRID_SIZE // 2
        y = head[1] * GRID_SIZE + GRID_SIZE // 2

        prev_x = x - self.direction[0] * GRID_SIZE
        prev_y = y - self.direction[1] * GRID_SIZE

        ix = prev_x + (x - prev_x) * self.interpolation
        iy = prev_y + (y - prev_y) * self.interpolation

        return (ix, iy)

    def draw(self, surface: pygame.Surface, time: float):
        """Draw snake with gradient and glow effects"""
        segments_list = list(self.segments)
        num_segments = len(segments_list)

        # Draw body segments from tail to head
        for i in range(num_segments - 1, -1, -1):
            segment = segments_list[i]

            # Calculate gradient color
            t = i / max(1, num_segments - 1)
            body_color = (
                int(SNAKE_BODY_START[0] + (SNAKE_BODY_END[0] - SNAKE_BODY_START[0]) * t),
                int(SNAKE_BODY_START[1] + (SNAKE_BODY_END[1] - SNAKE_BODY_START[1]) * t),
                int(SNAKE_BODY_START[2] + (SNAKE_BODY_END[2] - SNAKE_BODY_START[2]) * t),
            )

            # Calculate screen position
            x = segment[0] * GRID_SIZE + GRID_SIZE // 2
            y = segment[1] * GRID_SIZE + GRID_SIZE // 2

            if i == 0:  # Head
                # Interpolate head position for smooth movement
                prev_x = x - self.direction[0] * GRID_SIZE
                prev_y = y - self.direction[1] * GRID_SIZE
                x = int(prev_x + (x - prev_x) * self.interpolation)
                y = int(prev_y + (y - prev_y) * self.interpolation)

                # Draw head glow (pulsing)
                glow_pulse = 0.8 + 0.2 * math.sin(time * 5)
                glow_size = int(GRID_SIZE * 1.8 * glow_pulse)
                glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)

                for j in range(4):
                    alpha = int(40 * (1 - j / 4) * glow_pulse)
                    size = glow_size - j * 8
                    if size > 0:
                        pygame.draw.circle(glow_surface, (*SNAKE_HEAD_GLOW, alpha),
                                         (glow_size, glow_size), size)

                surface.blit(glow_surface, (x - glow_size, y - glow_size))

                # Head color
                color = SNAKE_HEAD_COLOR
                size = GRID_SIZE - 2
            else:
                # Body segment with subtle glow
                color = body_color
                size = GRID_SIZE - 4 - int(t * 2)  # Taper towards tail

                # Subtle glow for body
                if i < 5:  # Only first few segments get glow
                    glow_alpha = int(20 * (1 - i / 5))
                    glow_surf = pygame.Surface((size + 10, size + 10), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*body_color, glow_alpha),
                                     (size // 2 + 5, size // 2 + 5), size // 2 + 5)
                    surface.blit(glow_surf, (x - size // 2 - 5, y - size // 2 - 5))

            # Draw main segment
            pygame.draw.circle(surface, color, (x, y), size // 2)

            # Inner highlight
            if i == 0:
                highlight_color = (
                    min(255, color[0] + 60),
                    min(255, color[1] + 60),
                    min(255, color[2] + 60)
                )
                pygame.draw.circle(surface, highlight_color, (x - 2, y - 2), size // 4)

            # Draw eyes on head
            if i == 0:
                eye_offset = 5
                eye_size = 4
                pupil_size = 2
                eye_white = (240, 240, 240)
                eye_pupil = (20, 20, 20)

                # Calculate eye positions based on direction
                if self.direction == (1, 0):  # Right
                    eye_positions = [(x + 4, y - eye_offset), (x + 4, y + eye_offset)]
                elif self.direction == (-1, 0):  # Left
                    eye_positions = [(x - 4, y - eye_offset), (x - 4, y + eye_offset)]
                elif self.direction == (0, -1):  # Up
                    eye_positions = [(x - eye_offset, y - 4), (x + eye_offset, y - 4)]
                else:  # Down
                    eye_positions = [(x - eye_offset, y + 4), (x + eye_offset, y + 4)]

                for ex, ey in eye_positions:
                    pygame.draw.circle(surface, eye_white, (ex, ey), eye_size)
                    # Offset pupil slightly in direction of movement
                    px = ex + self.direction[0] * 1
                    py = ey + self.direction[1] * 1
                    pygame.draw.circle(surface, eye_pupil, (int(px), int(py)), pupil_size)


class Food:
    """Animated food with glow effect"""

    def __init__(self, is_bonus: bool = False):
        self.position = (0, 0)
        self.animation_time = 0
        self.spawn_animation = 0
        self.is_bonus = is_bonus
        self.respawn()

    def respawn(self, exclude_positions: List[Tuple[int, int]] = None):
        """Respawn food at random position"""
        exclude = set(exclude_positions) if exclude_positions else set()
        while True:
            pos = (random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2))
            if pos not in exclude:
                self.position = pos
                self.spawn_animation = 0
                break

    def update(self, dt: float):
        """Update food animation"""
        self.animation_time += dt
        self.spawn_animation = min(1.0, self.spawn_animation + dt * 3)

    def draw(self, surface: pygame.Surface):
        """Draw food with pulsing glow effect"""
        x = self.position[0] * GRID_SIZE + GRID_SIZE // 2
        y = self.position[1] * GRID_SIZE + GRID_SIZE // 2

        # Spawn scale animation
        spawn_scale = self.spawn_animation ** 0.5  # Ease out

        # Pulsing effect
        pulse = 0.85 + 0.15 * math.sin(self.animation_time * 5)

        # Choose colors
        if self.is_bonus:
            main_color = BONUS_FOOD_COLOR
            glow_color = BONUS_FOOD_GLOW
        else:
            main_color = FOOD_COLOR
            glow_color = FOOD_GLOW

        # Draw outer glow (multiple layers for smooth gradient)
        glow_size = int(GRID_SIZE * 2 * pulse * spawn_scale)
        if glow_size > 0:
            glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            for i in range(5):
                alpha = int(35 * (1 - i / 5) * spawn_scale)
                size = glow_size - i * 7
                if size > 0:
                    pygame.draw.circle(glow_surface, (*glow_color, alpha),
                                     (glow_size, glow_size), size)
            surface.blit(glow_surface, (x - glow_size, y - glow_size))

        # Draw main food body
        main_size = int((GRID_SIZE - 4) * pulse * spawn_scale)
        if main_size > 2:
            pygame.draw.circle(surface, main_color, (x, y), main_size // 2)

            # Inner shine
            shine_size = main_size // 3
            shine_color = FOOD_INNER if not self.is_bonus else (255, 255, 200)
            pygame.draw.circle(surface, shine_color,
                             (x - main_size // 6, y - main_size // 6), shine_size)

        # Floating particles around food
        if spawn_scale > 0.5:
            num_particles = 3
            for i in range(num_particles):
                angle = self.animation_time * 2 + i * (2 * math.pi / num_particles)
                dist = 12 + 4 * math.sin(self.animation_time * 3 + i)
                px = x + math.cos(angle) * dist
                py = y + math.sin(angle) * dist
                particle_alpha = int(150 * (0.5 + 0.5 * math.sin(self.animation_time * 4 + i)))

                p_surface = pygame.Surface((6, 6), pygame.SRCALPHA)
                pygame.draw.circle(p_surface, (*glow_color, particle_alpha), (3, 3), 2)
                surface.blit(p_surface, (int(px) - 3, int(py) - 3))


class Game:
    """Main game class"""

    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("🐍 Advanced Snake Game")
        self.clock = pygame.time.Clock()

        # Load or create fonts
        try:
            self.font_large = pygame.font.Font(None, 72)
            self.font_medium = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 32)
            self.font_tiny = pygame.font.Font(None, 24)
        except:
            self.font_large = pygame.font.SysFont('arial', 72)
            self.font_medium = pygame.font.SysFont('arial', 48)
            self.font_small = pygame.font.SysFont('arial', 32)
            self.font_tiny = pygame.font.SysFont('arial', 24)

        self.screen_shake = ScreenShake()
        self.high_score = 0
        self.game_time = 0
        self.reset()

    def reset(self):
        """Reset game state"""
        start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        self.snake = Snake(start_pos)
        self.food = Food()
        self.food.respawn(list(self.snake.segments))

        # Bonus food (appears sometimes)
        self.bonus_food: Optional[Food] = None
        self.bonus_timer = 0

        self.particles = ParticleSystem()
        self.background = BackgroundEffect(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.score = 0
        self.game_over = False
        self.game_over_timer = 0
        self.paused = False
        self.game_time = 0
        self.combo = 0
        self.combo_timer = 0

    def handle_input(self):
        """Handle keyboard input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p and not self.game_over:
                    self.paused = not self.paused

                if self.game_over:
                    if event.key == pygame.K_SPACE:
                        self.reset()
                    elif event.key == pygame.K_ESCAPE:
                        return False
                elif not self.paused:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.snake.set_direction((0, -1))
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.snake.set_direction((0, 1))
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        self.snake.set_direction((-1, 0))
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self.snake.set_direction((1, 0))
                    elif event.key == pygame.K_ESCAPE:
                        return False

        return True

    def spawn_bonus_food(self):
        """Spawn bonus food"""
        self.bonus_food = Food(is_bonus=True)
        self.bonus_food.respawn(list(self.snake.segments) + [self.food.position])
        self.bonus_timer = 8.0  # 8 seconds to get it

    def update(self, dt: float):
        """Update game state"""
        self.game_time += dt
        self.background.update(dt)
        self.particles.update(dt)
        self.screen_shake.update(dt)

        if self.game_over:
            self.game_over_timer += dt
            return

        if self.paused:
            return

        self.food.update(dt)

        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= dt
            if self.combo_timer <= 0:
                self.combo = 0

        # Bonus food
        if self.bonus_food:
            self.bonus_food.update(dt)
            self.bonus_timer -= dt
            if self.bonus_timer <= 0:
                # Bonus food expired - particles
                bx = self.bonus_food.position[0] * GRID_SIZE + GRID_SIZE // 2
                by = self.bonus_food.position[1] * GRID_SIZE + GRID_SIZE // 2
                self.particles.emit(bx, by, 10, BONUS_FOOD_GLOW)
                self.bonus_food = None
        elif random.random() < 0.002:  # Small chance to spawn bonus
            self.spawn_bonus_food()

        if self.snake.update(dt):
            head_pos = self.snake.get_head_pos()

            # Emit trail particle
            hx, hy = self.snake.get_interpolated_head_pos()
            if random.random() < 0.3:
                trail_color = (
                    SNAKE_BODY_START[0] // 2,
                    SNAKE_BODY_START[1] // 2,
                    SNAKE_BODY_START[2] // 2
                )
                self.particles.emit_trail(hx, hy, trail_color)

            # Check food collision
            if head_pos == self.food.position:
                self.combo += 1
                self.combo_timer = 3.0

                points = 10 * (1 + self.combo // 3)
                self.score += points
                self.high_score = max(self.high_score, self.score)
                self.snake.grow(1)

                # Emit particles
                food_x = self.food.position[0] * GRID_SIZE + GRID_SIZE // 2
                food_y = self.food.position[1] * GRID_SIZE + GRID_SIZE // 2
                self.particles.emit(food_x, food_y, 25, FOOD_COLOR)
                self.particles.emit_sparks(food_x, food_y, 10, FOOD_GLOW)

                self.food.respawn(list(self.snake.segments))

            # Check bonus food collision
            if self.bonus_food and head_pos == self.bonus_food.position:
                self.score += 50
                self.high_score = max(self.high_score, self.score)
                self.snake.grow(3)

                bx = self.bonus_food.position[0] * GRID_SIZE + GRID_SIZE // 2
                by = self.bonus_food.position[1] * GRID_SIZE + GRID_SIZE // 2
                self.particles.emit(bx, by, 40, BONUS_FOOD_COLOR)
                self.particles.emit_sparks(bx, by, 20, BONUS_FOOD_GLOW)

                self.bonus_food = None

            # Check self collision
            if self.snake.check_self_collision():
                self.game_over = True
                self.screen_shake.shake(15)

                # Death particles
                head = self.snake.get_head_pos()
                head_x = head[0] * GRID_SIZE + GRID_SIZE // 2
                head_y = head[1] * GRID_SIZE + GRID_SIZE // 2
                self.particles.emit(head_x, head_y, 60, SNAKE_HEAD_COLOR)
                self.particles.emit_sparks(head_x, head_y, 30, (255, 100, 100))

    def draw(self):
        """Draw everything"""
        # Create render surface
        render_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        # Draw background
        self.background.draw(render_surface)

        # Draw game objects
        self.food.draw(render_surface)
        if self.bonus_food:
            self.bonus_food.draw(render_surface)
        self.snake.draw(render_surface, self.game_time)
        self.particles.draw(render_surface)

        # Draw UI
        self.draw_ui(render_surface)

        if self.game_over:
            self.draw_game_over(render_surface)
        elif self.paused:
            self.draw_paused(render_surface)

        # Apply screen shake
        shake_offset = self.screen_shake.get_offset()
        self.screen.fill(DARK_BG)
        self.screen.blit(render_surface, shake_offset)

        pygame.display.flip()

    def draw_ui(self, surface: pygame.Surface):
        """Draw score and other UI elements"""
        # Score with glow effect
        score_text = f"Score: {self.score}"

        # Glow
        glow_surf = self.font_small.render(score_text, True, (100, 100, 150))
        surface.blit(glow_surf, (12, 12))

        # Main text
        score_surface = self.font_small.render(score_text, True, SCORE_COLOR)
        surface.blit(score_surface, (10, 10))

        # High score
        high_score_text = f"High: {self.high_score}"
        high_score_surface = self.font_small.render(high_score_text, True, SCORE_COLOR)
        surface.blit(high_score_surface, (WINDOW_WIDTH - high_score_surface.get_width() - 10, 10))

        # Snake length
        length_text = f"Length: {len(self.snake.segments)}"
        length_surface = self.font_tiny.render(length_text, True, (150, 150, 180))
        surface.blit(length_surface, (10, 45))

        # Combo
        if self.combo > 0:
            combo_text = f"Combo x{self.combo}!"
            combo_color = (
                min(255, 150 + self.combo * 20),
                max(100, 200 - self.combo * 10),
                100
            )
            combo_surface = self.font_small.render(combo_text, True, combo_color)
            combo_rect = combo_surface.get_rect(center=(WINDOW_WIDTH // 2, 30))
            surface.blit(combo_surface, combo_rect)

        # Bonus timer
        if self.bonus_food:
            timer_text = f"BONUS: {self.bonus_timer:.1f}s"
            timer_color = (255, 200, 50) if self.bonus_timer > 3 else (255, 100, 100)
            timer_surface = self.font_small.render(timer_text, True, timer_color)
            timer_rect = timer_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            surface.blit(timer_surface, timer_rect)

    def draw_game_over(self, surface: pygame.Surface):
        """Draw game over screen with effects"""
        # Darken screen with fade
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        alpha = min(200, int(self.game_over_timer * 250))
        overlay.fill((0, 0, 0, alpha))
        surface.blit(overlay, (0, 0))

        if self.game_over_timer > 0.3:
            # Game Over text with shadow and glow
            game_over_text = "GAME OVER"

            # Red glow
            glow_surface = self.font_large.render(game_over_text, True, (100, 30, 30))
            glow_rect = glow_surface.get_rect(center=(WINDOW_WIDTH // 2 + 2, WINDOW_HEIGHT // 2 - 48))
            surface.blit(glow_surface, glow_rect)

            # Main text
            text_surface = self.font_large.render(game_over_text, True, GAME_OVER_COLOR)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
            surface.blit(text_surface, text_rect)

            # Final score
            score_text = f"Final Score: {self.score}"
            score_surface = self.font_medium.render(score_text, True, WHITE)
            score_rect = score_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            surface.blit(score_surface, score_rect)

            # New high score?
            if self.score == self.high_score and self.score > 0:
                hs_text = "NEW HIGH SCORE!"
                hs_pulse = 0.7 + 0.3 * math.sin(self.game_over_timer * 5)
                hs_color = (int(255 * hs_pulse), int(200 * hs_pulse), 50)
                hs_surface = self.font_small.render(hs_text, True, hs_color)
                hs_rect = hs_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
                surface.blit(hs_surface, hs_rect)

            # Restart prompt
            restart_text = "Press SPACE to restart  |  ESC to quit"
            restart_surface = self.font_small.render(restart_text, True, SCORE_COLOR)
            restart_rect = restart_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 110))
            surface.blit(restart_surface, restart_rect)

    def draw_paused(self, surface: pygame.Surface):
        """Draw pause overlay"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        surface.blit(overlay, (0, 0))

        pause_text = "PAUSED"
        pause_surface = self.font_large.render(pause_text, True, TITLE_COLOR)
        pause_rect = pause_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        surface.blit(pause_surface, pause_rect)

        hint_text = "Press P to continue"
        hint_surface = self.font_small.render(hint_text, True, SCORE_COLOR)
        hint_rect = hint_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
        surface.blit(hint_surface, hint_rect)

    def run(self):
        """Main game loop"""
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            running = self.handle_input()
            self.update(dt)
            self.draw()

        pygame.quit()


def main():
    """Entry point"""
    game = Game()

    if HEADLESS:
        # Run a few frames for testing
        print("Running in headless mode for testing...")
        for i in range(100):
            dt = 1.0 / 60
            game.handle_input()
            game.update(dt)
            game.draw()
            if i == 50:
                # Simulate eating food
                game.snake.segments[0] = game.food.position
        print(f"Headless test complete. Score: {game.score}")
        pygame.quit()
    else:
        game.run()


if __name__ == "__main__":
    main()
