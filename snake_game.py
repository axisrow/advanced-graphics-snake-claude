#!/usr/bin/env python3
"""
Advanced Graphics Snake Game - ULTRA EDITION
A visually stunning snake game with cutting-edge graphics effects

Features:
- Smooth snake movement with interpolation
- Particle effects when eating food
- Glowing snake head and body gradient
- Animated pulsing food with orbiting particles
- Floating background stars with parallax
- Screen shake and chromatic aberration on death
- Trail effects with rainbow waves
- Lightning/electric effects on bonus
- Dynamic light rays from snake head
- Speed lines effect
- Bloom shader simulation
- Pulsing vignette effect
- Motion blur trails

Run with: python snake_game.py
For headless testing: SDL_VIDEODRIVER=dummy python snake_game.py --headless
"""

import os
import pygame
import random
import math
import sys
from collections import deque
from dataclasses import dataclass, field
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

# Rainbow colors for wave effect
RAINBOW_COLORS = [
    (255, 0, 0),    # Red
    (255, 127, 0),  # Orange
    (255, 255, 0),  # Yellow
    (0, 255, 0),    # Green
    (0, 255, 255),  # Cyan
    (0, 127, 255),  # Light blue
    (127, 0, 255),  # Purple
    (255, 0, 255),  # Magenta
]


def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linearly interpolate between two colors"""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def get_rainbow_color(t: float) -> Tuple[int, int, int]:
    """Get rainbow color from 0.0 to 1.0"""
    t = t % 1.0
    idx = t * (len(RAINBOW_COLORS) - 1)
    i = int(idx)
    frac = idx - i
    if i >= len(RAINBOW_COLORS) - 1:
        return RAINBOW_COLORS[-1]
    return lerp_color(RAINBOW_COLORS[i], RAINBOW_COLORS[i + 1], frac)


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
    particle_type: str = "circle"  # circle, spark, trail, lightning, ring, star
    rotation: float = 0.0
    rotation_speed: float = 0.0

    def update(self, dt: float) -> bool:
        """Update particle, return False if dead"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        self.rotation += self.rotation_speed * dt

        # Different physics for different types
        if self.particle_type == "spark":
            self.vy += 300 * dt  # Gravity
            self.vx *= 0.95
        elif self.particle_type == "trail":
            self.vx *= 0.9
            self.vy *= 0.9
        elif self.particle_type == "lightning":
            self.vx *= 0.85
            self.vy *= 0.85
        elif self.particle_type == "ring":
            self.size += 80 * dt  # Expanding ring
        elif self.particle_type == "star":
            self.vy += 50 * dt
            self.vx *= 0.98
        else:
            self.vx *= 0.98
            self.vy *= 0.98

        return self.life > 0

    def draw(self, surface: pygame.Surface):
        """Draw particle with fade effect"""
        alpha = self.life / self.max_life
        size = max(1, int(self.size * alpha)) if self.particle_type != "ring" else max(1, int(self.size))

        if size > 0:
            if self.particle_type == "spark":
                # Draw line spark
                color = tuple(int(c * alpha) for c in self.color)
                end_x = int(self.x - self.vx * 0.02)
                end_y = int(self.y - self.vy * 0.02)
                pygame.draw.line(surface, color, (int(self.x), int(self.y)), (end_x, end_y), max(1, size // 2))
            elif self.particle_type == "lightning":
                # Electric bolt effect
                color = tuple(int(c * alpha) for c in self.color)
                points = [(int(self.x), int(self.y))]
                for i in range(3):
                    offset_x = random.randint(-8, 8)
                    offset_y = random.randint(-8, 8)
                    points.append((int(self.x + self.vx * 0.01 * (i+1) + offset_x),
                                   int(self.y + self.vy * 0.01 * (i+1) + offset_y)))
                if len(points) >= 2:
                    pygame.draw.lines(surface, color, False, points, max(1, int(size * 0.5)))
            elif self.particle_type == "ring":
                # Expanding ring
                ring_alpha = int(255 * alpha)
                if ring_alpha > 0 and size > 2:
                    ring_surface = pygame.Surface((int(size * 2 + 4), int(size * 2 + 4)), pygame.SRCALPHA)
                    pygame.draw.circle(ring_surface, (*self.color, ring_alpha),
                                      (int(size + 2), int(size + 2)), int(size), max(2, int(3 * alpha)))
                    surface.blit(ring_surface, (int(self.x - size - 2), int(self.y - size - 2)))
            elif self.particle_type == "star":
                # Star shaped particle
                color = tuple(int(c * alpha) for c in self.color)
                self._draw_star(surface, int(self.x), int(self.y), size, color)
            else:
                color = tuple(int(c * alpha) for c in self.color)
                pygame.draw.circle(surface, color, (int(self.x), int(self.y)), size)

    def _draw_star(self, surface: pygame.Surface, x: int, y: int, size: float, color: Tuple[int, int, int]):
        """Draw a star shape"""
        points = []
        for i in range(5):
            angle = self.rotation + i * (2 * math.pi / 5) - math.pi / 2
            outer_x = x + math.cos(angle) * size
            outer_y = y + math.sin(angle) * size
            points.append((outer_x, outer_y))

            inner_angle = angle + math.pi / 5
            inner_x = x + math.cos(inner_angle) * (size * 0.4)
            inner_y = y + math.sin(inner_angle) * (size * 0.4)
            points.append((inner_x, inner_y))

        if len(points) >= 3:
            pygame.draw.polygon(surface, color, points)


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

    def emit_lightning(self, x: float, y: float, count: int, color: Tuple[int, int, int]):
        """Emit electric/lightning particles"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(150, 350)
            self.particles.append(Particle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=random.uniform(0.2, 0.5),
                max_life=0.5,
                color=color,
                size=random.uniform(4, 8),
                particle_type="lightning"
            ))

    def emit_ring(self, x: float, y: float, color: Tuple[int, int, int]):
        """Emit expanding ring effect"""
        self.particles.append(Particle(
            x=x,
            y=y,
            vx=0,
            vy=0,
            life=0.6,
            max_life=0.6,
            color=color,
            size=5,
            particle_type="ring"
        ))

    def emit_stars(self, x: float, y: float, count: int, color: Tuple[int, int, int]):
        """Emit star-shaped particles"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(80, 200)
            self.particles.append(Particle(
                x=x + random.uniform(-10, 10),
                y=y + random.uniform(-10, 10),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=random.uniform(0.5, 1.2),
                max_life=1.2,
                color=color,
                size=random.uniform(5, 12),
                particle_type="star",
                rotation=random.uniform(0, 2 * math.pi),
                rotation_speed=random.uniform(-5, 5)
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
        self.chromatic_aberration = 0  # For death effect

    def shake(self, amount: float, chromatic: bool = False):
        """Trigger screen shake"""
        self.shake_amount = amount
        if chromatic:
            self.chromatic_aberration = 8

    def update(self, dt: float):
        """Update shake"""
        if self.shake_amount > 0:
            self.shake_amount = max(0, self.shake_amount - self.shake_decay * dt)
        if self.chromatic_aberration > 0:
            self.chromatic_aberration = max(0, self.chromatic_aberration - 10 * dt)

    def get_offset(self) -> Tuple[int, int]:
        """Get current shake offset"""
        if self.shake_amount > 0.5:
            return (
                random.randint(int(-self.shake_amount), int(self.shake_amount)),
                random.randint(int(-self.shake_amount), int(self.shake_amount))
            )
        return (0, 0)

    def get_chromatic_offset(self) -> int:
        """Get chromatic aberration offset"""
        return int(self.chromatic_aberration)


class LightRay:
    """Dynamic light ray from snake head"""

    def __init__(self, angle: float, length: float, width: float, color: Tuple[int, int, int]):
        self.angle = angle
        self.length = length
        self.width = width
        self.color = color
        self.life = random.uniform(0.3, 0.8)
        self.max_life = self.life
        self.flicker = random.uniform(0.8, 1.2)

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.flicker = 0.7 + 0.3 * math.sin(self.life * 20)
        return self.life > 0

    def draw(self, surface: pygame.Surface, x: float, y: float):
        alpha = (self.life / self.max_life) * self.flicker
        if alpha <= 0:
            return

        # Create ray polygon
        end_x = x + math.cos(self.angle) * self.length * alpha
        end_y = y + math.sin(self.angle) * self.length * alpha

        perp_angle = self.angle + math.pi / 2
        w = self.width * alpha

        points = [
            (x + math.cos(perp_angle) * w * 0.5, y + math.sin(perp_angle) * w * 0.5),
            (x - math.cos(perp_angle) * w * 0.5, y - math.sin(perp_angle) * w * 0.5),
            (end_x, end_y),
        ]

        ray_surf = pygame.Surface((int(self.length * 2 + 20), int(self.length * 2 + 20)), pygame.SRCALPHA)
        offset = int(self.length + 10)

        shifted_points = [(p[0] - x + offset, p[1] - y + offset) for p in points]
        ray_color = (*self.color, int(80 * alpha))

        pygame.draw.polygon(ray_surf, ray_color, shifted_points)
        surface.blit(ray_surf, (int(x - offset), int(y - offset)))


class SpeedLine:
    """Speed line effect for fast movement"""

    def __init__(self, x: float, y: float, direction: Tuple[int, int]):
        self.x = x
        self.y = y
        self.direction = direction
        self.length = random.uniform(20, 50)
        self.life = 0.3
        self.max_life = 0.3
        self.offset = random.uniform(-30, 30)

    def update(self, dt: float) -> bool:
        self.life -= dt
        # Move opposite to snake direction
        self.x -= self.direction[0] * 400 * dt
        self.y -= self.direction[1] * 400 * dt
        return self.life > 0

    def draw(self, surface: pygame.Surface):
        alpha = self.life / self.max_life
        if alpha <= 0:
            return

        color = (100, 150, 200, int(100 * alpha))

        # Perpendicular offset
        if self.direction[0] != 0:
            start_y = self.y + self.offset
            start_x = self.x
            end_x = self.x - self.direction[0] * self.length
            end_y = start_y
        else:
            start_x = self.x + self.offset
            start_y = self.y
            end_y = self.y - self.direction[1] * self.length
            end_x = start_x

        line_surf = pygame.Surface((int(abs(end_x - start_x) + 10), int(abs(end_y - start_y) + 10)), pygame.SRCALPHA)
        local_start = (5, 5)
        local_end = (int(abs(end_x - start_x) + 5), int(abs(end_y - start_y) + 5))

        pygame.draw.line(line_surf, color, local_start, local_end, max(1, int(2 * alpha)))
        surface.blit(line_surf, (int(min(start_x, end_x) - 5), int(min(start_y, end_y) - 5)))


class BackgroundEffect:
    """Animated background with floating particles, grid, and parallax"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Multi-layer stars for parallax
        self.star_layers = []
        for layer in range(3):
            stars = []
            count = 40 + layer * 30
            for _ in range(count):
                stars.append({
                    'x': random.randint(0, width),
                    'y': random.randint(0, height),
                    'speed': random.uniform(3 + layer * 5, 10 + layer * 10),
                    'size': random.uniform(0.5 + layer * 0.3, 1.5 + layer * 0.5),
                    'brightness': random.uniform(0.2 + layer * 0.1, 0.6 + layer * 0.3),
                    'twinkle_speed': random.uniform(1, 4),
                    'color_shift': random.uniform(0, 1)
                })
            self.star_layers.append(stars)

        # Nebula blobs (background color variations)
        self.nebulas = []
        for _ in range(6):
            self.nebulas.append({
                'x': random.randint(0, width),
                'y': random.randint(0, height),
                'size': random.randint(100, 250),
                'color': random.choice([
                    (40, 20, 60),
                    (20, 40, 60),
                    (50, 20, 50),
                    (25, 45, 55),
                    (60, 30, 40),
                    (30, 50, 40)
                ]),
                'alpha': random.randint(15, 35),
                'drift_x': random.uniform(-5, 5),
                'drift_y': random.uniform(-3, 3)
            })

        # Shooting stars
        self.shooting_stars = []

        self.grid_alpha = 25
        self.time = 0
        self.vignette_pulse = 0

    def update(self, dt: float):
        """Update background animation"""
        self.time += dt
        self.vignette_pulse = 0.9 + 0.1 * math.sin(self.time * 2)

        # Update star layers with parallax
        for layer_idx, stars in enumerate(self.star_layers):
            for star in stars:
                star['y'] += star['speed'] * dt
                star['x'] += (layer_idx - 1) * 2 * dt  # Slight horizontal drift
                if star['y'] > self.height:
                    star['y'] = 0
                    star['x'] = random.randint(0, self.width)
                if star['x'] < 0:
                    star['x'] = self.width
                elif star['x'] > self.width:
                    star['x'] = 0

        # Update nebulas
        for nebula in self.nebulas:
            nebula['x'] += nebula['drift_x'] * dt
            nebula['y'] += nebula['drift_y'] * dt
            if nebula['x'] < -nebula['size']:
                nebula['x'] = self.width + nebula['size']
            elif nebula['x'] > self.width + nebula['size']:
                nebula['x'] = -nebula['size']
            if nebula['y'] < -nebula['size']:
                nebula['y'] = self.height + nebula['size']
            elif nebula['y'] > self.height + nebula['size']:
                nebula['y'] = -nebula['size']

        # Occasionally spawn shooting star
        if random.random() < 0.003:
            self.shooting_stars.append({
                'x': random.randint(0, self.width),
                'y': 0,
                'angle': random.uniform(0.5, 1.0),
                'speed': random.uniform(400, 700),
                'life': 1.0,
                'length': random.uniform(40, 80)
            })

        # Update shooting stars
        for ss in self.shooting_stars[:]:
            ss['x'] += math.cos(ss['angle']) * ss['speed'] * dt
            ss['y'] += math.sin(ss['angle']) * ss['speed'] * dt
            ss['life'] -= dt * 1.5
            if ss['life'] <= 0 or ss['y'] > self.height or ss['x'] > self.width:
                self.shooting_stars.remove(ss)

    def draw(self, surface: pygame.Surface):
        """Draw animated background"""
        # Fill with dark background
        surface.fill(DARK_BG)

        # Draw nebula blobs
        for nebula in self.nebulas:
            nebula_surface = pygame.Surface((nebula['size'] * 2, nebula['size'] * 2), pygame.SRCALPHA)
            for i in range(4):
                size = nebula['size'] - i * 25
                if size > 0:
                    alpha = nebula['alpha'] - i * 7
                    pygame.draw.circle(
                        nebula_surface,
                        (*nebula['color'], max(0, alpha)),
                        (nebula['size'], nebula['size']),
                        size
                    )
            surface.blit(nebula_surface, (int(nebula['x'] - nebula['size']), int(nebula['y'] - nebula['size'])))

        # Draw subtle animated grid
        grid_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        grid_pulse = 0.8 + 0.2 * math.sin(self.time * 0.5)
        grid_color = (40, 50, 70, int(self.grid_alpha * grid_pulse))

        for x in range(0, self.width, GRID_SIZE):
            pygame.draw.line(grid_surface, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, GRID_SIZE):
            pygame.draw.line(grid_surface, grid_color, (0, y), (self.width, y))

        surface.blit(grid_surface, (0, 0))

        # Draw star layers (back to front for parallax)
        for layer_idx, stars in enumerate(self.star_layers):
            for star in stars:
                twinkle = 0.6 + 0.4 * math.sin(self.time * star['twinkle_speed'] + star['x'] * 0.1)
                brightness = star['brightness'] * twinkle

                # Rainbow color shift for some stars
                if star['color_shift'] > 0.8:
                    hue = (self.time * 0.5 + star['color_shift'] * 10) % 1.0
                    rainbow = get_rainbow_color(hue)
                    color = tuple(max(0, min(255, int(c * brightness))) for c in rainbow)
                elif star['brightness'] > 0.5:
                    gray = int(150 * brightness)
                    color = (gray, gray, min(255, int(gray * 1.2)))  # Blue tint
                else:
                    gray = int(120 * brightness)
                    color = (gray, gray, gray)

                pos = (int(star['x']), int(star['y']))
                size = max(1, int(star['size'] * (0.8 + 0.2 * twinkle)))
                pygame.draw.circle(surface, color, pos, size)

        # Draw shooting stars
        for ss in self.shooting_stars:
            alpha = ss['life']
            length = ss['length'] * alpha
            end_x = ss['x'] - math.cos(ss['angle']) * length
            end_y = ss['y'] - math.sin(ss['angle']) * length

            # Gradient line for shooting star
            for i in range(3):
                seg_alpha = alpha * (1 - i * 0.3)
                seg_start_x = ss['x'] - math.cos(ss['angle']) * length * i / 3
                seg_start_y = ss['y'] - math.sin(ss['angle']) * length * i / 3
                seg_end_x = ss['x'] - math.cos(ss['angle']) * length * (i + 1) / 3
                seg_end_y = ss['y'] - math.sin(ss['angle']) * length * (i + 1) / 3

                color = (int(255 * seg_alpha), int(255 * seg_alpha), int(200 * seg_alpha))
                pygame.draw.line(surface, color,
                               (int(seg_start_x), int(seg_start_y)),
                               (int(seg_end_x), int(seg_end_y)),
                               max(1, int(3 * seg_alpha)))

    def draw_vignette(self, surface: pygame.Surface, intensity: float = 1.0):
        """Draw pulsing vignette effect"""
        vignette = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Create radial gradient vignette
        center_x, center_y = self.width // 2, self.height // 2
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

        # Draw concentric rectangles for vignette effect (more efficient)
        for i in range(10):
            t = i / 10
            alpha = int(60 * t * t * intensity * self.vignette_pulse)
            if alpha > 0:
                # Draw border rectangles
                border_width = int(self.width * t * 0.3)
                border_height = int(self.height * t * 0.3)

                pygame.draw.rect(vignette, (0, 0, 0, alpha),
                               (0, 0, self.width, border_height))
                pygame.draw.rect(vignette, (0, 0, 0, alpha),
                               (0, self.height - border_height, self.width, border_height))
                pygame.draw.rect(vignette, (0, 0, 0, alpha),
                               (0, 0, border_width, self.height))
                pygame.draw.rect(vignette, (0, 0, 0, alpha),
                               (self.width - border_width, 0, border_width, self.height))

        surface.blit(vignette, (0, 0))


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

        # Light rays from head
        self.light_rays: List[LightRay] = []
        self.ray_timer = 0

        # Rainbow wave mode (activated on long combos)
        self.rainbow_mode = False
        self.rainbow_intensity = 0.0

        # Power mode (after eating bonus)
        self.power_mode_timer = 0.0

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

    def activate_power_mode(self, duration: float = 3.0):
        """Activate power mode (visual effect after bonus)"""
        self.power_mode_timer = duration

    def set_rainbow_mode(self, enabled: bool):
        """Enable/disable rainbow wave mode"""
        self.rainbow_mode = enabled

    def update_effects(self, dt: float):
        """Update visual effects"""
        # Light rays
        self.ray_timer += dt
        if self.ray_timer > 0.1:
            self.ray_timer = 0
            # Spawn light rays in movement direction
            hx, hy = self.get_interpolated_head_pos()
            base_angle = math.atan2(self.direction[1], self.direction[0])

            for _ in range(2):
                angle = base_angle + random.uniform(-0.5, 0.5)
                self.light_rays.append(LightRay(
                    angle=angle,
                    length=random.uniform(30, 60),
                    width=random.uniform(8, 15),
                    color=SNAKE_HEAD_GLOW if not self.rainbow_mode else get_rainbow_color(random.random())
                ))

        # Update existing rays
        self.light_rays = [ray for ray in self.light_rays if ray.update(dt)]

        # Rainbow intensity
        if self.rainbow_mode:
            self.rainbow_intensity = min(1.0, self.rainbow_intensity + dt * 2)
        else:
            self.rainbow_intensity = max(0.0, self.rainbow_intensity - dt * 2)

        # Power mode timer
        if self.power_mode_timer > 0:
            self.power_mode_timer -= dt

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
        """Draw snake with gradient, glow, and rainbow effects"""
        segments_list = list(self.segments)
        num_segments = len(segments_list)

        # Draw light rays first (behind snake)
        hx, hy = self.get_interpolated_head_pos()
        for ray in self.light_rays:
            ray.draw(surface, hx, hy)

        # Draw motion trail (ghost segments)
        if len(self.position_history) > 5:
            for idx, (tx, ty) in enumerate(self.position_history[-20:]):
                trail_alpha = (idx / 20) * 0.3
                trail_size = int(GRID_SIZE * 0.3 * trail_alpha)
                if trail_size > 0:
                    trail_surf = pygame.Surface((trail_size * 2 + 4, trail_size * 2 + 4), pygame.SRCALPHA)
                    trail_color = get_rainbow_color(time * 0.5 + idx * 0.1) if self.rainbow_intensity > 0 else SNAKE_BODY_START
                    pygame.draw.circle(trail_surf, (*trail_color, int(50 * trail_alpha)),
                                      (trail_size + 2, trail_size + 2), trail_size)
                    surface.blit(trail_surf, (int(tx - trail_size - 2), int(ty - trail_size - 2)))

        # Draw body segments from tail to head
        for i in range(num_segments - 1, -1, -1):
            segment = segments_list[i]

            # Calculate gradient color with optional rainbow wave
            t = i / max(1, num_segments - 1)

            if self.rainbow_intensity > 0:
                # Rainbow wave effect
                wave_offset = time * 2 + i * 0.15
                rainbow_color = get_rainbow_color(wave_offset % 1.0)
                base_color = (
                    int(SNAKE_BODY_START[0] + (SNAKE_BODY_END[0] - SNAKE_BODY_START[0]) * t),
                    int(SNAKE_BODY_START[1] + (SNAKE_BODY_END[1] - SNAKE_BODY_START[1]) * t),
                    int(SNAKE_BODY_START[2] + (SNAKE_BODY_END[2] - SNAKE_BODY_START[2]) * t),
                )
                body_color = lerp_color(base_color, rainbow_color, self.rainbow_intensity)
            else:
                body_color = (
                    int(SNAKE_BODY_START[0] + (SNAKE_BODY_END[0] - SNAKE_BODY_START[0]) * t),
                    int(SNAKE_BODY_START[1] + (SNAKE_BODY_END[1] - SNAKE_BODY_START[1]) * t),
                    int(SNAKE_BODY_START[2] + (SNAKE_BODY_END[2] - SNAKE_BODY_START[2]) * t),
                )

            # Power mode pulsing
            if self.power_mode_timer > 0:
                pulse = 0.7 + 0.3 * math.sin(time * 15)
                body_color = (
                    min(255, int(body_color[0] * (1 + 0.5 * pulse))),
                    min(255, int(body_color[1] * (1 + 0.3 * pulse))),
                    min(255, int(body_color[2] * pulse)),
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

                # Draw head glow (pulsing) with color based on mode
                glow_pulse = 0.8 + 0.2 * math.sin(time * 5)
                glow_size = int(GRID_SIZE * 2.0 * glow_pulse)  # Bigger glow

                if self.power_mode_timer > 0:
                    glow_color = BONUS_FOOD_GLOW
                elif self.rainbow_intensity > 0:
                    glow_color = get_rainbow_color(time * 0.5)
                else:
                    glow_color = SNAKE_HEAD_GLOW

                glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)

                for j in range(5):
                    alpha = int(50 * (1 - j / 5) * glow_pulse)
                    size = glow_size - j * 7
                    if size > 0:
                        pygame.draw.circle(glow_surface, (*glow_color, alpha),
                                         (glow_size, glow_size), size)

                surface.blit(glow_surface, (x - glow_size, y - glow_size))

                # Head color
                if self.rainbow_intensity > 0:
                    head_rainbow = get_rainbow_color(time * 0.5)
                    color = lerp_color(SNAKE_HEAD_COLOR, head_rainbow, self.rainbow_intensity * 0.5)
                else:
                    color = SNAKE_HEAD_COLOR

                if self.power_mode_timer > 0:
                    color = BONUS_FOOD_COLOR

                size = GRID_SIZE - 2
            else:
                # Body segment with enhanced glow
                color = body_color
                # Sinusoidal size variation for organic look
                wave = math.sin(time * 3 + i * 0.5) * 2
                size = GRID_SIZE - 4 - int(t * 2) + int(wave)

                # Glow for body segments (more segments get glow)
                if i < 8:
                    glow_alpha = int(30 * (1 - i / 8))
                    glow_size_extra = 8 + int(4 * (1 - i / 8))
                    glow_surf = pygame.Surface((size + glow_size_extra * 2, size + glow_size_extra * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*body_color, glow_alpha),
                                     (size // 2 + glow_size_extra, size // 2 + glow_size_extra),
                                     size // 2 + glow_size_extra)
                    surface.blit(glow_surf, (x - size // 2 - glow_size_extra, y - size // 2 - glow_size_extra))

            # Draw main segment with outline
            if size > 4:
                # Outer ring (darker)
                outline_color = (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40))
                pygame.draw.circle(surface, outline_color, (x, y), size // 2)
                # Inner fill (brighter)
                pygame.draw.circle(surface, color, (x, y), size // 2 - 2)

            # Inner highlight
            if i == 0:
                highlight_color = (
                    min(255, color[0] + 80),
                    min(255, color[1] + 80),
                    min(255, color[2] + 80)
                )
                pygame.draw.circle(surface, highlight_color, (x - 3, y - 3), size // 4)

            # Draw eyes on head
            if i == 0:
                eye_offset = 5
                eye_size = 4
                pupil_size = 2
                eye_white = (250, 250, 250)
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
                    # Eye glow
                    eye_glow_surf = pygame.Surface((eye_size * 4, eye_size * 4), pygame.SRCALPHA)
                    pygame.draw.circle(eye_glow_surf, (255, 255, 255, 30), (eye_size * 2, eye_size * 2), eye_size * 2)
                    surface.blit(eye_glow_surf, (ex - eye_size * 2, ey - eye_size * 2))

                    pygame.draw.circle(surface, eye_white, (ex, ey), eye_size)
                    # Offset pupil slightly in direction of movement
                    px = ex + self.direction[0] * 1
                    py = ey + self.direction[1] * 1
                    pygame.draw.circle(surface, eye_pupil, (int(px), int(py)), pupil_size)

                    # Eye shine
                    pygame.draw.circle(surface, (255, 255, 255), (ex - 1, ey - 1), 1)


class Food:
    """Animated food with epic glow effects"""

    def __init__(self, is_bonus: bool = False):
        self.position = (0, 0)
        self.animation_time = 0
        self.spawn_animation = 0
        self.is_bonus = is_bonus
        self.rotation = 0
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
        self.rotation += dt * (3 if self.is_bonus else 1.5)

    def draw(self, surface: pygame.Surface):
        """Draw food with epic pulsing glow effect"""
        x = self.position[0] * GRID_SIZE + GRID_SIZE // 2
        y = self.position[1] * GRID_SIZE + GRID_SIZE // 2

        # Spawn scale animation with bounce
        t = self.spawn_animation
        spawn_scale = 1 - (1 - t) ** 3 if t < 1 else 1  # Ease out cubic
        if t > 0.8:
            spawn_scale *= 1 + 0.1 * math.sin((t - 0.8) * 25)  # Bounce

        # Pulsing effect
        pulse = 0.85 + 0.15 * math.sin(self.animation_time * 5)
        fast_pulse = 0.9 + 0.1 * math.sin(self.animation_time * 12)

        # Choose colors
        if self.is_bonus:
            main_color = BONUS_FOOD_COLOR
            glow_color = BONUS_FOOD_GLOW
            inner_color = (255, 255, 200)
        else:
            main_color = FOOD_COLOR
            glow_color = FOOD_GLOW
            inner_color = FOOD_INNER

        # Draw light rays for bonus food
        if self.is_bonus and spawn_scale > 0.5:
            num_rays = 8
            for i in range(num_rays):
                angle = self.rotation + i * (2 * math.pi / num_rays)
                ray_length = 25 + 10 * math.sin(self.animation_time * 3 + i)
                ray_width = 4 + 2 * math.sin(self.animation_time * 4 + i * 0.5)

                end_x = x + math.cos(angle) * ray_length
                end_y = y + math.sin(angle) * ray_length

                ray_surf = pygame.Surface((int(ray_length * 2 + 10), int(ray_length * 2 + 10)), pygame.SRCALPHA)
                center = int(ray_length + 5)

                # Draw ray as triangle
                perp = angle + math.pi / 2
                points = [
                    (center + math.cos(perp) * ray_width * 0.3, center + math.sin(perp) * ray_width * 0.3),
                    (center - math.cos(perp) * ray_width * 0.3, center - math.sin(perp) * ray_width * 0.3),
                    (center + math.cos(angle) * ray_length, center + math.sin(angle) * ray_length)
                ]
                pygame.draw.polygon(ray_surf, (*glow_color, int(60 * fast_pulse)), points)
                surface.blit(ray_surf, (int(x - center), int(y - center)))

        # Draw outer glow (multiple layers for smooth gradient)
        glow_size = int(GRID_SIZE * (2.5 if self.is_bonus else 2.0) * pulse * spawn_scale)
        if glow_size > 0:
            glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            for i in range(6):
                alpha = int((45 if self.is_bonus else 35) * (1 - i / 6) * spawn_scale)
                size = glow_size - i * 6
                if size > 0:
                    pygame.draw.circle(glow_surface, (*glow_color, alpha),
                                     (glow_size, glow_size), size)
            surface.blit(glow_surface, (x - glow_size, y - glow_size))

        # Draw main food body
        main_size = int((GRID_SIZE - 2) * pulse * spawn_scale)
        if main_size > 2:
            # Outer ring
            pygame.draw.circle(surface, tuple(max(0, c - 40) for c in main_color), (x, y), main_size // 2)
            # Main circle
            pygame.draw.circle(surface, main_color, (x, y), main_size // 2 - 2)

            # Inner shine (multiple layers)
            shine_size = main_size // 3
            pygame.draw.circle(surface, inner_color,
                             (x - main_size // 6, y - main_size // 6), shine_size)
            # Tiny highlight
            pygame.draw.circle(surface, (255, 255, 255),
                             (x - main_size // 5, y - main_size // 5), max(1, shine_size // 3))

        # Orbiting particles
        if spawn_scale > 0.5:
            num_particles = 5 if self.is_bonus else 3
            for i in range(num_particles):
                angle = self.animation_time * (3 if self.is_bonus else 2) + i * (2 * math.pi / num_particles)
                # Elliptical orbit
                dist_x = (15 + 5 * math.sin(self.animation_time * 3 + i)) * spawn_scale
                dist_y = (10 + 3 * math.sin(self.animation_time * 2 + i * 2)) * spawn_scale
                px = x + math.cos(angle) * dist_x
                py = y + math.sin(angle) * dist_y
                particle_alpha = int(180 * (0.5 + 0.5 * math.sin(self.animation_time * 4 + i)))

                psize = 3 if self.is_bonus else 2
                p_surface = pygame.Surface((psize * 2 + 4, psize * 2 + 4), pygame.SRCALPHA)
                # Glow
                pygame.draw.circle(p_surface, (*glow_color, particle_alpha // 2), (psize + 2, psize + 2), psize + 2)
                # Core
                pygame.draw.circle(p_surface, (*glow_color, particle_alpha), (psize + 2, psize + 2), psize)
                surface.blit(p_surface, (int(px) - psize - 2, int(py) - psize - 2))

        # Energy ring for bonus food
        if self.is_bonus and spawn_scale > 0.5:
            ring_size = int(20 + 5 * math.sin(self.animation_time * 6))
            ring_alpha = int(60 * fast_pulse * spawn_scale)
            ring_surf = pygame.Surface((ring_size * 2 + 4, ring_size * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (*glow_color, ring_alpha), (ring_size + 2, ring_size + 2), ring_size, 2)
            surface.blit(ring_surf, (x - ring_size - 2, y - ring_size - 2))


class Game:
    """Main game class"""

    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("🐍 ULTRA Snake Game")
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

        # Speed lines effect
        self.speed_lines: List[SpeedLine] = []

        # Bloom surface for post-processing
        self.bloom_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

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

        # Update speed lines
        self.speed_lines = [sl for sl in self.speed_lines if sl.update(dt)]

        if self.game_over:
            self.game_over_timer += dt
            return

        if self.paused:
            return

        self.food.update(dt)
        self.snake.update_effects(dt)

        # Rainbow mode on high combo
        if self.combo >= 5:
            self.snake.set_rainbow_mode(True)
        else:
            self.snake.set_rainbow_mode(False)

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
                self.particles.emit_ring(bx, by, (100, 80, 60))
                self.bonus_food = None
        elif random.random() < 0.002:  # Small chance to spawn bonus
            self.spawn_bonus_food()

        if self.snake.update(dt):
            head_pos = self.snake.get_head_pos()

            # Emit trail particle
            hx, hy = self.snake.get_interpolated_head_pos()
            if random.random() < 0.4:
                if self.snake.rainbow_intensity > 0:
                    trail_color = get_rainbow_color(self.game_time + random.random())
                else:
                    trail_color = (
                        SNAKE_BODY_START[0] // 2,
                        SNAKE_BODY_START[1] // 2,
                        SNAKE_BODY_START[2] // 2
                    )
                self.particles.emit_trail(hx, hy, trail_color)

            # Speed lines
            if random.random() < 0.3:
                self.speed_lines.append(SpeedLine(hx, hy, self.snake.direction))

            # Check food collision
            if head_pos == self.food.position:
                self.combo += 1
                self.combo_timer = 3.0

                points = 10 * (1 + self.combo // 3)
                self.score += points
                self.high_score = max(self.high_score, self.score)
                self.snake.grow(1)

                # Emit particles - enhanced
                food_x = self.food.position[0] * GRID_SIZE + GRID_SIZE // 2
                food_y = self.food.position[1] * GRID_SIZE + GRID_SIZE // 2
                self.particles.emit(food_x, food_y, 30, FOOD_COLOR)
                self.particles.emit_sparks(food_x, food_y, 15, FOOD_GLOW)
                self.particles.emit_ring(food_x, food_y, FOOD_GLOW)

                # Stars on combo
                if self.combo > 2:
                    self.particles.emit_stars(food_x, food_y, self.combo, FOOD_GLOW)

                self.food.respawn(list(self.snake.segments))

            # Check bonus food collision
            if self.bonus_food and head_pos == self.bonus_food.position:
                self.score += 50
                self.high_score = max(self.high_score, self.score)
                self.snake.grow(3)
                self.snake.activate_power_mode(3.0)

                bx = self.bonus_food.position[0] * GRID_SIZE + GRID_SIZE // 2
                by = self.bonus_food.position[1] * GRID_SIZE + GRID_SIZE // 2

                # Epic particle explosion
                self.particles.emit(bx, by, 50, BONUS_FOOD_COLOR)
                self.particles.emit_sparks(bx, by, 25, BONUS_FOOD_GLOW)
                self.particles.emit_lightning(bx, by, 15, (255, 255, 200))
                self.particles.emit_ring(bx, by, BONUS_FOOD_GLOW)
                self.particles.emit_stars(bx, by, 8, BONUS_FOOD_COLOR)

                # Screen shake
                self.screen_shake.shake(8)

                self.bonus_food = None

            # Check self collision
            if self.snake.check_self_collision():
                self.game_over = True
                self.screen_shake.shake(20, chromatic=True)

                # Death particles - massive explosion
                head = self.snake.get_head_pos()
                head_x = head[0] * GRID_SIZE + GRID_SIZE // 2
                head_y = head[1] * GRID_SIZE + GRID_SIZE // 2
                self.particles.emit(head_x, head_y, 80, SNAKE_HEAD_COLOR)
                self.particles.emit_sparks(head_x, head_y, 40, (255, 100, 100))
                self.particles.emit_lightning(head_x, head_y, 20, (255, 50, 50))
                self.particles.emit_ring(head_x, head_y, (255, 80, 80))
                self.particles.emit_ring(head_x, head_y, (255, 150, 150))

    def draw(self):
        """Draw everything with post-processing effects"""
        # Create render surface
        render_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        # Draw background
        self.background.draw(render_surface)

        # Draw speed lines (behind everything else)
        for speed_line in self.speed_lines:
            speed_line.draw(render_surface)

        # Draw game objects
        self.food.draw(render_surface)
        if self.bonus_food:
            self.bonus_food.draw(render_surface)
        self.snake.draw(render_surface, self.game_time)
        self.particles.draw(render_surface)

        # Draw vignette
        self.background.draw_vignette(render_surface, 1.0 if not self.game_over else 1.5)

        # Draw UI
        self.draw_ui(render_surface)

        if self.game_over:
            self.draw_game_over(render_surface)
        elif self.paused:
            self.draw_paused(render_surface)

        # Apply post-processing
        self.screen.fill(DARK_BG)

        # Chromatic aberration on death
        chromatic_offset = self.screen_shake.get_chromatic_offset()
        if chromatic_offset > 0:
            # Create RGB separation effect
            self.screen.fill(DARK_BG)

            # Red channel offset
            red_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            red_surface.blit(render_surface, (0, 0))
            red_surface.set_alpha(200)
            self.screen.blit(red_surface, (-chromatic_offset, 0), special_flags=pygame.BLEND_ADD)

            # Green channel (center)
            green_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            green_surface.blit(render_surface, (0, 0))
            green_surface.set_alpha(200)
            self.screen.blit(green_surface, (0, 0), special_flags=pygame.BLEND_ADD)

            # Blue channel offset
            blue_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            blue_surface.blit(render_surface, (0, 0))
            blue_surface.set_alpha(200)
            self.screen.blit(blue_surface, (chromatic_offset, 0), special_flags=pygame.BLEND_ADD)
        else:
            # Apply screen shake
            shake_offset = self.screen_shake.get_offset()
            self.screen.blit(render_surface, shake_offset)

        # Simple bloom effect simulation (additive layer)
        if self.snake.power_mode_timer > 0 or self.snake.rainbow_intensity > 0.5:
            bloom = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            bloom.blit(render_surface, (0, 0))
            bloom.set_alpha(30)
            self.screen.blit(bloom, (2, 2), special_flags=pygame.BLEND_ADD)
            self.screen.blit(bloom, (-2, -2), special_flags=pygame.BLEND_ADD)

        pygame.display.flip()

    def draw_ui(self, surface: pygame.Surface):
        """Draw score and other UI elements with enhanced effects"""
        # Score with animated glow effect
        score_text = f"Score: {self.score}"
        score_pulse = 0.9 + 0.1 * math.sin(self.game_time * 3)

        # Multiple glow layers
        for i in range(3):
            glow_alpha = int(80 * (1 - i * 0.3) * score_pulse)
            glow_color = (int(100 * score_pulse), int(100 * score_pulse), int(150 * score_pulse))
            glow_surf = self.font_small.render(score_text, True, glow_color)
            glow_surf.set_alpha(glow_alpha)
            surface.blit(glow_surf, (12 + i, 12 + i))

        # Main text
        score_surface = self.font_small.render(score_text, True, SCORE_COLOR)
        surface.blit(score_surface, (10, 10))

        # High score
        high_score_text = f"High: {self.high_score}"
        high_score_surface = self.font_small.render(high_score_text, True, SCORE_COLOR)
        surface.blit(high_score_surface, (WINDOW_WIDTH - high_score_surface.get_width() - 10, 10))

        # Snake length with icon
        length_text = f"Length: {len(self.snake.segments)}"
        length_surface = self.font_tiny.render(length_text, True, (150, 150, 180))
        surface.blit(length_surface, (10, 45))

        # Power mode indicator
        if self.snake.power_mode_timer > 0:
            power_pulse = 0.5 + 0.5 * math.sin(self.game_time * 10)
            power_text = "POWER MODE!"
            power_color = (int(255 * power_pulse), int(200 * power_pulse), 50)
            power_surface = self.font_small.render(power_text, True, power_color)
            power_rect = power_surface.get_rect(center=(WINDOW_WIDTH // 2, 60))

            # Glow behind
            glow = pygame.Surface((power_surface.get_width() + 20, power_surface.get_height() + 10), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*BONUS_FOOD_GLOW, int(50 * power_pulse)), (0, 0, glow.get_width(), glow.get_height()), border_radius=5)
            surface.blit(glow, (power_rect.x - 10, power_rect.y - 5))
            surface.blit(power_surface, power_rect)

        # Combo with rainbow effect on high combos
        if self.combo > 0:
            if self.combo >= 5:
                # Rainbow combo text
                combo_color = get_rainbow_color(self.game_time * 2)
            else:
                combo_color = (
                    min(255, 150 + self.combo * 20),
                    max(100, 200 - self.combo * 10),
                    100
                )

            combo_text = f"Combo x{self.combo}!"
            combo_scale = 1.0 + 0.1 * math.sin(self.game_time * 8)

            # Combo glow
            combo_glow = self.font_small.render(combo_text, True, combo_color)
            combo_glow.set_alpha(100)
            glow_rect = combo_glow.get_rect(center=(WINDOW_WIDTH // 2 + 2, 32))
            surface.blit(combo_glow, glow_rect)

            combo_surface = self.font_small.render(combo_text, True, combo_color)
            combo_rect = combo_surface.get_rect(center=(WINDOW_WIDTH // 2, 30))
            surface.blit(combo_surface, combo_rect)

            # Show "RAINBOW MODE!" text when combo is high
            if self.combo >= 5:
                rainbow_text = "RAINBOW MODE!"
                rainbow_surface = self.font_tiny.render(rainbow_text, True, get_rainbow_color(self.game_time * 3 + 0.5))
                rainbow_rect = rainbow_surface.get_rect(center=(WINDOW_WIDTH // 2, 55))
                surface.blit(rainbow_surface, rainbow_rect)

        # Bonus timer with pulsing effect
        if self.bonus_food:
            pulse = 0.8 + 0.2 * math.sin(self.game_time * 6)
            timer_text = f"BONUS: {self.bonus_timer:.1f}s"

            if self.bonus_timer > 3:
                timer_color = (int(255 * pulse), int(200 * pulse), 50)
            else:
                # Urgent pulsing when time is low
                urgent_pulse = 0.5 + 0.5 * math.sin(self.game_time * 15)
                timer_color = (255, int(100 * urgent_pulse), int(100 * urgent_pulse))

            timer_surface = self.font_small.render(timer_text, True, timer_color)
            timer_rect = timer_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))

            # Background glow
            timer_glow = pygame.Surface((timer_surface.get_width() + 30, timer_surface.get_height() + 16), pygame.SRCALPHA)
            pygame.draw.rect(timer_glow, (50, 40, 10, int(100 * pulse)), (0, 0, timer_glow.get_width(), timer_glow.get_height()), border_radius=8)
            surface.blit(timer_glow, (timer_rect.x - 15, timer_rect.y - 8))
            surface.blit(timer_surface, timer_rect)

    def draw_game_over(self, surface: pygame.Surface):
        """Draw game over screen with epic effects"""
        # Darken screen with fade
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        alpha = min(220, int(self.game_over_timer * 300))
        overlay.fill((0, 0, 0, alpha))
        surface.blit(overlay, (0, 0))

        if self.game_over_timer > 0.3:
            # Animated background particles on game over
            for _ in range(int(self.game_over_timer * 2)):
                px = random.randint(0, WINDOW_WIDTH)
                py = random.randint(0, WINDOW_HEIGHT)
                psize = random.randint(1, 3)
                palpha = random.randint(20, 60)
                ps = pygame.Surface((psize * 2, psize * 2), pygame.SRCALPHA)
                pygame.draw.circle(ps, (255, 50, 50, palpha), (psize, psize), psize)
                surface.blit(ps, (px, py))

            # Game Over text with multiple glow layers
            game_over_text = "GAME OVER"
            pulse = 0.8 + 0.2 * math.sin(self.game_over_timer * 4)

            # Multiple glow layers
            for i in range(4):
                glow_alpha = int(60 * (1 - i * 0.2) * pulse)
                glow_offset = i * 2
                glow_color = (int(150 * pulse), int(30 * pulse), int(30 * pulse))
                glow_surface = self.font_large.render(game_over_text, True, glow_color)
                glow_surface.set_alpha(glow_alpha)
                glow_rect = glow_surface.get_rect(center=(WINDOW_WIDTH // 2 + glow_offset, WINDOW_HEIGHT // 2 - 48 + glow_offset))
                surface.blit(glow_surface, glow_rect)

            # Main text with slight animation
            text_y_offset = math.sin(self.game_over_timer * 2) * 3
            text_surface = self.font_large.render(game_over_text, True, GAME_OVER_COLOR)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50 + text_y_offset))
            surface.blit(text_surface, text_rect)

            # Final score with glow
            score_text = f"Final Score: {self.score}"
            score_glow = self.font_medium.render(score_text, True, (100, 100, 100))
            score_glow.set_alpha(100)
            surface.blit(score_glow, score_glow.get_rect(center=(WINDOW_WIDTH // 2 + 2, WINDOW_HEIGHT // 2 + 22)))

            score_surface = self.font_medium.render(score_text, True, WHITE)
            score_rect = score_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            surface.blit(score_surface, score_rect)

            # New high score with rainbow effect
            if self.score == self.high_score and self.score > 0:
                hs_text = "NEW HIGH SCORE!"
                hs_color = get_rainbow_color(self.game_over_timer * 2)
                hs_pulse = 0.7 + 0.3 * math.sin(self.game_over_timer * 8)

                # Glow
                hs_glow = self.font_small.render(hs_text, True, hs_color)
                hs_glow.set_alpha(int(100 * hs_pulse))
                for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
                    surface.blit(hs_glow, hs_glow.get_rect(center=(WINDOW_WIDTH // 2 + offset[0], WINDOW_HEIGHT // 2 + 60 + offset[1])))

                hs_surface = self.font_small.render(hs_text, True, hs_color)
                hs_rect = hs_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
                surface.blit(hs_surface, hs_rect)

                # Star particles around text
                for i in range(5):
                    angle = self.game_over_timer * 2 + i * (2 * math.pi / 5)
                    star_x = WINDOW_WIDTH // 2 + math.cos(angle) * 120
                    star_y = WINDOW_HEIGHT // 2 + 60 + math.sin(angle) * 20
                    star_color = get_rainbow_color(self.game_over_timer + i * 0.2)
                    pygame.draw.circle(surface, star_color, (int(star_x), int(star_y)), 3)

            # Restart prompt with fade in
            prompt_alpha = min(255, int((self.game_over_timer - 0.5) * 200))
            if prompt_alpha > 0:
                restart_text = "Press SPACE to restart  |  ESC to quit"
                restart_surface = self.font_small.render(restart_text, True, SCORE_COLOR)
                restart_surface.set_alpha(prompt_alpha)
                restart_rect = restart_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 110))
                surface.blit(restart_surface, restart_rect)

    def draw_paused(self, surface: pygame.Surface):
        """Draw pause overlay with cool effects"""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 20, 180))
        surface.blit(overlay, (0, 0))

        # Animated background lines
        for i in range(20):
            line_y = (self.game_time * 30 + i * 40) % WINDOW_HEIGHT
            line_alpha = 30 + int(20 * math.sin(self.game_time * 2 + i))
            pygame.draw.line(surface, (100, 150, 200, line_alpha), (0, int(line_y)), (WINDOW_WIDTH, int(line_y)), 1)

        pause_text = "PAUSED"

        # Glow effect
        for i in range(3):
            glow_color = (int(80 - i * 20), int(160 - i * 40), int(220 - i * 50))
            glow_surface = self.font_large.render(pause_text, True, glow_color)
            glow_surface.set_alpha(80 - i * 20)
            glow_rect = glow_surface.get_rect(center=(WINDOW_WIDTH // 2 + i, WINDOW_HEIGHT // 2 + i))
            surface.blit(glow_surface, glow_rect)

        pause_surface = self.font_large.render(pause_text, True, TITLE_COLOR)
        pause_rect = pause_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        surface.blit(pause_surface, pause_rect)

        # Animated hint
        hint_pulse = 0.7 + 0.3 * math.sin(self.game_time * 3)
        hint_text = "Press P to continue"
        hint_color = (int(180 * hint_pulse), int(180 * hint_pulse), int(220 * hint_pulse))
        hint_surface = self.font_small.render(hint_text, True, hint_color)
        hint_rect = hint_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
        surface.blit(hint_surface, hint_rect)

        # Corner decorations
        corner_size = 30
        corner_color = (100, 150, 200, 100)
        corners = [
            (10, 10),
            (WINDOW_WIDTH - 10 - corner_size, 10),
            (10, WINDOW_HEIGHT - 10 - corner_size),
            (WINDOW_WIDTH - 10 - corner_size, WINDOW_HEIGHT - 10 - corner_size)
        ]
        for cx, cy in corners:
            pygame.draw.rect(surface, TITLE_COLOR, (cx, cy, corner_size, 3))
            pygame.draw.rect(surface, TITLE_COLOR, (cx, cy, 3, corner_size))

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
