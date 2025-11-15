#!/usr/bin/env python3
"""
td_winterfell_v7_1.py
Winterfell Defense v7.1 — same as v7 but with base.wav as the looping background ambience
(intended to be calm northern/wind atmosphere). Missing SFX are handled gracefully.

Controls:
- Left click: place a soldier (random type)
- SPACE: pause/resume
- R: reset
- G: toggle grid
- S: toggle snow
- M: toggle music (base.wav)
- V: toggle sfx
- Q: quit
"""

import pygame, sys, math, random, time, os
from collections import deque

# -------------------------
# Configuration
# -------------------------
SCREEN_W, SCREEN_H = 1280, 800
FPS = 60

# Enemy (red)
ENEMY_RADIUS = 12
ENEMY_COLOR = (225, 70, 70)
ENEMY_AURA = (255, 120, 120, 60)
ENEMY_SPEED = 65.0
ENEMY_HP = 8
ENEMY_JITTER = 3.2

ELITE_CHANCE_BASE = 0.07
ELITE_MULT_HP = 2.6
ELITE_COLOR = (255, 120, 120)

# Soldier/tower parameters (3 soldier types)
SOLDIER_TYPES = {
    "Archer":  {"range": 220, "fire_rate": 1.1, "damage": 3, "color": (100,200,170)},
    "Footman": {"range": 48,  "fire_rate": 1.0, "damage": 6, "color": (200,160,100)},  # melee
    "Ranger":  {"range": 160, "fire_rate": 1.0, "damage": 4, "color": (140,200,160)}
}

# Movement speeds by soldier type (pixels/sec)
SOLDIER_SPEED = {
    "Archer": 150.0,
    "Footman": 200.0,
    "Ranger": 170.0
}

SOLDIER_ICON_RADIUS = 12

# Projectiles
PROJECTILE_COLOR = (170, 240, 255)
PROJECTILE_SPEED = 780.0
PROJECTILE_LIFE = 1.0

# Base
BASE_RADIUS = 48
BASE_HP_MAX = 40

# Visuals & effects
SNOW_COUNT = 140
FOG_ALPHA = 70
GRID_SIZE = 28
VIGNETTE_STRONG = 200

# Spawn config (continuous)
SPAWN_BASE_RATE = 0.8
SPAWN_MIN = 0.16
SPAWN_ACCEL = 0.0009
SPAWN_ELITE_SCALE = 0.0004

NUM_PATHS = 24  # every 15 degrees

# Auto-deploy behavior (spawn when enemies approach)
AUTO_DEPLOY_RANGE = 350.0
AUTO_DEPLOY_COOLDOWN = 2.5   # seconds between auto deploys

# Wall troops limit
MAX_ACTIVE_SOLDIERS = 25

# Sound files folder (optional)
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sounds")
SOUND_FILES = {
    "shot": "arrow_shot.wav",
    "hit": "enemy_hit.wav",
    "die": "enemy_die.wav",
    "base": "base_hit.wav",   # plays when base is hit
    "music": "base.wav",      # looping background music
    "deploy": "war_horn.wav"
}

# -------------------------
# Utilities
# -------------------------
def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
def lerp(a,b,t):
    return a + (b-a) * t
def clamp(x,a,b):
    return max(a, min(b, x))

# -------------------------
# Path generation (hidden)
# -------------------------
BASE_POS = (SCREEN_W//2, SCREEN_H//2)

def make_paths(num_paths=NUM_PATHS, base=BASE_POS, outer_margin=80, mid_dist=260):
    cx, cy = base
    paths = {}
    max_dim = max(SCREEN_W, SCREEN_H)
    for i in range(num_paths):
        angle = (2*math.pi) * (i / num_paths)
        sx = cx + math.cos(angle) * (max_dim * 0.9 + outer_margin)
        sy = cy + math.sin(angle) * (max_dim * 0.9 + outer_margin)
        mx = cx + math.cos(angle) * mid_dist + random.uniform(-30,30)
        my = cy + math.sin(angle) * mid_dist + random.uniform(-30,30)
        paths[i] = [(sx, sy), (mx, my), base]
    return paths

MULTI_PATHS = make_paths()

def build_segments(points):
    segs = []; total = 0.0
    for i in range(len(points)-1):
        a = points[i]; b = points[i+1]
        l = dist(a,b)
        segs.append({'a':a, 'b':b, 'len':l})
        total += l
    return segs, total

PATH_SEGMENTS = {}
PATH_TOTAL = {}
for k, pts in MULTI_PATHS.items():
    segs, tot = build_segments(pts)
    PATH_SEGMENTS[k] = segs
    PATH_TOTAL[k] = tot

def pos_on_path_key(key, s):
    segs = PATH_SEGMENTS[key]
    if s <= 0: return segs[0]['a']
    rem = s
    for seg in segs:
        if rem <= seg['len']:
            t = rem / seg['len'] if seg['len']>0 else 0.0
            return (lerp(seg['a'][0], seg['b'][0], t), lerp(seg['a'][1], seg['b'][1], t))
        rem -= seg['len']
    return segs[-1]['b']

# -------------------------
# Entities
# -------------------------
class Enemy:
    def __init__(self, path_key, delay=0.0, elite=False):
        self.path_key = path_key
        self.s = 0.0
        self.speed = ENEMY_SPEED * (0.9 + random.uniform(-0.12,0.18))
        self.hp = int((ENEMY_HP * (ELITE_MULT_HP if elite else 1.0)) + random.randint(-1,1))
        self.spawn_delay = delay
        self.alive = True
        self.reached = False
        self.phase = random.uniform(0, 2*math.pi)
        self.is_elite = elite

    def step(self, dt):
        if not self.alive:
            return
        if self.spawn_delay > 0:
            self.spawn_delay -= dt
            return
        self.s += self.speed * dt
        if self.s >= PATH_TOTAL[self.path_key]:
            self.alive = False
            self.reached = True

    def pos(self):
        x,y = pos_on_path_key(self.path_key, self.s)
        jx = math.sin(time.time()*2.0 + self.phase) * ENEMY_JITTER
        jy = math.cos(time.time()*1.7 + self.phase) * (ENEMY_JITTER*0.6)
        nx = jx + random.uniform(-0.6,0.6)
        ny = jy + random.uniform(-0.6,0.6)
        return (x+nx, y+ny)

    def label(self):
        return "Elite Wight" if self.is_elite else "Wight"

class DeadSoldier:
    """Simple ragdoll/fade for dead soldiers"""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.age = 0.0
        self.life = 1.5
        self.fade = 255

    def step(self, dt):
        self.age += dt
        self.fade = max(0, int(255 * (1 - self.age / self.life)))
        self.y += 10 * dt
        return self.age >= self.life

    def draw(self, surf):
        alpha_color = (self.color[0], self.color[1], self.color[2], self.fade)
        s = pygame.Surface((40,40), pygame.SRCALPHA)
        pygame.draw.circle(s, alpha_color, (20,20), 12)
        surf.blit(s, (self.x-20, self.y-20))

class Soldier:
    def __init__(self, pos, kind="Archer", spawn_angle=None):
        self.x, self.y = pos
        self.kind = kind if kind in SOLDIER_TYPES else "Archer"
        spec = SOLDIER_TYPES[self.kind]
        self.range = spec["range"]
        self.fire_rate = spec["fire_rate"]
        self.damage = spec["damage"]
        self.color = spec["color"]
        self.cooldown = 0.0
        self.last_shot = 0.0
        self.speed = SOLDIER_SPEED[self.kind]
        self.phase = random.uniform(0,2*math.pi)

        # AI state
        self.target_enemy = None
        self.returning = False
        self.spawn_angle = spawn_angle if spawn_angle is not None else random.uniform(0, 2*math.pi)
        self.rim_x = BASE_POS[0] + math.cos(self.spawn_angle) * (BASE_RADIUS + 14)
        self.rim_y = BASE_POS[1] + math.sin(self.spawn_angle) * (BASE_RADIUS + 14)

    def pick_target(self, enemies):
        alive = [e for e in enemies if e.alive and e.spawn_delay <= 0]
        if not alive:
            self.target_enemy = None
            return
        alive.sort(key=lambda e: dist((self.x,self.y), e.pos()))
        self.target_enemy = alive[0]
        self.returning = False

    def step(self, dt, enemies, projectiles, sound_engine, stats):
        """
        Returns: (died_flag, dead_x, dead_y, dead_color) — if soldier died this step
        """
        # cooldown
        if self.cooldown > 0:
            self.cooldown -= dt

        # acquire/validate target
        if self.target_enemy is None or not self.target_enemy.alive:
            self.pick_target(enemies)

        # if no target: return to rim
        if self.target_enemy is None:
            dx = self.rim_x - self.x; dy = self.rim_y - self.y
            d = math.hypot(dx, dy)
            if d > 6:
                nx, ny = dx / d, dy / d
                self.x += nx * self.speed * dt * 0.8
                self.y += ny * self.speed * dt * 0.8
            self.returning = True
            return (False, None, None, None)

        # otherwise chase target
        ex, ey = self.target_enemy.pos()
        dx, dy = ex - self.x, ey - self.y
        d = math.hypot(dx, dy)

        # soldier can be killed if enemy gets too close (bite)
        if d < (ENEMY_RADIUS + SOLDIER_ICON_RADIUS + 2):
            # soldier dies, create ragdoll from caller
            return (True, self.x, self.y, self.color)

        # movement toward desired distance
        desired_distance = self.range * 0.85 if self.kind != "Footman" else (self.range * 0.5)
        if d > desired_distance:
            if d > 0:
                nx, ny = dx / d, dy / d
                self.x += nx * self.speed * dt
                self.y += ny * self.speed * dt
            return (False, None, None, None)

        # in attack zone
        if self.cooldown <= 0:
            if self.kind == "Footman":
                # melee strike - requires VERY close
                if d <= (ENEMY_RADIUS + 8 + 4):
                    self.target_enemy.hp -= self.damage
                    if hasattr(sound_engine, "sfx_enabled") and sound_engine.sfx_enabled and sound_engine.sounds.get("hit"):
                        sound_engine.sounds["hit"].play()
                    if self.target_enemy.hp <= 0:
                        # kill
                        if not getattr(self.target_enemy, "alive", False):
                            pass
                        self.target_enemy.alive = False
                        stats["kills"] += 1
                        if self.target_enemy.is_elite:
                            stats["elite_kills"] += 1
                        if hasattr(sound_engine, "sfx_enabled") and sound_engine.sfx_enabled and sound_engine.sounds.get("die"):
                            sound_engine.sounds["die"].play()
                    self.cooldown = 1.0 / self.fire_rate
            else:
                # ranged attack (projectile)
                if d > 1:
                    dirx, diry = dx/d, dy/d
                    projectiles.append(Projectile((self.x, self.y), (dirx, diry), PROJECTILE_SPEED, PROJECTILE_LIFE, self.damage, self.target_enemy))
                    if hasattr(sound_engine, "sfx_enabled") and sound_engine.sfx_enabled and sound_engine.sounds.get("shot"):
                        sound_engine.sounds["shot"].play()
                    self.cooldown = 1.0 / self.fire_rate
        return (False, None, None, None)

    def draw(self, surf, font):
        glow = pygame.Surface((self.range*2+6, self.range*2+6), pygame.SRCALPHA)
        pygame.draw.circle(glow, (self.color[0], self.color[1], self.color[2], 26), (int(self.range+3), int(self.range+3)), int(self.range))
        surf.blit(glow, (self.x - self.range - 3, self.y - self.range - 3))
        pygame.draw.circle(surf, (28,28,32), (int(self.x), int(self.y)), SOLDIER_ICON_RADIUS+4)
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), SOLDIER_ICON_RADIUS)
        px1 = int(self.x + math.cos(self.phase)*SOLDIER_ICON_RADIUS)
        py1 = int(self.y + math.sin(self.phase)*SOLDIER_ICON_RADIUS)
        px2 = int(self.x + math.cos(self.phase)* (SOLDIER_ICON_RADIUS + 18))
        py2 = int(self.y + math.sin(self.phase)* (SOLDIER_ICON_RADIUS + 18))
        pygame.draw.line(surf, (200,200,200), (px1,py1), (px2,py2), 3)
        txt = font.render(self.kind, True, (230,230,230))
        surf.blit(txt, (int(self.x - txt.get_width()/2), int(self.y - SOLDIER_ICON_RADIUS - 20)))

class Projectile:
    def __init__(self, pos, dirv, speed, life, damage, target):
        self.x, self.y = pos
        self.dirx, self.diry = dirv
        self.speed = speed
        self.life = life
        self.damage = damage
        self.target = target
        self.age = 0.0

    def step(self, dt, sound_engine, stats):
        self.age += dt
        if self.target and self.target.alive:
            tx,ty = self.target.pos()
            vx, vy = tx - self.x, ty - self.y
            d = math.hypot(vx,vy)
            if d > 1:
                nx, ny = vx/d, vy/d
                blend = 0.12
                self.dirx = (1-blend)*self.dirx + blend*nx
                self.diry = (1-blend)*self.diry + blend*ny
                nd = math.hypot(self.dirx, self.diry)
                if nd>0:
                    self.dirx /= nd; self.diry /= nd
        self.x += self.dirx * self.speed * dt
        self.y += self.diry * self.speed * dt
        if self.target and self.target.alive:
            if dist((self.x,self.y), self.target.pos()) <= ENEMY_RADIUS + 4:
                self.target.hp -= self.damage
                if hasattr(sound_engine, "sfx_enabled") and sound_engine.sfx_enabled and sound_engine.sounds.get("hit"):
                    sound_engine.sounds["hit"].play()
                if self.target.hp <= 0:
                    # mark dead and count stats
                    self.target.alive = False
                    stats["kills"] += 1
                    if self.target.is_elite:
                        stats["elite_kills"] += 1
                    if hasattr(sound_engine, "sfx_enabled") and sound_engine.sfx_enabled and sound_engine.sounds.get("die"):
                        sound_engine.sounds["die"].play()
                self.age = self.life + 1.0
        return self.age > self.life

class Base:
    def __init__(self, pos):
        self.x, self.y = pos
        self.hp = BASE_HP_MAX

    def draw(self, surf, font):
        pygame.draw.circle(surf, (50,50,60), (int(self.x), int(self.y)), BASE_RADIUS)
        batt_w = 9
        for i in range(-3,4):
            bx = int(self.x + i * (batt_w + 2))
            by = int(self.y - BASE_RADIUS - 6)
            pygame.draw.rect(surf, (38,38,48), (bx, by, batt_w, 12))
            pygame.draw.rect(surf, (18,18,26), (bx, by, batt_w, 12), 1)
        flag_x = int(self.x + BASE_RADIUS + 12)
        flag_y = int(self.y - BASE_RADIUS + 8)
        pygame.draw.rect(surf, (28,28,36), (flag_x, flag_y, 6, 32))
        pygame.draw.polygon(surf, (210,210,210), [(flag_x+6, flag_y+4), (flag_x+36, flag_y+12), (flag_x+6, flag_y+28)])
        lbl = font.render("Winterfell Keep", True, (235,235,235))
        surf.blit(lbl, (int(self.x - lbl.get_width()/2), int(self.y - BASE_RADIUS - 60)))
        bar_w, bar_h = 200, 16
        bx = self.x - bar_w/2; by = self.y - BASE_RADIUS - 36
        pygame.draw.rect(surf, (28,28,36), (bx-2, by-2, bar_w+4, bar_h+4), border_radius=6)
        pygame.draw.rect(surf, (150,150,150), (bx, by, bar_w, bar_h), border_radius=6)
        frac = clamp(self.hp / BASE_HP_MAX, 0.0, 1.0)
        pygame.draw.rect(surf, (90,200,120), (bx+2, by+2, int((bar_w-4) * frac), bar_h-4), border_radius=6)
        if frac < 0.25:
            pygame.draw.rect(surf, (220,40,40), (bx+2+int((bar_w-4)*frac), by+2, int((bar_w-4)*(0.25-frac)), bar_h-4), border_radius=4)

# -------------------------
# Snow particles
# -------------------------
class SnowParticle:
    def __init__(self):
        self.reset()
    def reset(self):
        self.x = random.uniform(0, SCREEN_W)
        self.y = random.uniform(-SCREEN_H, 0)
        self.speed = random.uniform(24, 100)
        self.size = random.uniform(1.2, 3.6)
        self.wind = random.uniform(-18, 18)
        self.alpha = random.uniform(80, 230)
    def step(self, dt):
        self.y += self.speed * dt
        self.x += self.wind * dt
        if self.y > SCREEN_H + 12 or self.x < -24 or self.x > SCREEN_W + 24:
            self.reset()

# -------------------------
# Sound engine (optional)
# -------------------------
class SoundEngine:
    def __init__(self, sounds_dir=SOUNDS_DIR):
        try:
            pygame.mixer.init()
        except Exception:
            pass
        self.sounds = {}
        self.ambience = None     # music / looping background
        self.ambience_playing = False
        self.sfx_enabled = True
        self.ambience_enabled = True

        for key, fname in SOUND_FILES.items():
            path = os.path.join(sounds_dir, fname)
            if os.path.isfile(path):
                try:
                    # treat "music" as looping ambience
                    if key == "music":
                        self.ambience = pygame.mixer.Sound(path)
                    else:
                        self.sounds[key] = pygame.mixer.Sound(path)
                except Exception:
                    self.sounds[key] = None

    def toggle_ambience(self):
        if not self.ambience: return
        if self.ambience_playing:
            try:
                self.ambience.stop()
            except Exception:
                pass
            self.ambience_playing = False
        else:
            try:
                self.ambience.play(loops=-1)
            except Exception:
                pass
            self.ambience_playing = True

    def set_sfx(self, on):
        self.sfx_enabled = bool(on)

# -------------------------
# Main engine
# -------------------------
def run():
    pygame.init()
    try:
        pygame.mixer.init()
    except Exception:
        pass

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Winterfell Defense v7.1 — Calm Northern Ambience")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Verdana", 16)
    small_font = pygame.font.SysFont("Arial", 12)

    base = Base(BASE_POS)
    soldiers = []
    enemies = deque()
    projectiles = []
    snow = [SnowParticle() for _ in range(SNOW_COUNT)]
    dead_soldiers = []

    # stats
    total_soldiers_deployed = 0
    soldiers_by_type = {"Archer": 0, "Footman": 0, "Ranger": 0}
    stats = {"kills": 0, "elite_kills": 0}

    # vignette & fog
    vignette = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    for i in range(int(VIGNETTE_STRONG/2)):
        alpha = int(160 * (i/(VIGNETTE_STRONG/2))**1.2)
        pygame.draw.rect(vignette, (8, 14, 22, alpha), (i, i, SCREEN_W-2*i, SCREEN_H-2*i), 1)
    fog = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    fog.fill((160,180,200, FOG_ALPHA))

    # Sound engine (loads 'music' as ambience -> base.wav)
    try:
        sound_engine = SoundEngine()
        if sound_engine.ambience and sound_engine.ambience_enabled:
            try:
                sound_engine.ambience.play(loops=-1)
                sound_engine.ambience_playing = True
            except Exception:
                pass
    except Exception:
        sound_engine = type("SE", (), {"sfx_enabled": False, "sounds": {}, "ambience": None, "ambience_playing": False, "toggle_ambience": lambda self=None: None, "set_sfx": lambda self, on=None: None})()

    # spawning
    spawn_timer = 0.0
    spawn_interval = SPAWN_BASE_RATE
    runtime = 0.0

    # auto-deploy
    auto_deploy_timer = 0.0

    show_grid = False
    show_snow = True
    paused = False
    running = True

    # game over
    game_over = False

    def spawn_one():
        nonlocal enemies, spawn_interval, runtime
        key = random.randrange(NUM_PATHS)
        delay = random.uniform(0.0, 0.2)
        elite_prob = ELITE_CHANCE_BASE + runtime * SPAWN_ELITE_SCALE
        elite = random.random() < elite_prob
        enemies.append(Enemy(key, delay=delay, elite=elite))

    # initial burst
    for _ in range(12):
        spawn_one()

    while running:
        dt = clock.tick(FPS) / 1000.0
        runtime += dt
        spawn_timer += dt
        auto_deploy_timer += dt
        spawn_interval = max(SPAWN_MIN, SPAWN_BASE_RATE - runtime * SPAWN_ACCEL)

        # events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE and not game_over:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    # restart everything
                    soldiers.clear(); enemies.clear(); projectiles.clear(); dead_soldiers.clear()
                    total_soldiers_deployed = 0
                    soldiers_by_type = {"Archer": 0, "Footman": 0, "Ranger": 0}
                    stats = {"kills": 0, "elite_kills": 0}
                    base.hp = BASE_HP_MAX
                    runtime = 0.0; spawn_interval = SPAWN_BASE_RATE; spawn_timer = 0.0; auto_deploy_timer = 0.0
                    game_over = False
                    for _ in range(12): spawn_one()
                    paused = False
                elif ev.key == pygame.K_g:
                    show_grid = not show_grid
                elif ev.key == pygame.K_s:
                    show_snow = not show_snow
                elif ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_m:
                    # toggle background music (base.wav)
                    if hasattr(sound_engine, "toggle_ambience"):
                        sound_engine.toggle_ambience()
                elif ev.key == pygame.K_v:
                    if hasattr(sound_engine, "set_sfx"):
                        sound_engine.set_sfx(not sound_engine.sfx_enabled)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and not game_over:
                mx,my = ev.pos
                kind = random.choice(list(SOLDIER_TYPES.keys()))
                ang = math.atan2(my - BASE_POS[1], mx - BASE_POS[0])
                soldiers.append(Soldier((mx,my), kind=kind, spawn_angle=ang))
                total_soldiers_deployed += 1
                soldiers_by_type[kind] += 1

        if not paused and not game_over:
            # spawn continuous
            while spawn_timer >= spawn_interval:
                spawn_one()
                spawn_timer -= spawn_interval

            if show_snow:
                for p in snow: p.step(dt)

            # update enemies
            for e in list(enemies):
                e.step(dt)
                if not e.alive and getattr(e, "reached", False):
                    base.hp -= (3 if e.is_elite else 1)
                    if hasattr(sound_engine, "sfx_enabled") and sound_engine.sfx_enabled and sound_engine.sounds.get("base"):
                        # base damage sfx (not the looping music)
                        try:
                            sound_engine.sounds["base"].play()
                        except Exception:
                            pass
                    try: enemies.remove(e)
                    except ValueError: pass
                elif not e.alive:
                    # killed by soldiers/projectiles
                    try:
                        enemies.remove(e)
                    except ValueError:
                        pass

            # auto-deploy logic (respect troop limit)
            danger_key = None
            for e in enemies:
                if e.spawn_delay <= 0 and e.alive:
                    ex,ey = e.pos()
                    if dist((ex,ey), BASE_POS) <= AUTO_DEPLOY_RANGE:
                        danger_key = e.path_key
                        break

            if (danger_key is not None and auto_deploy_timer >= AUTO_DEPLOY_COOLDOWN
                and len(soldiers) < MAX_ACTIVE_SOLDIERS):
                auto_deploy_timer = 0.0
                start_point = PATH_SEGMENTS[danger_key][0]['a']
                ang = math.atan2(start_point[1] - BASE_POS[1], start_point[0] - BASE_POS[0])
                spawn_dist = BASE_RADIUS + 14
                sx = BASE_POS[0] + math.cos(ang) * spawn_dist
                sy = BASE_POS[1] + math.sin(ang) * spawn_dist
                kind = random.choice(list(SOLDIER_TYPES.keys()))
                soldiers.append(Soldier((sx, sy), kind=kind, spawn_angle=ang))
                total_soldiers_deployed += 1
                soldiers_by_type[kind] += 1
                if hasattr(sound_engine, "sfx_enabled") and sound_engine.sfx_enabled and sound_engine.sounds.get("deploy"):
                    try:
                        sound_engine.sounds["deploy"].play()
                    except Exception:
                        pass

            # soldiers step (AI)
            for s in list(soldiers):
                died, dx, dy, dcolor = s.step(dt, list(enemies), projectiles, sound_engine, stats)
                if died:
                    dead_soldiers.append(DeadSoldier(dx, dy, dcolor))
                    try:
                        soldiers.remove(s)
                    except ValueError:
                        pass

            # update dead soldier ragdolls
            for ds in list(dead_soldiers):
                if ds.step(dt):
                    try:
                        dead_soldiers.remove(ds)
                    except ValueError:
                        pass

            # projectiles step
            for p in list(projectiles):
                if p.step(dt, sound_engine, stats):
                    try: projectiles.remove(p)
                    except ValueError: pass

            # update stats counters (sync)
            kill_count = stats["kills"]
            elite_kills = stats["elite_kills"]

            if base.hp <= 0:
                game_over = True
                paused = True

        # --- drawing
        screen.fill((18,22,30))
        bg = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            v = int(20 + (y/SCREEN_H)*28)
            bg.fill((v+6, v+12, v+18), (0,y,SCREEN_W,1))
        screen.blit(bg, (0,0))

        # faint speckles
        speck = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for i in range(60):
            sx = (i * 173) % SCREEN_W
            sy = (i * 97) % (SCREEN_H//3)
            pygame.draw.circle(speck, (255,255,255,18), (sx, sy+30), 1)
        screen.blit(speck, (0,0))

        # fog behind objects
        screen.blit(fog.copy(), (0,0))

        # grid overlay
        if show_grid:
            grid = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            for x in range(0, SCREEN_W, GRID_SIZE):
                pygame.draw.line(grid, (100,110,120,40), (x,0), (x,SCREEN_H))
            for y in range(0, SCREEN_H, GRID_SIZE):
                pygame.draw.line(grid, (100,110,120,40), (0,y), (SCREEN_W,y))
            screen.blit(grid, (0,0))

        # draw dead ragdolls (behind active)
        for ds in dead_soldiers:
            ds.draw(screen)

        # soldiers
        for s in soldiers:
            s.draw(screen, font)

        # projectiles
        for p in projectiles:
            age_frac = max(0.0, 1.0 - p.age / p.life)
            ex, ey = p.x, p.y
            end_x = ex - p.dirx * 10
            end_y = ey - p.diry * 10
            thickness = int(2 + 4*age_frac)
            pygame.draw.line(screen, PROJECTILE_COLOR, (ex,ey), (end_x,end_y), thickness)

        # enemies
        sorted_enemies = sorted(list(enemies), key=lambda e: -e.s)
        for e in sorted_enemies:
            if e.spawn_delay > 0:
                continue
            x,y = e.pos()
            aura = pygame.Surface((ENEMY_RADIUS*6, ENEMY_RADIUS*6), pygame.SRCALPHA)
            c = (ENEMY_AURA[0], ENEMY_AURA[1], ENEMY_AURA[2], ENEMY_AURA[3])
            pygame.draw.circle(aura, c, (ENEMY_RADIUS*3, ENEMY_RADIUS*3), ENEMY_RADIUS*3)
            screen.blit(aura, (int(x-ENEMY_RADIUS*3), int(y-ENEMY_RADIUS*3)), special_flags=pygame.BLEND_RGBA_ADD)
            color = ELITE_COLOR if e.is_elite else ENEMY_COLOR
            pygame.draw.circle(screen, color, (int(x),int(y)), ENEMY_RADIUS)
            pygame.draw.circle(screen, (24,24,24), (int(x),int(y)), ENEMY_RADIUS, 2)
            pygame.draw.circle(screen, (255,255,255), (int(x+3), int(y-4)), 3)
            lbl = small_font.render(e.label(), True, (240,240,240))
            screen.blit(lbl, (int(x - lbl.get_width()/2), int(y - ENEMY_RADIUS - 24)))
            bw, bh = 38, 6
            bx = x - bw/2; by = y - ENEMY_RADIUS - 12
            pygame.draw.rect(screen, (50,50,50), (bx, by, bw, bh), border_radius=3)
            frac = clamp(e.hp / (ENEMY_HP * (ELITE_MULT_HP if e.is_elite else 1.0)), 0.0, 1.0)
            pygame.draw.rect(screen, (80,200,120), (bx+2, by+1, int((bw-4)*frac), bh-2), border_radius=3)

        # base
        base.draw(screen, font)

        # snow foreground
        if show_snow:
            for p in snow:
                pygame.draw.circle(screen, (255,255,255, int(p.alpha)), (int(p.x), int(p.y)), int(p.size))

        # vignette
        screen.blit(vignette, (0,0))

        # HUD: active enemies, wall hp, spawn rate
        active = len([e for e in enemies if e.spawn_delay <= 0])
        hud = font.render(f"Active Wights: {active}   Wall Integrity: {base.hp}   Spawn interval: {spawn_interval:.2f}s", True, (220,220,220))
        screen.blit(hud, (14,14))

        # soldiers deployed stats
        soldier_hud = font.render(
            f"Soldiers Deployed: {total_soldiers_deployed}  (A:{soldiers_by_type['Archer']}  F:{soldiers_by_type['Footman']}  R:{soldiers_by_type['Ranger']})",
            True, (220,220,200)
        )
        screen.blit(soldier_hud, (14, 40))

        hints = font.render("Click to place soldiers • SPACE pause • R reset • G grid • S snow • M music • V sfx • Q quit", True, (180,180,200))
        screen.blit(hints, (12, SCREEN_H - 26))

        # Game Over Summary overlay
        if game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            screen.blit(overlay, (0,0))
            big = pygame.font.SysFont("Georgia", 48, bold=True)
            msg = big.render("WINTERFELL HAS FALLEN", True, (255,220,220))
            screen.blit(msg, (SCREEN_W/2 - msg.get_width()/2, SCREEN_H/2 - 200))
            mid = pygame.font.SysFont("Arial", 26)
            lines = [
                f"Time Survived: {runtime:.1f} seconds",
                f"Soldiers Deployed: {total_soldiers_deployed}",
                f"  Archers: {soldiers_by_type['Archer']}",
                f"  Footmen: {soldiers_by_type['Footman']}",
                f"  Rangers: {soldiers_by_type['Ranger']}",
                "",
                f"Enemies Killed: {stats['kills']}",
                f"Elite Wights Killed: {stats['elite_kills']}",
                "",
                "Press R to Restart • Press Q to Quit"
            ]
            y = SCREEN_H/2 - 100
            for line in lines:
                txt = mid.render(line, True, (240,240,240))
                screen.blit(txt, (SCREEN_W/2 - txt.get_width()/2, y))
                y += 34

        pygame.display.flip()

    # stop ambience gracefully
    try:
        if sound_engine and getattr(sound_engine, "ambience_playing", False):
            sound_engine.ambience.stop()
    except Exception:
        pass

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run()
