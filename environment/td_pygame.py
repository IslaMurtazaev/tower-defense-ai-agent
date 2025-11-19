#!/usr/bin/env python3
"""
td_winterfell_final.py
Final Winterfell Defense — Single-file engine (Option A, final confirmed)

Features implemented (as requested):
- Soldiers attack nearest enemy (wight or Night King). They do 0 damage to Night Kings.
- Soldiers kill ordinary wights instantly on contact (burn animation 0.1s).
- Night Kings spawn in scheduled waves (5 total). Each NK walks in, becomes visible
  and engages when close enough. NK performs a sweeping AOE attack every 1.5s that
  instantly kills soldiers in the sweep radius.
- Jon Snow and Daenerys are the ONLY units that can damage Night Kings.
- After a Night King dies:
    - NEXT_NK_DELAY += 12.0
    - Soldiers continue normal behavior (no freezing/retreat)
    - Jon/Daenerys are flagged to return (visual halo)
    - Bran warning effect stops
- No battle log UI; removed bulky notification panels.
- HUD shows: Soldiers Alive, Soldiers Killed, Soldiers Deployed, Wights Killed, NK Remaining.
- Bran the Seer gives a raven/vision overlay when a NK is approaching.
- Burn animation for wights is 0.1 seconds.
- Sound system is optional: looks for sounds/ folder but runs without them.
- All behaviors and rules match the final specification from the user.

Place this file next to an optional `sounds/` folder (handled gracefully if absent).
"""

import pygame, sys, os, math, random, time
from collections import deque

# -------------------------
# CONFIGURATION
# -------------------------
SCREEN_W, SCREEN_H = 1280, 800
FPS = 60

# Enemy (wights)
ENEMY_RADIUS = 12
ENEMY_COLOR = (220, 70, 70)
ENEMY_SPEED = 70.0
ENEMY_HP = 1  # unused - instant burn on contact
ENEMY_JITTER = 2.6

# Night King parameters
NK_RADIUS = 40
NK_COLOR = (160, 230, 255)
NK_AURA = (150, 230, 255, 110)
NK_SPEED = 28.0
NK_HP_BASE = 320
NK_ENGAGE_RADIUS = 340.0
NK_SWEEP_INTERVAL = 1.5
NK_SWEEP_RADIUS_CHOICE = {
    "A": 60,
    "B": 95,
    "C": 130
}
# The user previously chose cinematic; we'll go with Medium default unless adjusted.
NK_SWEEP_RADIUS = NK_SWEEP_RADIUS_CHOICE["B"]

# Soldier parameters
SOLDIER_RADIUS = 12
SOLDIER_COLOR = (80, 100, 130)
SOLDIER_SPEED = 160.0
SOLDIER_MAX = 32
SOLDIER_ATTACK_RANGE = 30  # contact range
SOLDIER_BURN_DURATION = 0.1  # seconds (instant burn animation)
SOLDIER_VISION_RADIUS = 200.0  # limited vision range

# Heroes
JON_RADIUS = 18
JON_SPEED = 260.0
JON_DAMAGE = 60.0
JON_COOLDOWN = 0.9

DAEN_SPEED = 320.0
DRAGON_BREATH_DPS = 160.0
DRAGON_BREATH_RANGE = 300.0

# Base
BASE_POS = (SCREEN_W//2, SCREEN_H//2)
BASE_RADIUS = 56
BASE_HP_MAX = 80

# Visuals & spawn
SNOW_BASE = 110
SNOW_BOOST = 420
FOG_ALPHA = 70
GRID_SIZE = 28
NUM_PATHS = 24

# Spawn configuration
SPAWN_BASE_RATE = 0.9
SPAWN_MIN = 0.16
SPAWN_ACCEL = 0.00035

# Auto-deploy (we keep auto-deploy but soldiers no longer retreat; auto deploy just spawns soldiers)
AUTO_DEPLOY_RANGE = 520.0
AUTO_DEPLOY_COOLDOWN = 2.2

# Night King scheduling (5 waves). We'll schedule these times in seconds.
NK_SCHEDULE_TIMES = [45, 120, 210, 320, 480]
NK_RESPAWN_COOLDOWN = 16.0
NEXT_NK_DELAY = NK_RESPAWN_COOLDOWN  # will be increased by +12 after each NK death.

# Bran
BRAN_LEAD_TIME = 10.0

# Sound files
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
SOUND_FILES = {
    "music": "base.wav",
    "bran": "bran_vision.wav",
    "deploy": "war_horn.wav",
    "shot": "arrow_shot.wav",
    "hit": "enemy_hit.wav",
    "die": "enemy_die.wav",
    "base": "base_hit.wav",
    "breath": "dragon_breath.wav",
    "jon": "jon_swing.wav",
    "shock": "ice_shock.wav",
    "victory": "victory.wav"
}

# -------------------------
# UTILITIES
# -------------------------
def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def lerp(a,b,t): return a + (b-a)*t
def clamp(x,a,b): return max(a, min(b, x))

# -------------------------
# PATHS (radial hidden paths)
# -------------------------
def make_paths(num_paths=NUM_PATHS, base=BASE_POS, outer_margin=60, mid_dist=300):
    cx, cy = base
    paths = {}
    max_dim = max(SCREEN_W, SCREEN_H)
    for i in range(num_paths):
        angle = 2*math.pi*(i/num_paths)
        sx = cx + math.cos(angle)*(max_dim*0.9 + outer_margin)
        sy = cy + math.sin(angle)*(max_dim*0.9 + outer_margin)
        mx = cx + math.cos(angle)*mid_dist + random.uniform(-20,20)
        my = cy + math.sin(angle)*mid_dist + random.uniform(-20,20)
        paths[i] = [(sx,sy),(mx,my),base]
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
            t = rem / seg['len'] if seg['len'] > 0 else 0.0
            return (lerp(seg['a'][0], seg['b'][0], t), lerp(seg['a'][1], seg['b'][1], t))
        rem -= seg['len']
    return segs[-1]['b']

# -------------------------
# SOUND ENGINE
# -------------------------
class SoundEngine:
    def __init__(self, sounds_dir=SOUNDS_DIR):
        try:
            pygame.mixer.init()
        except Exception:
            pass
        self.sounds = {}
        self.ambience = None
        self.ambience_playing = False
        self.sfx_enabled = True
        self.ambience_enabled = True
        for key, fname in SOUND_FILES.items():
            path = os.path.join(sounds_dir, fname)
            if os.path.isfile(path):
                try:
                    if key == "music":
                        self.ambience = pygame.mixer.Sound(path)
                    else:
                        self.sounds[key] = pygame.mixer.Sound(path)
                except Exception:
                    self.sounds[key] = None
    def toggle_ambience(self):
        if not self.ambience: return
        if self.ambience_playing:
            try: self.ambience.stop()
            except: pass
            self.ambience_playing = False
        else:
            try: self.ambience.play(loops=-1)
            except: pass
            self.ambience_playing = True
    def set_sfx(self, on):
        self.sfx_enabled = bool(on)

# -------------------------
# ENTITIES
# -------------------------
class Enemy:
    def __init__(self, path_key, delay=0.0):
        self.path_key = path_key
        self.s = 0.0
        self.speed = ENEMY_SPEED * (0.9 + random.uniform(-0.1,0.1))
        self.hp = ENEMY_HP
        self.spawn_delay = delay
        self.alive = True
        self.reached = False
        self.phase = random.uniform(0, 2*math.pi)
        self.is_nk = False
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
        jx = math.sin(time.time()*1.9 + self.phase) * ENEMY_JITTER
        jy = math.cos(time.time()*1.6 + self.phase) * (ENEMY_JITTER*0.5)
        return (x + jx + random.uniform(-0.5,0.5), y + jy + random.uniform(-0.5,0.5))

class NightKing(Enemy):
    def __init__(self, path_key, spawn_fraction, idx):
        super().__init__(path_key, delay=0.0)
        self.is_nk = True
        self.index = idx
        # spawn closer to edge than a normal enemy; position along path using spawn_fraction
        self.s = PATH_TOTAL[self.path_key] * spawn_fraction
        self.speed = NK_SPEED * (0.98 + random.uniform(-0.03, 0.03))
        self.hp = int(NK_HP_BASE * (1.0 + 0.08 * idx))
        self.locked_for_battle = False
        self.sweep_cooldown = NK_SWEEP_INTERVAL * (0.9 + random.uniform(-0.05, 0.05))
        self.death_processed = False
    def step(self, dt):
        if not self.alive:
            return None
        if self.spawn_delay > 0:
            self.spawn_delay -= dt
            return None
        # move in until engaged
        if not self.locked_for_battle:
            self.s += self.speed * dt
            px,py = pos_on_path_key(self.path_key, self.s)
            # lock when within engage radius of base
            if dist((px,py), BASE_POS) <= NK_ENGAGE_RADIUS:
                self.locked_for_battle = True
        else:
            # when locked, continue to station near base (advance slowly)
            self.s += (self.speed * 0.4) * dt
        # sweep cooldown countdown & trigger if ready
        self.sweep_cooldown -= dt
        if self.sweep_cooldown <= 0 and self.alive and self.locked_for_battle:
            # reset
            self.sweep_cooldown = NK_SWEEP_INTERVAL
            return "sweep"
        return None
    def pos(self):
        x,y = pos_on_path_key(self.path_key, self.s)
        jx = math.sin(time.time()*0.75 + self.phase) * (ENEMY_JITTER*0.6)
        jy = math.cos(time.time()*0.6 + self.phase) * (ENEMY_JITTER*0.5)
        return (x + jx, y + jy)

class BurningEffect:
    def __init__(self, x, y):
        self.x = x; self.y = y
        self.age = 0.0
        self.life = SOLDIER_BURN_DURATION
    def step(self, dt):
        self.age += dt
        return self.age >= self.life
    def draw(self, surf):
        frac = clamp(1.0 - self.age/self.life, 0.0, 1.0)
        rad = int(20 * frac) + 4
        core = pygame.Surface((rad*2+4, rad*2+4), pygame.SRCALPHA)
        core_alpha = int(220 * frac)
        pygame.draw.circle(core, (255,200,80, core_alpha), (rad+2, rad+2), int(rad*0.5))
        pygame.draw.circle(core, (255,120,24, int(200*frac)), (rad+2, rad+2), rad)
        surf.blit(core, (int(self.x-rad-2), int(self.y-rad-2)))

class DeadSoldier:
    def __init__(self, x, y, color):
        self.x = x; self.y = y; self.color = color; self.age = 0.0; self.life = 1.6; self.fade = 255
    def step(self, dt):
        self.age += dt
        self.fade = max(0, int(255 * (1 - self.age/self.life)))
        self.y += 10 * dt
        return self.age >= self.life
    def draw(self, surf):
        s = pygame.Surface((44,44), pygame.SRCALPHA)
        pygame.draw.circle(s, (self.color[0], self.color[1], self.color[2], self.fade), (22,22), 14)
        surf.blit(s, (self.x-22, self.y-22))

class Soldier:
    def __init__(self, pos, spawn_angle=None):
        self.x, self.y = pos
        self.original_x, self.original_y = pos  # Store original placement position
        self.phase = random.uniform(0,2*math.pi)
        self.color = SOLDIER_COLOR
        self.spawn_angle = spawn_angle if spawn_angle is not None else random.uniform(0,2*math.pi)
        self.speed = SOLDIER_SPEED
        self.cooldown = 0.0
        # Note: Soldiers have limited vision and return to original position when no enemies in range.
    def find_nearest_enemy(self, enemies):
        # Only consider enemies within vision radius
        alive = [e for e in enemies if e.alive and e.spawn_delay <= 0]
        in_vision = [e for e in alive if dist((self.x, self.y), e.pos()) <= SOLDIER_VISION_RADIUS]
        if not in_vision: return None
        in_vision.sort(key=lambda e: dist((self.x,self.y), e.pos()))
        return in_vision[0]
    def move_toward(self, tx, ty, dt, factor=1.0):
        dx = tx - self.x; dy = ty - self.y
        d = math.hypot(dx,dy)
        if d > 1e-3:
            nx, ny = dx/d, dy/d
            self.x += nx * self.speed * dt * factor
            self.y += ny * self.speed * dt * factor
    def step(self, dt, enemies, burns, sound_engine, stats):
        # Look for nearest enemy within vision radius
        if self.cooldown > 0:
            self.cooldown -= dt
        target = self.find_nearest_enemy(enemies)
        if target is None:
            # No enemies in vision - return to original placement position
            d_to_origin = dist((self.x, self.y), (self.original_x, self.original_y))
            if d_to_origin > 5:  # Only move if not already at original position
                self.move_toward(self.original_x, self.original_y, dt, factor=0.5)
            return (False, None, None, None)
        tx, ty = target.pos()
        d = dist((self.x,self.y), (tx,ty))
        # If it's a wight and contact range -> instant burn-kill
        if not getattr(target, "is_nk", False):
            if d <= (SOLDIER_ATTACK_RANGE + ENEMY_RADIUS):
                # kill wight instantly
                burns.append(BurningEffect(tx, ty))
                target.alive = False
                stats['wights_killed'] += 1
                if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("die"):
                    try: sound_engine.sounds["die"].play()
                    except: pass
                return (False, None, None, None)
            else:
                # approach
                self.move_toward(tx, ty, dt)
                return (False, None, None, None)
        else:
            # target is a Night King: soldiers will still approach and attempt to hit (but deal 0)
            if d <= (SOLDIER_ATTACK_RANGE + NK_RADIUS):
                # "attack" - but does 0 damage; remain in place (hero must finish NK)
                # make a small recoil or visual hit (no hp effect)
                if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("hit"):
                    # play a faint hit sound to indicate useless strikes (if available)
                    try: sound_engine.sounds["hit"].play()
                    except: pass
                # do not change NK hp
                return (False, None, None, None)
            else:
                self.move_toward(tx, ty, dt)
                return (False, None, None, None)
    def draw(self, surf, font):
        pygame.draw.circle(surf, (22,22,26), (int(self.x), int(self.y)), SOLDIER_RADIUS+4)
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), SOLDIER_RADIUS)
        # sword indicator
        pygame.draw.rect(surf, (200,200,200), (int(self.x+8), int(self.y-6), 6, 10))

class Jon:
    def __init__(self, spawn_angle):
        self.x = BASE_POS[0] + math.cos(spawn_angle)*(BASE_RADIUS+58)
        self.y = BASE_POS[1] + math.sin(spawn_angle)*(BASE_RADIUS+58)
        self.target = None
        self.speed = JON_SPEED
        self.cool = 0.0
        self.swinging = False
        self.swing_prog = 0.0
        self.returning = False
        self.active = False
    def deploy_for(self, nk):
        self.target = nk
        self.active = True
        self.returning = False
    def step(self, dt, stats, sound_engine):
        if not self.active:
            return
        if (not self.target) or (not self.target.alive):
            # return to keep
            self.returning = True
            dx = BASE_POS[0] - self.x; dy = BASE_POS[1] - self.y
            d = math.hypot(dx,dy)
            if d > 12:
                nx, ny = dx/d, dy/d
                self.x += nx * self.speed * dt
                self.y += ny * self.speed * dt
            else:
                self.active = False
                self.returning = False
            return
        tx, ty = self.target.pos(); dx = tx - self.x; dy = ty - self.y
        d = math.hypot(dx,dy)
        # approach
        if d > 22:
            nx, ny = dx/d, dy/d
            self.x += nx * self.speed * dt
            self.y += ny * self.speed * dt
            self.swinging = False
            self.swing_prog = 0.0
        else:
            # hit if cooldown allows
            if self.cool <= 0:
                # strike
                self.swinging = True
                self.swing_prog = 0.0
                self.target.hp -= JON_DAMAGE
                if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("jon"):
                    try: sound_engine.sounds["jon"].play()
                    except: pass
                self.cool = JON_COOLDOWN
                if self.target.hp <= 0:
                    self.target.alive = False
                    stats['nk_kills'] += 1
            else:
                self.cool -= dt
            if self.swinging:
                self.swing_prog += dt / 0.35
                if self.swing_prog >= 1.0:
                    self.swinging = False
                    self.swing_prog = 0.0
    def draw(self, surf):
        cx, cy = int(self.x), int(self.y)
        # cloak
        pygame.draw.polygon(surf, (12,12,14), [(cx-16, cy+20),(cx, cy-28),(cx+16, cy+20)])
        pygame.draw.circle(surf, (255,255,255), (cx, cy-6), 6)
        pygame.draw.circle(surf, (28,28,30), (cx, cy+4), JON_RADIUS)
        # sword longclaw as bright line
        pygame.draw.line(surf, (220,220,240), (cx+8, cy-4), (cx+34, cy-12), 6)
        if self.returning:
            halo = pygame.Surface((80,80), pygame.SRCALPHA)
            pygame.draw.circle(halo, (180,180,255,50), (40,40), 36)
            surf.blit(halo, (cx-40, cy-40))

class Daenerys:
    def __init__(self, spawn_angle):
        self.x = BASE_POS[0] + math.cos(spawn_angle)*(BASE_RADIUS+28)
        self.y = BASE_POS[1] + math.sin(spawn_angle)*(BASE_RADIUS+28) - 100
        self.target = None
        self.speed = DAEN_SPEED
        self.breathing = False
        self.returning = False
        self.active = False
    def deploy_for(self, nk):
        self.target = nk
        self.active = True
        self.returning = False
    def step(self, dt, sound_engine, stats):
        if not self.active:
            return
        if (not self.target) or (not self.target.alive):
            self.returning = True
            dx = BASE_POS[0] - self.x; dy = BASE_POS[1] - self.y
            d = math.hypot(dx,dy)
            if d > 12:
                nx, ny = dx/d, dy/d
                self.x += nx * self.speed * dt
                self.y += ny * self.speed * dt
            else:
                self.active = False
                self.returning = False
            return
        tx, ty = self.target.pos(); dx = tx - self.x; dy = ty - self.y
        d = math.hypot(dx,dy)
        desired = DRAGON_BREATH_RANGE * 0.7
        if d > desired:
            nx, ny = dx/d, dy/d
            self.x += nx * self.speed * dt
            self.y += ny * self.speed * dt
            self.breathing = False
        else:
            self.breathing = True
            self.target.hp -= DRAGON_BREATH_DPS * dt
            if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("breath"):
                if random.random() < 0.05:
                    try: sound_engine.sounds["breath"].play()
                    except: pass
            if self.target.hp <= 0:
                self.target.alive = False
                stats['nk_kills'] += 1
    def draw(self, surf):
        # dragon silhouette + rider
        pygame.draw.polygon(surf, (190,140,120), [(int(self.x-48), int(self.y+8)), (int(self.x-20), int(self.y-54)), (int(self.x), int(self.y))])
        pygame.draw.circle(surf, (255,240,200), (int(self.x+6), int(self.y-8)), 5)
        if self.returning:
            halo = pygame.Surface((90,90), pygame.SRCALPHA)
            pygame.draw.circle(halo, (180,180,255,50), (45,45), 40)
            surf.blit(halo, (int(self.x-45), int(self.y-45)))

# -------------------------
# Bran vision (simplified)
# -------------------------
class BranVision:
    def __init__(self):
        self.active = False
        self.timer = 0.0
        self.duration = 3.0
        self.angle = None
    def start(self, angle):
        self.active = True
        self.timer = self.duration
        self.angle = angle
    def step(self, dt):
        if not self.active:
            return
        self.timer -= dt
        if self.timer <= 0:
            self.active = False
            self.angle = None
    def draw(self, surf, font):
        if not self.active: return
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        alpha = int(160 * (self.timer / self.duration))
        overlay.fill((80,120,160, alpha))
        surf.blit(overlay, (0,0))
        # raven simple
        xs = 60; ys = 60
        for i in range(5):
            rx = SCREEN_W - 120 - i*30
            ry = 80 + (i%2)*10
            pygame.draw.polygon(surf, (20,20,20), [(rx,ry),(rx+10, ry+4),(rx+18, ry)])
        # small text above bran area
        txt = font.render("Bran senses a great evil...", True, (240,240,255))
        surf.blit(txt, (14, SCREEN_H - 86))

# -------------------------
# MAIN ENGINE
# -------------------------
def run():
    global NEXT_NK_DELAY
    pygame.init()
    try:
        pygame.mixer.init()
    except:
        pass

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Winterfell Defense — Final")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Verdana", 16)
    small_font = pygame.font.SysFont("Arial", 12)
    big_font = pygame.font.SysFont("Georgia", 36, bold=True)

    # sound engine
    try:
        sound_engine = SoundEngine()
        if sound_engine.ambience and sound_engine.ambience_enabled:
            try: sound_engine.ambience.play(loops=-1); sound_engine.ambience_playing = True
            except: pass
    except:
        sound_engine = type("SE", (), {"sfx_enabled": False, "sounds": {}, "ambience": None, "ambience_playing": False, "toggle_ambience": lambda self=None: None, "set_sfx": lambda self, on=None: None})()

    # game state
    base_hp = BASE_HP_MAX
    soldiers = []
    enemies = deque()
    nks = []
    heroes = []
    burns = []
    dead_soldiers = []

    # snow
    snow = [ [random.uniform(0, SCREEN_W), random.uniform(-SCREEN_H,0), random.uniform(20,120), random.uniform(1.2,3.8), random.uniform(-18,18), random.uniform(100,240)] for _ in range(SNOW_BASE) ]

    # stats & counters
    total_soldiers_deployed = 0
    soldiers_killed = 0
    wights_killed = 0
    nk_kills = 0

    stats = {'wights_killed': wights_killed, 'soldiers_killed': soldiers_killed, 'nk_kills': nk_kills}

    # schedule NKs
    nk_schedule = []
    for idx, t in enumerate(NK_SCHEDULE_TIMES):
        path_key = random.randrange(NUM_PATHS)
        warn_time = max(0.0, t - BRAN_LEAD_TIME)
        spawn_frac = 0.02 + random.uniform(0.0, 0.06)
        nk_schedule.append({'index': idx, 'spawn_time': t, 'warn_time': warn_time, 'path_key': path_key, 'spawn_frac': spawn_frac, 'warned': False, 'spawned': False})

    # spawn initial wights
    def spawn_wight():
        key = random.randrange(NUM_PATHS)
        enemies.append(Enemy(key, delay=random.uniform(0.0, 0.25)))

    for _ in range(12):
        spawn_wight()

    # bran vision
    bran = BranVision()

    # timers & runtime
    spawn_timer = 0.0
    spawn_interval = SPAWN_BASE_RATE
    runtime = 0.0
    auto_deploy_timer = 0.0
    nk_state = "idle"  # 'idle','active','cooldown'
    current_nk = None
    nk_cooldown_timer = 0.0
    NEXT_NK_DELAY = NK_RESPAWN_COOLDOWN

    show_grid = False
    show_snow = True
    paused = False
    running = True
    game_over = False
    victory = False

    START_TIME = time.time()

    # helper: find next scheduled unspawned NK
    def next_ns():
        for ns in nk_schedule:
            if not ns['spawned']:
                return ns
        return None

    # main loop
    while running:
        dt = clock.tick(FPS) / 1000.0
        runtime += dt
        spawn_timer += dt
        auto_deploy_timer += dt
        spawn_interval = max(SPAWN_MIN, SPAWN_BASE_RATE - runtime * SPAWN_ACCEL)

        # bran warnings and NK spawn schedule
        for ns in nk_schedule:
            if not ns['warned'] and runtime >= ns['warn_time']:
                ns['warned'] = True
                ang = (2*math.pi)*(ns['path_key'] / NUM_PATHS)
                bran.start(ang)
                # small sound if available
                if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("bran"):
                    try: sound_engine.sounds["bran"].play()
                    except: pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_over:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # reset everything
                    soldiers.clear(); enemies.clear(); nks.clear(); heroes.clear(); burns.clear(); dead_soldiers.clear()
                    snow = [ [random.uniform(0, SCREEN_W), random.uniform(-SCREEN_H,0), random.uniform(20,120), random.uniform(1.2,3.8), random.uniform(-18,18), random.uniform(100,240)] for _ in range(SNOW_BASE) ]
                    for _ in range(12):
                        spawn_wight()
                    total_soldiers_deployed = 0
                    stats = {'wights_killed':0, 'soldiers_killed':0, 'nk_kills':0}
                    base_hp = BASE_HP_MAX
                    runtime = 0.0
                    spawn_timer = 0.0
                    auto_deploy_timer = 0.0
                    nk_state = "idle"
                    current_nk = None
                    nk_cooldown_timer = 0.0
                    NEXT_NK_DELAY = NK_RESPAWN_COOLDOWN
                    for ns in nk_schedule:
                        ns['warned'] = False
                        ns['spawned'] = False
                    bran = BranVision()
                    paused = False
                    game_over = False
                    victory = False
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_s:
                    show_snow = not show_snow
                elif event.key == pygame.K_m:
                    if hasattr(sound_engine, "toggle_ambience"): sound_engine.toggle_ambience()
                elif event.key == pygame.K_v:
                    if hasattr(sound_engine, "set_sfx"): sound_engine.set_sfx(not sound_engine.sfx_enabled)
                elif event.key == pygame.K_q:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over:
                # Left click deploy soldier manually
                mx, my = event.pos
                if len(soldiers) < SOLDIER_MAX:
                    ang = math.atan2(my - BASE_POS[1], mx - BASE_POS[0])
                    s = Soldier((mx, my), spawn_angle=ang)
                    soldiers.append(s)
                    total_soldiers_deployed += 1

        bran.step(dt)

        if not paused and not game_over:
            # spawn next NK when appropriate
            ns = next_ns()
            if nk_state == "idle":
                if ns and runtime >= ns['spawn_time']:
                    # spawn NK
                    nk = NightKing(ns['path_key'], ns['spawn_frac'], ns['index'])
                    nks.append(nk)
                    ns['spawned'] = True
                    nk_state = "active"
                    current_nk = nk
                    # spawn an initial handful of wights to make it chaotic
                    for _ in range(6):
                        spawn_wight()
                    if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("deploy"):
                        try: sound_engine.sounds["deploy"].play()
                        except: pass
            elif nk_state == "active":
                # if NK is dead or has left (reached base), process and go to cooldown
                if current_nk and (not current_nk.alive or current_nk.reached):
                    # Inserted block after NK death/reach
                    if (not current_nk.alive) and (not getattr(current_nk, "death_processed", False)):
                        current_nk.death_processed = True
                        # Extend time before next Night King
                        NEXT_NK_DELAY += 12.0
                        # Soldiers already behave normally by design; ensure no special flags remain
                        for s in soldiers:
                            # no state attributes in current soldier design, but ensure speed normalized
                            s.speed = SOLDIER_SPEED
                        # Set heroes to return
                        for h in heroes:
                            if isinstance(h, Jon) or isinstance(h, Daenerys):
                                h.returning = True
                        # Bran vision stops
                        bran.active = False
                        bran.angle = None
                    # Clear current NK and go cooldown
                    current_nk = None
                    nk_state = "cooldown"
                    nk_cooldown_timer = 0.0

            elif nk_state == "cooldown":
                nk_cooldown_timer += dt
                if nk_cooldown_timer >= NEXT_NK_DELAY:
                    NEXT_NK_DELAY = NK_RESPAWN_COOLDOWN
                    nk_state = "idle"

            # spawn regular wights (fewer while NK active)
            if nk_state == "active":
                while spawn_timer >= spawn_interval * 2.2:
                    spawn_wight(); spawn_timer -= spawn_interval * 2.2
            else:
                while spawn_timer >= spawn_interval:
                    spawn_wight(); spawn_timer -= spawn_interval

            # auto-deploy soldiers if enemies near base (this just spawns extra soldiers)
            danger_key = None
            for e in enemies:
                if e.spawn_delay <= 0 and e.alive:
                    ex, ey = e.pos()
                    if dist((ex,ey), BASE_POS) <= AUTO_DEPLOY_RANGE:
                        danger_key = e.path_key
                        break
            if danger_key is not None and auto_deploy_timer >= AUTO_DEPLOY_COOLDOWN and len(soldiers) < SOLDIER_MAX:
                auto_deploy_timer = 0.0
                start_point = PATH_SEGMENTS[danger_key][0]['a']
                ang = math.atan2(start_point[1] - BASE_POS[1], start_point[0] - BASE_POS[0])
                spawn_dist = BASE_RADIUS + 36
                sx = BASE_POS[0] + math.cos(ang) * spawn_dist
                sy = BASE_POS[1] + math.sin(ang) * spawn_dist
                s = Soldier((sx, sy), spawn_angle=ang)
                soldiers.append(s)
                total_soldiers_deployed += 1

            # Update enemies and NKs
            # iterate copies because we may remove during iteration
            for e in list(list(enemies) + list(nks)):
                if getattr(e, "is_nk", False):
                    res = e.step(dt)
                    if res == "sweep":
                        # perform sweep: all soldiers within sweep radius die
                        kx, ky = e.pos()
                        for s in list(soldiers):
                            if dist((s.x,s.y), (kx,ky)) <= NK_SWEEP_RADIUS:
                                # kill soldier
                                try:
                                    soldiers.remove(s)
                                except ValueError:
                                    pass
                                dead_soldiers.append(DeadSoldier(s.x, s.y, s.color))
                                stats['soldiers_killed'] += 1
                                if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("shock"):
                                    try: sound_engine.sounds["shock"].play()
                                    except: pass
                    # if NK locked for battle and hero not yet deployed, deploy hero(s)
                    if e.locked_for_battle and e.alive:
                        hero_present = any((isinstance(h, Jon) and h.active and h.target is e) or (isinstance(h, Daenerys) and h.active and h.target is e) for h in heroes)
                        if not hero_present:
                            # decide hero by index parity (matching earlier behavior)
                            if e.index % 2 == 0:
                                new_jon = Jon(math.atan2(e.pos()[1] - BASE_POS[1], e.pos()[0] - BASE_POS[0]))
                                new_jon.deploy_for(e)
                                heroes.append(new_jon)
                            else:
                                new_daen = Daenerys(math.atan2(e.pos()[1] - BASE_POS[1], e.pos()[0] - BASE_POS[0]))
                                new_daen.deploy_for(e)
                                heroes.append(new_daen)
                else:
                    e.step(dt)
                # ordinary enemy death/reach handling
                if not getattr(e, "is_nk", False):
                    if not e.alive and getattr(e, "reached", False):
                        base_hp -= 1
                        if sound_engine and getattr(sound_engine, "sfx_enabled", False) and sound_engine.sounds.get("base"):
                            try: sound_engine.sounds["base"].play()
                            except: pass
                        try: enemies.remove(e)
                        except: pass
                    elif not e.alive:
                        # killed by soldier: remove and increment counters already handled in soldier
                        try: enemies.remove(e)
                        except: pass

            # soldiers update
            for s in list(soldiers):
                died, _, _, _ = s.step(dt, list(enemies) + list(nks), burns, sound_engine, stats)
                # death of soldier now handled inside NK sweep; nothing else to do here

            # update burn effects (wight death visuals)
            for b in list(burns):
                if b.step(dt):
                    try: burns.remove(b)
                    except: pass

            # hero update
            for h in list(heroes):
                if isinstance(h, Jon):
                    h.step(dt, stats, sound_engine)
                elif isinstance(h, Daenerys):
                    h.step(dt, sound_engine, stats)
                # remove inactive heroes that returned
                if hasattr(h, 'active') and not h.active:
                    try:
                        heroes.remove(h)
                    except: pass

            # dead soldier ragdolls update
            for ds in list(dead_soldiers):
                if ds.step(dt):
                    try: dead_soldiers.remove(ds)
                    except: pass

            # wight spawn growth - keep the battlefield busy (already handled with spawn_timer)
            if base_hp <= 0:
                game_over = True
                paused = True

            # victory check: all NKs spawned and all dead
            spawned_count = sum(1 for ns in nk_schedule if ns['spawned'])
            killed_count = sum(1 for ns in nk_schedule if ns['spawned'] and not any(k for k in nks if k.index == ns['index'] and k.alive))
            # simpler: consider victory when all scheduled times passed and no active NKs remain
            all_spawned = all(ns['spawned'] for ns in nk_schedule)
            active_nks = any(k for k in nks if k.alive)
            if all_spawned and not active_nks and all(ns['spawned'] for ns in nk_schedule):
                # ensure spawn finished
                if sum(1 for ns in nk_schedule if ns['spawned']) >= len(nk_schedule):
                    # win when all scheduled NKs have been killed (they are not in nks list)
                    # we check by counting how many NKs have been killed via stats
                    if stats.get('nk_kills', 0) >= len(nk_schedule):
                        game_over = True
                        paused = True
                        victory = True

        # DRAW
        screen.fill((18,20,28))
        # background gradient
        bg = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            v = int(18 + (y/SCREEN_H)*36)
            bg.fill((v+8, v+12, v+18), (0,y,SCREEN_W,1))
        screen.blit(bg, (0,0))

        # atmospheric speckles
        speck = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for i in range(60):
            sx = (i * 173) % SCREEN_W
            sy = (i * 97) % (SCREEN_H//3)
            pygame.draw.circle(speck, (255,255,255,14), (sx, sy+30), 1)
        screen.blit(speck, (0,0))

        # fog
        fog_layer = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        fog_layer.fill((160,180,200, FOG_ALPHA))
        screen.blit(fog_layer, (0,0))

        # optional grid
        if show_grid:
            grid_s = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            for x in range(0, SCREEN_W, GRID_SIZE):
                pygame.draw.line(grid_s, (100,110,120,40), (x,0), (x,SCREEN_H))
            for y in range(0, SCREEN_H, GRID_SIZE):
                pygame.draw.line(grid_s, (100,110,120,40), (0,y), (SCREEN_W,y))
            screen.blit(grid_s, (0,0))

        # draw dead soldier ragdolls (behind)
        for ds in dead_soldiers:
            ds.draw(screen)

        # draw soldiers
        for s in soldiers:
            s.draw(screen, font)

        # draw enemies (wights)
        for e in sorted(list(enemies), key=lambda z: -getattr(z,'s',0)):
            if e.spawn_delay > 0:
                continue
            x,y = e.pos()
            aura = pygame.Surface((ENEMY_RADIUS*5, ENEMY_RADIUS*5), pygame.SRCALPHA)
            pygame.draw.circle(aura, (255,120,120,64), (ENEMY_RADIUS*2, ENEMY_RADIUS*2), ENEMY_RADIUS*2)
            screen.blit(aura, (int(x-ENEMY_RADIUS*2), int(y-ENEMY_RADIUS*2)), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.circle(screen, ENEMY_COLOR, (int(x), int(y)), ENEMY_RADIUS)
            pygame.draw.circle(screen, (24,24,24), (int(x), int(y)), ENEMY_RADIUS, 2)
            pygame.draw.circle(screen, (255,255,255), (int(x+3), int(y-4)), 3)

        # draw Night Kings
        for nk in list(nks):
            if not nk.alive: continue
            x,y = nk.pos()
            # aura
            aura = pygame.Surface((NK_RADIUS*6, NK_RADIUS*6), pygame.SRCALPHA)
            pygame.draw.circle(aura, NK_AURA, (NK_RADIUS*3, NK_RADIUS*3), NK_RADIUS*3)
            screen.blit(aura, (int(x-NK_RADIUS*3), int(y-NK_RADIUS*3)), special_flags=pygame.BLEND_RGBA_ADD)
            pulse = 1.0 + 0.05*math.sin(time.time()*2.5 + nk.index)
            rcore = int(NK_RADIUS * pulse)
            pygame.draw.circle(screen, NK_COLOR, (int(x), int(y)), rcore)
            pygame.draw.circle(screen, (18,22,28), (int(x), int(y)), rcore, 3)
            pygame.draw.circle(screen, (255,255,255), (int(x+6), int(y-6)), 5)
            # label
            lfont = pygame.font.SysFont("Georgia", 18, bold=True)
            lbl = lfont.render("NIGHT KING", True, (200,240,255))
            screen.blit(lbl, (int(x - lbl.get_width()/2), int(y - NK_RADIUS - 26)))
            # HP bar
            bw, bh = 120, 12
            bx = x - bw/2; by = y - NK_RADIUS - 16
            pygame.draw.rect(screen, (20,20,26), (bx, by, bw, bh), border_radius=4)
            frac = clamp(nk.hp / (NK_HP_BASE * (1.0 + 0.08*nk.index)), 0.0, 1.0)
            pygame.draw.rect(screen, (160,220,255), (bx+4, by+3, int((bw-8)*frac), bh-6), border_radius=3)
            if nk.locked_for_battle:
                small = pygame.font.SysFont("Verdana", 14, bold=True)
                st = small.render("ENGAGED", True, (255,220,180))
                screen.blit(st, (int(x - st.get_width()/2), int(y + NK_RADIUS + 8)))

        # draw heroes
        for h in heroes:
            if isinstance(h, Daenerys):
                if getattr(h, "breathing", False) and h.target and h.target.alive:
                    tx,ty = h.target.pos()
                    beam = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                    pygame.draw.line(beam, (255,140,40,160), (int(h.x), int(h.y)), (int(tx), int(ty)), 22)
                    screen.blit(beam, (0,0))
                h.draw(screen)
                if h.active:
                    ttxt = small_font.render("Daenerys & Drogon", True, (240,220,200))
                    screen.blit(ttxt, (int(h.x - ttxt.get_width()/2), int(h.y - 48)))
            elif isinstance(h, Jon):
                h.draw(screen)
                if h.active:
                    ttxt = small_font.render("Jon Snow — Longclaw", True, (220,220,230))
                    screen.blit(ttxt, (int(h.x - ttxt.get_width()/2), int(h.y - 58)))

        # burn effects (wight death flashes)
        for b in burns:
            b.draw(screen)

        # draw base (on top)
        pygame.draw.circle(screen, (50,50,60), (BASE_POS[0], BASE_POS[1]), BASE_RADIUS)
        batt_w = 10
        for i in range(-3,4):
            bx = int(BASE_POS[0] + i*(batt_w+2)); by = int(BASE_POS[1] - BASE_RADIUS - 6)
            pygame.draw.rect(screen, (38,38,48), (bx,by,batt_w,12)); pygame.draw.rect(screen, (18,18,26), (bx,by,batt_w,12),1)
        # flag & label
        flag_x = int(BASE_POS[0] + BASE_RADIUS + 12); flag_y = int(BASE_POS[1] - BASE_RADIUS + 8)
        pygame.draw.rect(screen, (28,28,36), (flag_x, flag_y, 6, 32))
        pygame.draw.polygon(screen, (210,210,210), [(flag_x+6, flag_y+4), (flag_x+36, flag_y+12), (flag_x+6, flag_y+28)])
        lbl = font.render("Winterfell Keep", True, (235,235,235))
        screen.blit(lbl, (int(BASE_POS[0] - lbl.get_width()/2), int(BASE_POS[1] - BASE_RADIUS - 64)))
        bar_w, bar_h = 280, 16
        bx = BASE_POS[0] - bar_w/2; by = BASE_POS[1] - BASE_RADIUS - 44
        pygame.draw.rect(screen, (28,28,36), (bx-2, by-2, bar_w+4, bar_h+4), border_radius=6)
        pygame.draw.rect(screen, (150,150,150), (bx, by, bar_w, bar_h), border_radius=6)
        frac = clamp(base_hp / BASE_HP_MAX, 0.0, 1.0)
        pygame.draw.rect(screen, (90,200,120), (bx+4, by+4, int((bar_w-8) * frac), bar_h-8), border_radius=6)
        if frac < 0.25:
            pygame.draw.rect(screen, (220,40,40), (bx+4+int((bar_w-8)*frac), by+4, int((bar_w-8)*(0.25-frac)), bar_h-8), border_radius=4)

        # bran overlay
        bran.draw(screen, font)

        # snow intensity (more snow during NK active)
        snow_target = SNOW_BOOST if nk_state == "active" else SNOW_BASE
        while len(snow) < snow_target:
            snow.append([random.uniform(0, SCREEN_W), random.uniform(-SCREEN_H,0), random.uniform(20,120), random.uniform(1.2,3.8), random.uniform(-18,18), random.uniform(100,240)])
        while len(snow) > snow_target:
            snow.pop()
        if show_snow:
            for p in snow:
                p[1] += p[2] * dt * (1.0 + (0.6 if nk_state == "active" else 0.0))
                p[0] += p[4] * dt
                if p[1] > SCREEN_H + 12 or p[0] < -24 or p[0] > SCREEN_W + 24:
                    p[0] = random.uniform(0, SCREEN_W); p[1] = random.uniform(-SCREEN_H,0)
                pygame.draw.circle(screen, (255,255,255, int(p[5])), (int(p[0]), int(p[1])), int(p[3]))

        # vignette
        v = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for i in range(140):
            a = int(80 * (i/140)**1.2)
            pygame.draw.rect(v, (6,10,14,a), (i,i, SCREEN_W-2*i, SCREEN_H-2*i), 1)
        screen.blit(v, (0,0))

        # HUD
        active_enemies = len([e for e in enemies if e.spawn_delay <= 0]) + len([k for k in nks if k.alive])
        hud = font.render(f"Enemies: {active_enemies}   Wall Integrity: {base_hp}", True, (220,220,220))
        screen.blit(hud, (14, 12))
        soldier_hud = font.render(f"Soldiers Alive: {len(soldiers)}   Killed: {stats['soldiers_killed']}   Deployed: {total_soldiers_deployed}", True, (220,220,200))
        screen.blit(soldier_hud, (14, 40))
        killed_hud = font.render(f"Wights Killed: {stats['wights_killed']}   Night Kings Defeated: {stats['nk_kills']}/{len(NK_SCHEDULE_TIMES)}", True, (220,220,200))
        screen.blit(killed_hud, (14, 68))
        hints = small_font.render("Click to recruit soldier • SPACE pause • R reset • G grid • S snow • M music • V sfx • Q quit", True, (180,180,200))
        screen.blit(hints, (12, SCREEN_H - 26))

        # Game over / victory overlay
        if game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0,0,0,190))
            screen.blit(overlay, (0,0))
            title = big_font.render("WINTERFELL PREVAILS" if victory else "WINTERFELL HAS FALLEN", True, (200,230,210) if victory else (255,190,180))
            screen.blit(title, (SCREEN_W/2 - title.get_width()/2, SCREEN_H/2 - 220))
            mid = pygame.font.SysFont("Arial", 24)
            lines = [
                f"Time Survived: {int(runtime)} s",
                f"Soldiers Deployed: {total_soldiers_deployed}",
                f"Soldiers Killed: {stats['soldiers_killed']}",
                f"Wights Killed: {stats['wights_killed']}",
                f"Night Kings Defeated: {stats['nk_kills']}/{len(NK_SCHEDULE_TIMES)}",
                "",
                "Press R to Restart • Press Q to Quit"
            ]
            y = SCREEN_H/2 - 60
            for line in lines:
                txt = mid.render(line, True, (240,240,240))
                screen.blit(txt, (SCREEN_W/2 - txt.get_width()/2, y))
                y += 36

        pygame.display.flip()

    # stop sounds
    try:
        if sound_engine and getattr(sound_engine, "ambience_playing", False):
            sound_engine.ambience.stop()
    except:
        pass
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run()
