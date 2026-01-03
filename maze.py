import pygame as g
import math
import random
from time import time
from typing import Literal

random.seed(time())


class AABB: # Axis-Aligned Bounding Box

    __slots__ = ("xmin", "xmax", "ymin", "ymax")

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.ymin: float = ymin
        self.ymax: float = ymax

    def set_all (self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def is_overlap (self, other: 'AABB'):
        return (
            self.xmax > other.xmin and
            self.xmin < other.xmax and
            self.ymax > other.ymin and
            self.ymin < other.ymax
        )

    def get_size (self) -> g.Vector2:
        return g.Vector2(
            self.xmax - self.xmin,
            self.ymax - self.ymin
        )

    def copy (self):
        return AABB(self.xmin, self.xmax, self.ymin, self.ymax)

class CollisionMesh:

    __slots__ = ("vertices", "__vertices", "aabb")

    def __init__ (self, vertices: tuple[g.Vector2]):
        self.vertices = vertices
        self.__vertices = tuple(vertices)
        self.aabb = AABB(0, 0, 0, 0)
        self.recalc_AABB()

    def copy (self) -> 'CollisionMesh':
        new = CollisionMesh.__new__(CollisionMesh)
        new.vertices = tuple(self.vertices)
        new.__vertices = tuple(self.__vertices)
        new.aabb = self.aabb.copy()
        return new

    def recalc_AABB (self):
        aabb = self.aabb
        xs = tuple(v.x for v in self.vertices)
        aabb.xmin = min(xs)
        aabb.xmax = max(xs)
        ys = tuple(v.y for v in self.vertices)
        aabb.ymin = min(ys)
        aabb.ymax = max(ys)

    def recalc_coll_mesh (self, position: g.Vector2, orientation: float):
        self.vertices = tuple(
            v.rotate(-orientation) + position
            for v in self.__vertices
        )
        self.recalc_AABB()

    def calc_SA (self) -> list[g.Vector2]:
        vertices = self.vertices
        return [
            (vertices[i] - vertices[i+1])
            for i in range(-1, len(vertices)-1)
        ]

    # based on separating axis theorem
    def SA_coll (self, other: 'CollisionMesh') -> bool:

        axes = self.calc_SA()
        axes.extend(other.calc_SA())

        for axis in axes:
            projection = [axis.dot(vert) for vert in self.vertices]
            min1 = min(projection)
            max1 = max(projection)

            projection = [axis.dot(vert) for vert in other.vertices]
            min2 = min(projection)
            max2 = max(projection)

            if max1 <= min2 or max2 <= min1:
                return False

        return True

    def is_colliding (self, other: 'CollisionMesh') -> bool:
        return self.aabb.is_overlap(other.aabb) and CollisionMesh.SA_coll(self, other)

class Event:
    __slots__ = ("funcs")
    def __init__(self):
        self.funcs = []
    def connect (self, f):
        self.funcs.append(f)
    def delete (self, f):
        while 1:
            try:
                self.funcs.remove(f)
            except:
                return
    def execute (self):
        for f in self.funcs:
            f()

class Animation:
    def __init__(self, sheet: g.Surface, frame_size: g.Vector2, frame_count: int, frame_margin: int = 0):
        self.sheet = sheet
        self.ori_frame = g.Rect(0, 0, frame_size.x, frame_size.y)
        self.cur_frame = g.Rect(0, 0, frame_size.x, frame_size.y)
        self.frame_dist = int(frame_size.x + frame_margin)

        self.frame_count = frame_count
        self.anim_progress = 0
        self.anim_tick = 0.0

    def reset_anim (self, reverse: bool = False):
        self.anim_tick = self.frame_count * reverse
        self.anim_progress = self.anim_tick

    # returns if there's a change
    def tick_anim (self, speed: float) -> bool:
        self.anim_tick = (self.anim_tick + speed) % self.frame_count
        tick = math.floor(self.anim_tick)
        if tick == self.anim_progress:
            return False
        self.anim_progress = tick
        self.update_frame()
        return True

    def update_frame(self):
        self.cur_frame = self.ori_frame.move(self.frame_dist * self.anim_progress, 0)


class Object:

    FLAG_ALIVE      = 0b00000001
    FLAG_SIMULATED  = 0b00000010
    FLAG_COLLIDABLE = 0b00000100
    FLAG_ANIMATED   = 0b00001000
    FLAG_RENDERED   = 0b00010000
    FLAG_SHOWN      = 0b00100000

    TYPE_NONE = 0
    TYPE_PLAYER = 1
    TYPE_ENEMY = 2
    TYPE_WINCOND = 3
    TYPE_WALL = 4
    TYPE_STTINV_WALL = 5
    TYPE_BULLET = 7
    TYPE_PARTICLE = 8
    TYPE_TEXTDISPLAY = 9

    def __init__(self):

        self.type: int = Object.TYPE_NONE
        self.name: str = None

        self.position: g.Vector2 = g.Vector2(0, 0)
        self.orientation: float = 0.0
        self.flags: int = Object.FLAG_ALIVE

    def set_metadata (self, type: int = 0, name: str = ""):
        if type: self.type = type
        if name: self.name = name

    def is_alive (self) -> bool:
        return self.flags != 0b0

    def is_simulated (self) -> bool:
        return self.flags & Object.FLAG_SIMULATED

    def is_collidable (self) -> bool:
        return self.flags & Object.FLAG_COLLIDABLE

    def is_animated (self) -> bool:
        return self.flags & Object.FLAG_ANIMATED

    def is_rendered (self) -> bool:
        return self.flags & Object.FLAG_RENDERED

    def is_shown (self) -> bool:
        return self.flags & Object.FLAG_SHOWN

    def destroy (self):
        self.flags = 0b0

class SimulatedObject:

    def create_simulation (self, sim_init_func, sim_handle_func):
        self.sim_init_func = sim_init_func
        self.sim_handle_func = sim_handle_func
        self.flags |= Object.FLAG_SIMULATED

    def init_simulation (self):
        if not self.sim_init_func:
            return
        self.sim_init_func(self)
        self.sim_init_func = None

    def handle_simulation (self):
        self.sim_handle_func(self)

class CollidableObject:

    COLL_NONE = 0b00000000
    COLL_PLAYER = 0b00000001
    COLL_STTINV = 0b00000010
    COLL_BULLET = 0b00000100
    COLL_WINCOND = 0b00001000

    def create_collider (self, hitbox: CollisionMesh, coll_layer: int, coll_handler):
        self.hitbox = hitbox
        self.recalc_hitbox()
        self.coll_handler = coll_handler
        self.flags |= Object.FLAG_COLLIDABLE
        self.coll_layer = coll_layer
        self.coll_mask: int = not CollidableObject.COLL_NONE

    def make_coll_mask (self, mask: int):
        self.collision_mask = mask

    def recalc_hitbox (self):
        self.hitbox.recalc_coll_mesh(self.position, self.orientation)

    def collides_with (self, other: 'CollidableObject') -> bool:
        return bool(
            other.coll_layer & self.coll_mask or
            self.coll_layer & other.coll_mask
        )

    def is_collision (self, collider: 'CollidableObject') -> bool:
        return self.hitbox.is_colliding(collider.hitbox)

    def handle_collision (self, collider: 'CollidableObject'):
        handled = False
        self.coll_handler (self, collider, handled)

        handled = True
        collider.coll_handler (collider, self, handled)

    def render_hitbox (self, dest: g.Surface):
        g.draw.lines(dest, (128, 30, 20), True, self.hitbox.vertices, 2)

class SpriteObject:

    def create_sprite (self, z: int, image: g.Surface, shown: bool):
        self.z = z
        self.set_image(image)
        self.flags |= Object.FLAG_RENDERED | (Object.FLAG_SHOWN * (shown==True))

    def set_image (self, image: g.Surface):
        self.__image = image
        self.image = image.copy()
        self.image.set_alpha(image.get_alpha())
        self.image_offset = g.Vector2(image.get_size()) / 2
        self.rerotate_image()
        self.recalc_blitdata()

    def rerotate_image (self):
        if self.orientation <= 1 and self.orientation >= 359:
            return
        self.image = g.transform.rotate(self.__image, self.orientation)
        self.image.set_alpha(self.__image.get_alpha())
        self.image_offset = g.Vector2(self.image.get_size()) / 2

    def recalc_blitdata (self):
        self.blit_position = self.position - self.image_offset

    def get_blitsdata (self):
        return (self.image, self.blit_position)

    def show (self):
        self.flags |= Object.FLAG_SHOWN

    def hide (self):
        self.flags = self.flags & ~Object.FLAG_SHOWN

class AnimatedObject (SpriteObject):
    pass


class Wall (Object, CollidableObject, SpriteObject):

    def __init__(self, name: str, position: g.Vector2 | tuple[float, float], w: float, h: float, color: tuple[int, int, int]):
        super().__init__()
        self.set_metadata(Object.TYPE_WALL, name)
        self.position.update(position)
        img = alpha_Surface((w, h))
        img.fill(color)
        hitbox = CollisionMesh((
            g.Vector2(-w/2-2,-h/2-2), g.Vector2( w/2+2,-h/2-2),
            g.Vector2( w/2+2, h/2+2), g.Vector2(-w/2-2, h/2+2),
        ))
        self.create_collider(hitbox, CollidableObject.COLL_STTINV, self.collision_handler)
        self.create_sprite(-10, img, True)

    @staticmethod
    def collision_handler (self, collidee: Object, handled: bool):

        if collidee.type == Object.TYPE_PLAYER:

            wall = self
            player = collidee
            wbb = wall.hitbox.aabb
            pbb = player.hitbox.aabb
            ww, wh = wbb.get_size() / 2
            pw, ph = pbb.get_size() / 2

            total_w = ww + pw
            total_h = wh + ph

            distance_x = (wbb.xmax - pbb.xmax) + (wbb.xmin - pbb.xmin)
            distance_y = (wbb.ymax - pbb.ymax) + (wbb.ymin - pbb.ymin)

            ratio_x = abs(distance_x) / total_w
            ratio_y = abs(distance_y) / total_h

            SNAP_ROOM = 0.5

            if ratio_x > ratio_y:
                if distance_x < 0:
                    snap_position = (wbb.xmax + pw + SNAP_ROOM, player.position.y)
                else:
                    snap_position = (wbb.xmin - pw - SNAP_ROOM, player.position.y)
            else:
                if distance_y < 0:
                    snap_position = (player.position.x, wbb.ymax + ph + SNAP_ROOM)
                else:
                    snap_position = (player.position.x, wbb.ymin - ph - SNAP_ROOM)

            player.position.update(snap_position)
            player.recalc_hitbox()
            player.recalc_blitdata()

class SttInv_Wall (Object, CollidableObject):

    def __init__(self, name: str, position: g.Vector2 | tuple[float, float], hitbox: CollisionMesh):
        super().__init__()
        self.set_metadata(Object.TYPE_STTINV_WALL, name)
        self.position.update(position)
        self.create_collider(hitbox, CollidableObject.COLL_STTINV, self.collision_handler)

    @staticmethod
    def collision_handler (self, collidee: Object, handled: bool):

        if collidee.type == Object.TYPE_PLAYER:

            wall = self
            player = collidee
            wbb = wall.hitbox.aabb
            pbb = player.hitbox.aabb
            ww, wh = wbb.get_size() / 2
            pw, ph = pbb.get_size() / 2

            total_w = ww + pw
            total_h = wh + ph

            distance_x = (wbb.xmax - pbb.xmax) + (wbb.xmin - pbb.xmin)
            distance_y = (wbb.ymax - pbb.ymax) + (wbb.ymin - pbb.ymin)

            ratio_x = abs(distance_x) / total_w
            ratio_y = abs(distance_y) / total_h

            SNAP_ROOM = 0.5

            if ratio_x > ratio_y:
                if distance_x < 0:
                    snap_position = (wbb.xmax + pw + SNAP_ROOM, player.position.y)
                else:
                    snap_position = (wbb.xmin - pw - SNAP_ROOM, player.position.y)
            else:
                if distance_y < 0:
                    snap_position = (player.position.x, wbb.ymax + ph + SNAP_ROOM)
                else:
                    snap_position = (player.position.x, wbb.ymin - ph - SNAP_ROOM)

            player.position.update(snap_position)
            player.recalc_hitbox()
            player.recalc_blitdata()


class Player (Object, SimulatedObject, CollidableObject, SpriteObject):

    def __init__(self, name: str, position: g.Vector2 | tuple[float, float], hitbox: CollisionMesh):
        global soul_png, soul_hurt_png
        super().__init__()
        self.set_metadata(Object.TYPE_PLAYER, name)

        self.hp = Settings.PLAYER_HEALTH
        self.max_hp = Settings.PLAYER_HEALTH
        self.defense = 0
        self.mul_defense = 0
        self.set_defense(Settings.PLAYER_DEFENSE)

        self.img_soul = soul_png
        self.img_soul_hurt = soul_hurt_png

        self.invincible = False

        self.position.update(position)
        self.create_simulation(None, self.sim_handler)
        self.create_collider(hitbox, CollidableObject.COLL_PLAYER, Player.collision_handler)
        self.create_sprite(10, self.img_soul, True)

    def collision_handler (self, collidee: Object, handled: bool):
        if collidee.type == Object.TYPE_BULLET:
            self.iframe()
        elif collidee.type == Object.TYPE_ENEMY:
            return

    def move (self, movement: g.Vector2 | tuple[float, float]):
        self.position += movement
        self.recalc_hitbox()
        self.recalc_blitdata()

    @staticmethod
    def sim_handler (self: 'Player'):
        pass

    def set_defense (self, defense: int):
        self.defense = float(defense)
        if defense > (1/256):
            self.mul_defense = math.log(defense) + 1
            self.sub_defense = defense ** 1.125 / 10.0
            return
        self.mul_defense = (1/256)
        self.sub_defense = 0

    def damage (self, attack: float, damage_type: int = 0):
        if self.invincible:
            return;
        damage = max(
            0,
            (attack - self.sub_defense) / self.mul_defense
        )
        self.hp = max(
            0,
            int(self.hp - damage * Settings.DAMAGE_MULTIPLIER)
        )

    def heal (self, heal: float):
        self.hp = min(
            self.max_hp,
            int(self.hp + heal * Settings.HEAL_MULTIPLIER)
        )

    def iframe (self):
        self.set_image(self.img_soul_hurt)

class Knight (Object, CollidableObject, SpriteObject, SimulatedObject):

    IDLE             = 0
    ATTACK_STARTHROW = 1
    ATTACK_KNIFESHOT = 2
    ATTACK_SLASH     = 3
    ATTACK_SHARD     = 4

    set_positions = [
        g.Vector2(540, 120),
        g.Vector2(550, 200),
        g.Vector2(550, 300),
        g.Vector2(540, 380),

        g.Vector2(600, 165),
        g.Vector2(600, 335)
    ]

    def __init__(self, name: str, position: g.Vector2 | tuple[float, float], hitbox: CollisionMesh, image: g.Surface, attack: int, magic: int):
        super().__init__()
        self.set_metadata(Object.TYPE_ENEMY, name)
        self.create_collider(hitbox, CollidableObject.COLL_NONE, Knight.collision_handler)
        self.make_coll_mask(CollidableObject.COLL_PLAYER)
        self.make_coll_mask((Object.TYPE_PLAYER,))
        self.create_sprite(50, image, True)
        self.create_simulation(None, self.idle_state)

        self.attack = attack
        self.magic = magic

        self.state = 0
        self.attack_count = 0
        self.action = None
        self.pending_action = []

        self.tick = 0
        self.particle_tick = 0
        self.lerp_ticks = Settings.KNIGHT_LERP_TIME
        self.bobbing_ticks = Settings.KNIGHT_BOB_TIME
        self.bobbing_ticks2 = Settings.KNIGHT_BOB_TIME/2
        self.delay = Settings.KNIGHT_START_DELAY + random.randint(-32, 32)

        self.trail_velocity = g.Vector2(1.4, 0)

        self.position.update(position)
        self.original_position = self.position.copy()
        self.target_position = self.position.copy()

        global object_handler
        self.sword_t = BulletT.get("Sword")
        self.target = object_handler.get_object("SOUL", Object.TYPE_PLAYER)

    def collision_handler (self, collidee: Object, handled: bool):
        if collidee.type == Object.TYPE_PLAYER:
            global state_running
            state_running = False

    def move (self, movement: g.Vector2):
        self.position += movement
        self.recalc_hitbox()
        self.recalc_blitdata()

    def rotate (self, r: g.Vector2):
        self.orientation += r
        self.rerotate_image()

    def interp (self) -> bool:
        self.tick += 1
        x, y = (self.target_position - self.original_position) * math.sqrt(self.tick / self.lerp_ticks)

        if self.tick == self.lerp_ticks:
            self.position.update(self.target_position)
            self.original_position.update(self.target_position)
            self.recalc_hitbox()
            self.recalc_blitdata()
            self.tick = 0
            return True

        self.position.update(self.original_position + (x, y))
        self.recalc_hitbox()
        self.recalc_blitdata()

        return False

    def generate_trail (self, fade: float):
        global object_handler
        img_copy = self.image.copy()
        img_copy.set_alpha(self.image.get_alpha())
        trail = Particle(
            self.position, self.z -1,
            72, None, knight_trail_sim_handler,
            img_copy
        )
        trail.velocity = self.trail_velocity.copy()
        trail.fade = fade
        object_handler.add_finished_object(trail)

    def next_pending_action (self):
        self.action = self.pending_action.pop(0)

    @staticmethod
    def idle_state (self: 'Knight'):
        self.position.y = self.original_position.y + 8 * math.sin(self.tick / self.bobbing_ticks2 * math.pi)
        self.recalc_hitbox()
        self.recalc_blitdata()

        self.particle_tick = (self.particle_tick + 1) % 8
        self.tick = (self.tick + 1) % (self.bobbing_ticks*2)
        self.delay -= 1

        if self.particle_tick == 0:
            self.generate_trail(0.948)
        if self.delay <= 0:
            self.sim_handle_func = self.attack_state
            self.start_attack()

    @staticmethod
    def attack_state (self):
        self.particle_tick = (self.particle_tick + 1) % 8
        if self.particle_tick == 0:
            self.generate_trail(0.932)

        self.action()
        if self.state == 0:
            self.sim_handle_func = self.idle_state

    def start_attack (self):
        self.state = random.randint(1, 4)
        self.tick = 0
        self.init_attack()
        self.next_pending_action()

    def end_attack (self):
        self.state = 0
        self.tick = 0
        self.delay = Settings.KNIGHT_ATTACK_COOLDOWN + random.randint(-20, 20)
        self.trail_velocity.update(1.4, 0)

    def select_random_pos (self):
        self.target_position = (
            random.choice(Knight.set_positions) +
            (random.randint(-10, 10), random.randint(-10, 10))
        )
        self.original_position.update(self.position)
        self.next_pending_action()

    def interpolate_position (self):
        if self.interp():
            self.trail_velocity.update(1.4, 0)
            self.next_pending_action()

    def wait_delay (self):
        self.delay -= 1
        if self.delay == 0: self.next_pending_action()

    def init_attack (self):
        (None, self.init_attack_starthrow, self.init_attack_knifeshot,
        self.init_attack_slash, self.init_attack_shard)[self.state]()

    def init_attack_starthrow (self):
        self.attack_count = 3
        self.pending_action.extend((
            self.tick_attack_starthrow,
        ))

    def init_attack_knifeshot (self):
        self.attack_count = 3
        self.target_position.update((560, 250))
        self.pending_action.extend((
            self.tick_attack_knifeshot,
        ))

    def init_attack_slash (self):
        self.attack_count = 3
        self.pending_action.extend((
            self.tick_attack_slash,
        ))

    def init_attack_shard (self):
        self.attack_count = 3
        self.pending_action.extend((
            self.tick_attack_shard,
        ))

    def tick_attack_starthrow (self):
        if self.attack_count != 0:
            self.attack_count -= 1
            self.delay = 45
            self.pending_action.extend((
                self.select_random_pos,
                self.interpolate_position,
                self.attack_frame,
                self.wait_delay,
                self.tick_attack_starthrow
            ))
            self.next_pending_action()
            return

        self.delay = 60
        self.target_position.update(560, 250)
        self.pending_action.extend((
            self.wait_delay,
            self.interpolate_position,
            self.end_attack
        ))
        self.next_pending_action()

    def tick_attack_knifeshot (self):
        self.tick_attack_starthrow()

    def tick_attack_slash (self):
        self.tick_attack_starthrow()

    def tick_attack_shard (self):
        self.tick_attack_starthrow()

    def attack_frame (self):
        global object_handler

        vec = (self.target.position - self.position).normalize()
        vec1 = vec.rotate(12)
        vec2 = vec
        vec3 = vec.rotate(-12)

        vel1 = vec1 * Settings.SWORD_SPEED
        vel2 = vec2 * Settings.SWORD_SPEED
        vel3 = vec3 * Settings.SWORD_SPEED

        init1 = lambda o: sword_init_imf(o, vel1, math.atan2(vec1.x, vec1.y) / math.pi * 180)
        init2 = lambda o: sword_init_imf(o, vel2, math.atan2(vec2.x, vec2.y) / math.pi * 180)
        init3 = lambda o: sword_init_imf(o, vel3, math.atan2(vec3.x, vec3.y) / math.pi * 180)

        sword_t = self.sword_t
        sw1 = BulletInst(
            sword_t, self.position.copy(),
            self.attack, self.magic,
            Settings.SWORD_THOW_LIFETIME, init1
        )
        sw2 = BulletInst(
            sword_t, self.position+ vec*3,
            self.attack, self.magic,
            Settings.SWORD_THOW_LIFETIME, init2
        )
        sw3 = BulletInst(
            sword_t, self.position.copy(),
            self.attack, self.magic,
            Settings.SWORD_THOW_LIFETIME, init3
        )
        object_handler.add_finished_objects((sw1, sw2, sw3))

        if Settings.MORE_SWORD_THROW:
            vecx = vec.rotate(24)
            vecy = vec.rotate(-24)
            velx = vecx * Settings.SWORD_SPEED
            vely = vecy * Settings.SWORD_SPEED
            initx = lambda o: sword_init_imf(o, velx, math.atan2(vecx.x, vecx.y) / math.pi * 180)
            inity = lambda o: sword_init_imf(o, vely, math.atan2(vecy.x, vecy.y) / math.pi * 180)
            swx = BulletInst(
                sword_t, self.position.copy(),
                self.attack, self.magic,
                Settings.SWORD_THOW_LIFETIME, initx
            )
            swy = BulletInst(
                sword_t, self.position.copy(),
                self.attack, self.magic,
                Settings.SWORD_THOW_LIFETIME, inity
            )
            object_handler.add_finished_objects((swx, swy))

        play_swing_heavy()
        self.next_pending_action()

class FooWinCondition (Object, CollidableObject, SpriteObject):
    def __init__(self, name: str, image: g.Surface, hitbox: CollisionMesh):
        super().__init__()
        self.position.update(10, 250)
        self.set_metadata(Object.TYPE_WINCOND, name)
        self.create_collider(hitbox, CollidableObject.COLL_WINCOND, FooWinCondition.wccol)
        self.make_coll_mask(CollidableObject.COLL_PLAYER)
        self.create_sprite(60, image, True)
    @staticmethod
    def wccol (self: 'FooWinCondition', other: CollidableObject, handled: bool):
        if other.type == Object.TYPE_PLAYER:
            global state_win
            state_win = True
            play_heal()


class BulletT:
    types: list['BulletT'] = []
    type_count: int = 0

    def __init__(self, name: str, image: g.Surface, attack_mul: float, magic_mul: float, hitbox: CollisionMesh, coll_handler, sim_init, sim_handler):
        self.name = name
        self.id = BulletT.type_count
        self.attack_mul = attack_mul
        self.magic_mul = magic_mul

        self.image = image
        self.hitbox = hitbox
        self.coll_handler = coll_handler
        self.sim_init_func = sim_init
        self.sim_handler_func = sim_handler

        BulletT.types.append(self)
        BulletT.type_count += 1

    def do_damage (self, player: Player, attack: int, magic: int):
        player.damage(
            attack * self.attack_mul +
            magic * self.magic_mul
        )

    @classmethod
    def get (cls, name: str) -> str | None:
        for t in cls.types:
            if t.name == name:
                return t
        return None

class BulletInst (Object, SimulatedObject, CollidableObject, SpriteObject):

    def __init__(self, type: BulletT, position: g.Vector2, attack: int, magic: int, lifetime_max: float, second_init = None):
        super().__init__()
        self.set_metadata(Object.TYPE_BULLET, f"{type.name}#{str(id(self))[-9:-1]}")
        self.btype = type
        self.attacker_attack = attack
        self.attacker_magic = magic
        self.lifetime = 0
        self.lifetime_max = lifetime_max
        self.position = position.copy()

        self.create_simulation(type.sim_init_func, type.sim_handler_func)
        self.create_collider(type.hitbox.copy(), CollidableObject.COLL_BULLET, type.coll_handler)
        self.make_coll_mask(CollidableObject.COLL_PLAYER)
        self.create_sprite(0, type.image, True)
        self.init_simulation()
        if second_init: second_init(self)

    def move (self, mv: g.Vector2):
        self.position += mv
        self.recalc_hitbox()
        self.recalc_blitdata()

    def rotate (self, r: float):
        self.orientation += r
        self.rerotate_image()
        self.recalc_blitdata()
        

class Particle (Object, SimulatedObject, SpriteObject):

    def __init__(self, position: g.Vector2 | tuple[float, float], z: float, max_lifetime: int, sim_init, sim_handler, image: g.Surface):
        super().__init__()
        self.set_metadata(Object.TYPE_PARTICLE)
        self.position.update(position)
        self.velocity = g.Vector2(0, 0)
        self.create_simulation(sim_init, sim_handler)
        self.create_sprite(z, image, True)
        self.max_lifetime = max_lifetime
        self.lifetime = 0

    def tick_sim (self):
        if self.lifetime >= self.max_lifetime: self.destroy()
        self.lifetime += 1
        self.position += self.velocity

class HBar (Object, SimulatedObject, SpriteObject):

    def __init__(self, name: str, position: g.Vector2, size: g.Vector2, bg_col: tuple, bar_col: tuple, sim_init, sim_handler):
        super().__init__()
        self.set_metadata(Object.TYPE_NONE, name)
        self.position = position
        self.bar_size = size - (4, 4)
        self.fill: float = 1

        self.bg = srcalpha_Surface(size)
        self.bar = srcalpha_Surface(self.bar_size)
        self.bg.fill(bg_col)
        self.bar.fill(bar_col)

        self.create_simulation(sim_init, sim_handler)
        self.create_sprite(-100, self.bg, True)
        self.update()
        self.recalc_blitdata()
        self.init_simulation()

    def update (self):
        img = self.bg.copy()
        img.blit(self.bar, (2, 2), g.Rect(0, 0, self.bar_size.x * self.fill, self.bar_size.y))
        self.image = img

class TextDisplay (Object, SimulatedObject, SpriteObject):
    def __init__(self, name: str, font: g.font.Font, color: tuple[int, ...], pos: g.Vector2, display_init, display_handler):
        super().__init__()
        self.set_metadata(Object.TYPE_TEXTDISPLAY, name)
        self.position.update(pos)
        self.font = font
        self.color = color
        self.create_simulation(display_init, display_handler)
        self.create_sprite(999, None, True)
        self.init_simulation()
    def create_sprite (self, z: int, image: g.Surface, shown: bool):
        self.z = z
        self.image = image
        self.flags |= Object.FLAG_RENDERED | (Object.FLAG_SHOWN * (shown==True))
    def get_blitsdata(self):
        return (self.image, self.position)


class ObjectHandler:

    def __init__(self):

        self.objects: list[Object] = []
        self.simulated: list[SimulatedObject] = []
        self.collidable: list[CollidableObject] = []
        self.animated: list[AnimatedObject] = []
        self.rendered: list[SpriteObject] = []

        self.getter_cache_obj: list[Object] = []
        self.getter_cache_time: list[int] = []

        self.tick = 0

        self.hb_toggled: bool = False
        self.hb_toggle_held: bool = False
        self.render_hitboxes: bool = False

    def add_finished_object (self, object: Object):
        self.objects.append(object)
        if object.is_simulated() : self.simulated.append(object)
        if object.is_collidable(): self.collidable.append(object)
        if object.is_animated()  : self.animated.append(object)
        if object.is_rendered()  : self.rendered.append(object)

    def add_finished_objects (self, objects: tuple[Object] | list[Object]):
        for o in objects:
            self.add_finished_object(o)

    def handle_objects (self):
        self.handle_simulation()
        self.handle_collision()
        self.handle_animation()

        if self.tick == 0:
            self.cleaning_tick()

        global hbt_display, fnt

        if self.hb_toggled:
            if not self.hb_toggle_held:
                self.render_hitboxes = not self.render_hitboxes
                hbt_display.image = fnt.render(f"hitboxes [H]: {self.render_hitboxes}", True, hbt_display.color)
                self.hb_toggle_held = True
            self.hb_toggled = False
        elif self.hb_toggle_held:
            self.hb_toggle_held = False

        self.hb_toggled = False

        self.tick = (self.tick + 1) % 20

    def handle_simulation (self):
        self.simulated = [o for o in self.simulated if o.is_simulated()]
        for o in self.simulated:
            o.handle_simulation()

    def handle_collision (self):
        self.collidable = [o for o in self.collidable if o.is_collidable()]

        for i, coller in enumerate(self.collidable):
            for collee in self.collidable[i+1:]:
                if coller.collides_with(collee) and coller.is_collision(collee):
                    coller.handle_collision(collee)

    def handle_animation (self):
        self.animated = [o for o in self.animated if o.is_animated()]

    def blits_to (self, dest: g.Surface):
        self.rendered.sort(key = lambda o: o.z)
        bdat = [o.get_blitsdata() for o in self.rendered if o.is_shown()]

        dest.blits(bdat, False)

        if self.render_hitboxes:
            for o in self.collidable:
                o.render_hitbox(dest)

    def cleaning_tick (self):

        tick = g.time.get_ticks()

        self.objects = [o for o in self.objects if o.is_alive()]
        self.rendered = [o for o in self.rendered if o.is_rendered()]

        getter_o_buff = self.getter_cache_obj
        getter_t_buff = self.getter_cache_time
        self.getter_cache_obj = []
        self.getter_cache_time = []

        SECOND = 1000#ms
        cache_irrelevant = tick + 15*SECOND

        for o, t in zip(getter_o_buff, getter_t_buff):
            if t < cache_irrelevant and o.is_alive():
                self.getter_cache_obj.append(o)
                self.getter_cache_time.append(t)

    def toggle_hitbox (self):
        self.hb_toggled = True

    def cache_gotten_object (self, object: Object):
        self.getter_cache_obj.append(object)
        self.getter_cache_time.append(g.time.get_ticks())

    def get_object (self, name: str, type: int = None) -> Object | None:
        if type != None:
            for i, o in enumerate(self.getter_cache_obj):
                if o.type == type and o.name == name:
                    self.getter_cache_time[i] = g.time.get_ticks()
                    return o;
            for o in self.objects:
                if o.type == type and o.name == name:
                    self.cache_gotten_object(o)
                    return o;
        else:
            for i, o in enumerate(self.getter_cache_obj):
                if o.name == name:
                    self.getter_cache_time[i] = g.time.get_ticks()
                    return o;
            for o in self.objects:
                if o.name == name:
                    self.cache_gotten_object(o)
                    return o;
        return None

    def destroy_all (self, type: int = None, name: str = None):
        if type != None:
            if name == None:
                for o in self.objects:
                    if o.type == type:
                        o.destroy()
                return;

            for o in self.objects:
                if o.type == type and o.name == name:
                    o.destroy()
            return;

        if name == None:
            return;

        for o in self.objects:
            if o.name == name:
                o.destroy()
        return;


class InputHandler:

    def __init__(self):
        self.event_w = Event()
        self.event_s = Event()
        self.event_a = Event()
        self.event_d = Event()
        self.event_h = Event()

    def handle_input (self):

        for e in g.event.get():

            if e.type == g.QUIT:
                global state_running
                state_running = False
                return;

        key = g.key.get_pressed()

        if key[g.K_w]:
            self.event_w.execute()
        if key[g.K_s]:
            self.event_s.execute()
        if key[g.K_a]:
            self.event_a.execute()
        if key[g.K_d]:
            self.event_d.execute()
        if key[g.K_f]:
            self.event_f.execute()
        if key[g.K_h]:
            self.event_h.execute()
        if key[g.K_j]:
            self.event_j.execute()


class MazeMaker:
    def __init__(self, maze_pos: g.Vector2, maze_size: g.Vector2):
        self.maze_tl_pos = maze_pos - maze_size/2
        self.maze_size = maze_size
        self.mask_w = 0
        self.mask_h = 0
        self.hwalls = []
        self.vwalls = []
        self.heal_spots = []

    @staticmethod
    def load_mask (fname: str) -> g.Mask:
        surf = g.image.load(fname)
        mask = g.mask.from_surface(surf)
        return mask

    def parse_mask (self, mask: g.Mask):
        mask_w, mask_h = mask.get_size()
        self.mask_w = mask_w
        self.mask_h = mask_h

        wall_count = 0
        adjacent_empty_map = [
            [0 for _h in range(mask_w)]
            for _v in range(mask_h)
        ]

        # make horizontal walls and calculate adj empties
        for y in range(mask_h):
            for x in range(mask_w):
                if mask.get_at((x, y)):
                    wall_count += 1
                else:
                    if wall_count > 1:
                        self.hwalls.append((x-wall_count, y, wall_count-1))
                    wall_count = 0
                    if x > mask_w and mask.get_at((x+1, y)):
                        adjacent_empty_map[y][x] += 1
                        adjacent_empty_map[y][x+1] += 1
                    if y > mask_h and mask.get_at((x, y+1)):
                        adjacent_empty_map[y][x] += 1
                        adjacent_empty_map[y+1][x] += 1
            if wall_count > 1:
                self.hwalls.append((x-wall_count+1, y, wall_count-1))
            wall_count = 0

        x = 0
        y = 0
        start_y = 0

        # make vertical walls
        while x < mask_w:
            start_y = y
            while mask.get_at((x, y)):
                y += 1
                if mask_h == y:
                    if (y-1) > start_y:
                        self.vwalls.append((x, start_y, y-start_y-1))
                    y = 0
                    x += 1
                    break
            else:
                if (y-1) > start_y:
                    self.vwalls.append((x, start_y, y-start_y-1))
            while x < mask_w and not mask.get_at((x, y)):
                if mask_h == y:
                    y = 0
                    x += 1
                    break
                y += 1

    def make_walls (self, thickness: float, color: tuple[int, int, int]) -> list[Wall]:
        li = []
        maze_w, maze_h = self.maze_size
        mask_w, mask_h = self.mask_w, self.mask_h
        mx = maze_w / mask_w
        my = maze_h / mask_h
        tl_pos = self.maze_tl_pos

        for x, y, wl in self.hwalls:
            len =  wl * mx
            pos = tl_pos + g.Vector2(x*mx + len/2, y*my)
            w = Wall("Maze HWall", pos, len+thickness, thickness, color)
            li.append(w)

        for x, y, wl in self.vwalls:
            len =  wl * my
            pos = tl_pos + g.Vector2(x*mx, y*my + len/2)
            w = Wall("Maze VWall", pos, thickness, len+thickness, color)
            li.append(w)

        return li

    def make_heals (self) -> list[any]:
        pass


def converted_Surface (size) -> g.Surface:
    return g.Surface(size).convert()
def alpha_Surface (size, color_key: tuple[int, int, int] = (255, 0, 255), alpha: int = 255) -> g.Surface:
    new = g.Surface(size, 0).convert()
    new.set_colorkey(color_key)
    new.set_alpha(alpha)
    return new
def srcalpha_Surface (size) -> g.Surface:
    new = g.Surface(size).convert_alpha()
    new.fill((0, 0, 0, 0))
    return new

def load_img (name: str, mode: Literal["raw", "converted", "alpha", "srcalpha"]) -> g.Surface:
    if mode == "srcalpha":
        return g.image.load(name).convert_alpha()
    if mode == "alpha":
        new = g.image.load(name).convert()
        new.set_colorkey(EPINK)
        new.set_alpha(255)
        return new
    if mode == "raw":
        return g.image.load(name)
    if mode == "converted":
        return g.image.load(name).convert()
def bite_img (source: g.Surface, bite: g.Rect, mode: Literal["raw", "converted", "alpha", "srcalpha"]) -> g.Surface:
    if mode == "srcalpha":
        new = srcalpha_Surface((bite.w, bite.h))
        new.blit(source, (0, 0), bite)
        return new
    if mode == "alpha":
        new = alpha_Surface((bite.w, bite.h))
        new.blit(source, (0, 0), bite)
        return new
    if mode == "raw":
        new = g.Surface((bite.w, bite.h))
        new.blit(source, (0, 0), bite)
        return new
    if mode == "converted":
        new = converted_Surface((bite.w, bite.h))
        new.blit(source, (0, 0), bite)
        return new
def load_snd (name: str, vol: float) -> g.mixer.Sound:
    new = g.mixer.Sound(name)
    new.set_volume(vol)
    return new


def knight_trail_sim_handler (self: Particle):
    self.tick_sim()
    self.image.set_alpha(self.image.get_alpha() * self.fade)
    self.recalc_blitdata()

def sword_init_imf (sword: BulletInst, velocity: g.Vector2, angle: float):
    sword.velocity = velocity
    sword.orientation = angle
    sword.image = rk_sword_png_angled_array[int(angle)%360]
    sword.image_offset = g.Vector2(sword.image.get_size()) / 2
    sword.recalc_hitbox()
    sword.recalc_blitdata()
def sword_sim (sword: BulletInst):
    sword.move(sword.velocity)
    sword.lifetime += 1
    if sword.lifetime > sword.lifetime_max:
        sword.destroy()
def sword_col (sword: BulletInst, other: CollidableObject, handled: bool):
    if other.type == Object.TYPE_PLAYER:
        btype = sword.btype
        btype.do_damage(other, sword.attacker_attack, sword.attacker_magic)
        sword.destroy()
        play_hurt()


def hpbar_init (bar: HBar):
    global soul
    bar.hp_link = soul
    bar.__temp_hp = -1
def hpbar_sim (bar: HBar):
    soul = bar.hp_link
    if soul.hp == bar.__temp_hp:
        return
    bar.__temp_hp = soul.hp
    bar.fill = soul.hp / soul.max_hp
    bar.update()

def fps_display_init (disp: TextDisplay):
    global clock
    disp.clock = clock
def fps_display_handler (disp: TextDisplay):
    disp.image = disp.font.render(
        f"fps: {disp.clock.get_fps(): .2f}",
        True, disp.color
    )
def obj_display_init (disp: TextDisplay):
    global object_handler
    disp.object_handler = object_handler
def obj_display_handler (disp: TextDisplay):
    disp.image = disp.font.render(
        f"object count: {len(object_handler.objects)}",
        True, disp.color
    )
def hbt_display_init (disp: TextDisplay):
    global object_handler
    disp.image = fnt.render(f"hitboxes [H]: {object_handler.render_hitboxes}", True, disp.color)
def hbt_display_handler (disp: TextDisplay):
    pass

def fade_img (surf: g.Surface, fade: float) -> None:
    surf.fill((255, 255, 255, int(fade*255)), special_flags=g.BLEND_MULT)
def new_faded_img (surf: g.Surface, fade: float) -> g.Surface:
    new = surf.copy()
    new.fill((255, 255, 255, int(fade*255)), special_flags=g.BLEND_MULT)
    return new

def new_rotated_img (surf: g.Surface, degree: float) -> g.Surface:
    return g.transform.rotate(surf, degree)

def play_hurt():
    global hurt_wav
    hurt_wav.play()
def play_heal():
    global heal_wav
    heal_wav.play()
def play_swing_heavy():
    global swingheavy_wav
    swingheavy_wav.play()


g.init()
g.mixer.init()
win = g.display.set_mode((700, 500))
g.display.set_caption("Knight Labirynth")


SECOND = 60.0  # 60 ticks per second
PX = 1
DELTA_T = 1/SECOND
EPINK = (255, 0, 255)
WIN_FILLER = (0, 0, 0)
TRANSPARENT = (0, 0, 0, 0)
clock = g.time.Clock()


class Settings:
    PLAYER_SPEED = 54*PX * DELTA_T
    SWORD_SPEED = 204*PX * DELTA_T
    STAR_SPEED = 156*PX * DELTA_T   # NOT USED YET

    PLAYER_HEALTH = 400
    PLAYER_DEFENSE = 10
    DAMAGE_MULTIPLIER = 1
    HEAL_MULTIPLIER = 1
    PLAYER_IFRAMES = int(1*SECOND)  # NOT USED YET

    KNIGHTS = 1
    KNIGHT_ATTACK = 12
    KNIGHT_MAGIC = 20

    KNIGHT_BOB_TIME = int(3.25*SECOND)
    KNIGHT_LERP_TIME = int(0.5*SECOND)
    KNIGHT_START_DELAY = int(3*SECOND)
    KNIGHT_ATTACK_COOLDOWN = int(2*SECOND)

    SWORD_ATTACK_MUL = 7
    SWORD_MAGIC_MUL = 2

    SWORD_THOW_LIFETIME = 2.5*SECOND
    MORE_SWORD_THROW = True

    SWORD_LAUNCH_TARGETING_TICKS = 1*SECOND    # NOT USED YET


knight_ogg = load_snd("knight.ogg", 0.10)

hurt_wav = load_snd("snd_hurt1.wav", 0.5)
heal_wav = load_snd("snd_power.wav", 0.5)

swing_wav = load_snd("snd_swing.wav", 0.5)
swingsmall_wav = load_snd("snd_smallswing.wav", 0.5)
swingheavy_wav = load_snd("snd_heavyswing.wav", 0.5)

treasure = g.transform.scale(load_img("treasure.png", "srcalpha"), (40, 40))
soul_png = load_img("soul.png", "srcalpha")
g.display.set_icon(soul_png)
soul_hurt_png = soul_png.copy()
soul_hurt_png.set_alpha(128)


rk_ssh_png = load_img("roaring_knight_spritesheet.png", "srcalpha")

rk_idle_png = bite_img(rk_ssh_png, g.Rect(1357, 298, 72, 65), "srcalpha")
rk_sword_png = bite_img(rk_ssh_png, g.Rect(681, 463, 29, 67), "srcalpha")
rk_sword_png_angled_array = tuple(
    new_rotated_img(rk_sword_png, n)
    for n in range(360)
)


maze1_schm = MazeMaker.load_mask("maze1.png")
maze2_schm = MazeMaker.load_mask("maze2.png")
maze_schm = MazeMaker.load_mask("MazeXXX.png")

maze_schms = (maze1_schm, maze2_schm)


sheet_equip_sword = bite_img(rk_ssh_png, g.Rect(8, 130, 2408, 127), "srcalpha")
sheet_point = bite_img(rk_ssh_png, g.Rect(8, 290, 520, 88), "srcalpha")
sheet_slash = bite_img(rk_ssh_png, g.Rect(538, 290, 605, 115), "srcalpha")
sheet_throw = bite_img(rk_ssh_png, g.Rect(741, 443, 630, 127), "srcalpha")


anim_equip_sword = Animation(sheet_equip_sword, g.Vector2(122, 127), 19, 5)
anim_point = Animation(sheet_point, g.Vector2(100, 88), 5, 5)
anim_slash = Animation(sheet_slash, g.Vector2(117, 115), 5, 5)
anim_throw = Animation(sheet_throw, g.Vector2(122, 127), 5, 5)


fnt = g.font.Font(None, 22)
Sword_BulletT = BulletT(
    "Sword", rk_sword_png,
    Settings.SWORD_ATTACK_MUL, Settings.SWORD_MAGIC_MUL,
    CollisionMesh((
        g.Vector2(-6, 25), g.Vector2( 6, 25),
        g.Vector2( 6,-25), g.Vector2(-6,-25)
    )),
    sword_col, None, sword_sim
)
# BulletStar_T = BulletT(
#     "Star", 0.2, 1, ...,
#     HitMesh((
#         g.Vector2(0, 0), g.Vector2(0, 0),
#         g.Vector2(0, 0), g.Vector2(0, 0)
#     )),
#     ..., ..., ...
# )


object_handler = ObjectHandler()
input_handler = InputHandler()

soul = Player(
    "SOUL", (300, 250),
    CollisionMesh((
        g.Vector2(-4,-4), g.Vector2( 4,-4),
        g.Vector2( 4, 4), g.Vector2(-4, 4)
    ))
)
knight = lambda: Knight(
    "The Roaring Knight", (560, 250),
    CollisionMesh((
        g.Vector2( 0,-32), g.Vector2( 32, 0),
        g.Vector2( 0, 32), g.Vector2(-32, 0)
    )), rk_idle_png,
    Settings.KNIGHT_ATTACK, Settings.KNIGHT_MAGIC
)
wc = FooWinCondition(
    "WC", treasure,
    CollisionMesh((
        g.Vector2(-18,-18), g.Vector2( 18,-18),
        g.Vector2( 18, 18), g.Vector2(-18, 18),
    ))
)

hp_bar = HBar(
    "HP Bar", g.Vector2(120, 460), g.Vector2(200, 12),
    (225, 225, 255, 112), (255, 20, 20),
    hpbar_init, hpbar_sim
)

fps_display = TextDisplay(
    "FPS Display", fnt,
    (255, 0, 0, 255), g.Vector2(8, 10),
    fps_display_init, fps_display_handler
)
obj_display = TextDisplay(
    "Object Count Display", fnt,
    (0, 127, 255, 255), g.Vector2(8, 30),
    obj_display_init, obj_display_handler
)
hbt_display = TextDisplay(
    "Hitbox Toggle Display", fnt,
    (100, 200, 100), g.Vector2(550, 10),
    hbt_display_init, hbt_display_handler
)

object_handler.add_finished_objects((
    SttInv_Wall("SIW-L", (0, 0),
        CollisionMesh((g.Vector2(0,-200), g.Vector2(0, 700),
            g.Vector2(-200, 700), g.Vector2(-200,-200)))
    ),
    SttInv_Wall("SIW-R", (700, 0),
        CollisionMesh((g.Vector2(0,-200), g.Vector2(0, 700),
            g.Vector2(200, 700), g.Vector2(200,-200)))
    ),
    SttInv_Wall("SIW-U", (0, 0),
        CollisionMesh((g.Vector2(-200, 0), g.Vector2(900, 0),
            g.Vector2(900,-200), g.Vector2(-200,-200)))
    ),
    SttInv_Wall("SIW-B", (0, 500),
        CollisionMesh((g.Vector2(-200, 0), g.Vector2(900, 0),
            g.Vector2(900, 200), g.Vector2(-200, 200)))
    ),
    soul,
    wc,
    hp_bar,
    fps_display,
    obj_display, 
    hbt_display
))


maze_maker = MazeMaker(g.Vector2(300, 250), g.Vector2(300, 300))
maze_maker.parse_mask(random.choice(maze_schms))
walls = maze_maker.make_walls(5, (12, 150, 8))

object_handler.add_finished_objects(walls)


for i in range(Settings.KNIGHTS):
    object_handler.add_finished_object(knight())


input_handler.event_w.connect(lambda: soul.move(( 0,-Settings.PLAYER_SPEED)))
input_handler.event_s.connect(lambda: soul.move(( 0, Settings.PLAYER_SPEED)))
input_handler.event_a.connect(lambda: soul.move((-Settings.PLAYER_SPEED, 0)))
input_handler.event_d.connect(lambda: soul.move(( Settings.PLAYER_SPEED, 0)))
input_handler.event_h.connect(lambda: object_handler.toggle_hitbox())

AnimTest_RK = srcalpha_Surface((122, 127))
AnimTest_RK.blit(rk_idle_png, (13, 30))

knight_ogg.play(-1)

state_running = True
state_win = False

while state_running:

    input_handler.handle_input()
    object_handler.handle_objects()

    if anim_throw.tick_anim(0.2): # temporary; to test animation
        AnimTest_RK.fill(TRANSPARENT)
        AnimTest_RK.blit(anim_throw.sheet, (0, 0), anim_throw.cur_frame)

    win.fill(WIN_FILLER)
    object_handler.blits_to(win)
    win.blit(AnimTest_RK, (0, 0)) # temporary; to test animation
    g.display.flip()


    if state_win:
        break

    clock.tick(SECOND)


g.mixer.quit()
g.quit()
