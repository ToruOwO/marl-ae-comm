import numpy as np
from enum import IntEnum
from gym_minigrid.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

# map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'orange': np.array([255, 165, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'cyan': np.array([0, 139, 139]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'olive': np.array([128, 128, 0]),
    'grey': np.array([100, 100, 100]),
    'worst': np.array([74, 65, 42]),
    'pink': np.array([255, 0, 189]),
    'white': np.array([255, 255, 255]),
    'prestige': np.array([255, 255, 255]),
    'shadow': np.array([35, 25, 30]),  # dark purple color for invisible cells
}

# used to map colors to integers
COLOR_TO_IDX = dict({v: k for k, v in enumerate(COLORS.keys())})

OBJECT_TYPES = []


class RegisteredObjectType(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if name not in OBJECT_TYPES:
            OBJECT_TYPES.append(cls)

        def get_recursive_subclasses():
            return OBJECT_TYPES

        cls.recursive_subclasses = staticmethod(get_recursive_subclasses)
        return cls


class WorldObj(metaclass=RegisteredObjectType):
    def __init__(self, color='worst', state=0):
        self.color = color
        self.state = state
        self.contains = None

        # some objects can have agents on top (e.g. floor, open doors, etc)
        self.agents = []

        self.pos_init = None
        self.pos = None
        self.is_agent = False

    @property
    def dir(self):
        return None

    def set_position(self, pos):
        if self.pos_init is None:
            self.pos_init = pos
        self.pos = pos

    @property
    def numeric_color(self):
        return COLORS[self.color]

    @property
    def type(self):
        return self.__class__.__name__

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def can_contain(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def encode(self, str_class=False):
        if bool(str_class):
            enc_class = self.type
        else:
            enc_class = self.recursive_subclasses().index(self.__class__)
        if isinstance(self.color, int):
            enc_color = self.color
        else:
            enc_color = COLOR_TO_IDX[self.color]
        return enc_class, enc_color, self.state

    def describe(self):
        return f'Obj: {self.type}({self.color}, {self.state})'

    @classmethod
    def decode(cls, type, color, state):
        if isinstance(type, str):
            cls_subclasses = {c.__name__: c for c in cls.recursive_subclasses()}
            if type not in cls_subclasses:
                raise ValueError(
                    f'Not sure how to construct a {cls} of (sub)type {type}'
                )
            return cls_subclasses[type](color, state)
        elif isinstance(type, int):
            subclass = cls.recursive_subclasses()[type]
            return subclass(color, state)

    def render(self, img):
        raise NotImplementedError

    def str_render(self, dir=0):
        return '??'


class GridAgent(WorldObj):
    def __init__(self, *args, neutral_shape, can_overlap, color='red',
                 **kwargs):
        super().__init__(*args, **{'color': color, **kwargs})
        self.metadata = {
            'color': color,
        }
        self.is_agent = True
        self.comm = 0
        self.neutral_shape = neutral_shape

        self._can_overlap = can_overlap

    @property
    def dir(self):
        return self.state % 4

    @property
    def type(self):
        return 'Agent'

    @dir.setter
    def dir(self, dir):
        self.state = self.state // 4 + dir % 4

    def str_render(self, dir=0):
        return ['>>', 'VV', '<<', '^^'][(self.dir + dir) % 4]

    def can_overlap(self):
        return self._can_overlap

    def render(self, img):
        if self.neutral_shape:
            shape_fn = point_in_circle(0.5, 0.5, 0.31)
        else:
            shape_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50),
                                         (0.12, 0.81),)
            shape_fn = rotate_fn(shape_fn, cx=0.5, cy=0.5,
                                 theta=1.5 * np.pi * self.dir)
        fill_coords(img, shape_fn, COLORS[self.color])


class BulkObj(WorldObj, metaclass=RegisteredObjectType):
    def __hash__(self):
        return hash((self.__class__, self.color, self.state,
                     tuple(self.agents)))

    def __eq__(self, other):
        return hash(self) == hash(other)


class BonusTile(WorldObj):
    def __init__(self, reward, penalty=-0.1, bonus_id=0, n_bonus=1,
                 initial_reward=True, reset_on_mistake=False, color='yellow',
                 *args, **kwargs):
        super().__init__(*args, **{'color': color, **kwargs, 'state': bonus_id})
        self.reward = reward
        self.penalty = penalty
        self.n_bonus = n_bonus
        self.bonus_id = bonus_id
        self.initial_reward = initial_reward
        self.reset_on_mistake = reset_on_mistake

    def can_overlap(self):
        return True

    def str_render(self, dir=0):
        return 'BB'

    def get_reward(self, agent):
        # If the agent hasn't hit any bonus tiles, set its bonus state so that
        #  it'll get a reward from hitting this tile.
        first_bonus = False
        if agent.bonus_state is None:
            agent.bonus_state = (self.bonus_id - 1) % self.n_bonus
            first_bonus = True

        if agent.bonus_state == self.bonus_id:
            # This is the last bonus tile the agent hit
            rew = -np.abs(self.penalty)
        elif (agent.bonus_state + 1) % self.n_bonus == self.bonus_id:
            # The agent hit the previous bonus tile before this one
            agent.bonus_state = self.bonus_id
            # rew = agent.bonus_value
            rew = self.reward
        else:
            # The agent hit any other bonus tile before this one
            rew = -np.abs(self.penalty)

        if self.reset_on_mistake:
            agent.bonus_state = self.bonus_id

        if first_bonus and not bool(self.initial_reward):
            return 0
        else:
            return rew

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, reward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def can_overlap(self):
        return True

    def get_reward(self, agent):
        return self.reward

    def str_render(self, dir=0):
        return 'GG'

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Wall(BulkObj):
    def see_behind(self):
        return False

    def str_render(self, dir=0):
        return 'WW'

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class FreeDoor(WorldObj):
    class states(IntEnum):
        open = 1
        closed = 2

    def is_open(self):
        return self.state == self.states.open

    def can_overlap(self):
        return False

    def see_behind(self):
        return self.is_open()

    def toggle(self, agent, pos):
        if self.state == self.states.closed:
            self.state = self.states.open
        elif self.state == self.states.open:
            # door can only be opened once
            pass
        else:
            raise ValueError(f'?!?!?! FreeDoor in state {self.state}')
        return True

    def render(self, img):
        c = COLORS[self.color]

        if self.state == self.states.open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)
