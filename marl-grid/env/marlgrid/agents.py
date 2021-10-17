import gym
import numba
import numpy as np
import warnings
from enum import IntEnum

from .objects import GridAgent, BonusTile


class GridAgentInterface(GridAgent):
    class actions(IntEnum):
        # NOTE: the first 4 actions also match the directional vector

        right = 0  # Move right
        down = 1  # Move down
        left = 2  # Move left
        up = 3  # Move up
        done = 4  # Done completing task / Stay
        toggle = 5  # Toggle/activate an object
        pickup = 6  # Pick up an object
        drop = 7  # Drop an object

    class skills(IntEnum):
        none = 0
        dist = 1  # Knows everyone's distance-to-goal
        sight = 2  # Can see the whole map

    def __init__(
            self,
            view_size=7,
            view_tile_size=5,
            view_offset=0,
            observation_style='image',
            observe_rewards=False,
            observe_position=False,
            observe_self_position=False,
            observe_done=False,
            observe_self_env_act=False,
            observe_orientation=False,
            observe_t=False,
            restrict_actions=False,
            see_through_walls=False,
            hide_item_types=[],
            prestige_beta=0.95,
            prestige_scale=2,
            allow_negative_prestige=False,
            spawn_delay=0,
            comm_dim=0,
            comm_len=1,
            discrete_comm=True,
            n_agents=1,
            is_adversary=False,
            observe_identity=True,
            skill=0,
            **kwargs):
        super().__init__(**kwargs)

        self.view_size = view_size
        self.view_tile_size = view_tile_size
        self.view_offset = view_offset
        self.observation_style = observation_style
        self.observe_rewards = observe_rewards
        self.observe_position = observe_position
        self.observe_self_position = observe_self_position
        self.observe_done = observe_done
        self.observe_self_env_act = observe_self_env_act
        self.observe_orientation = observe_orientation
        self.observe_t = observe_t
        self.hide_item_types = hide_item_types
        self.see_through_walls = see_through_walls
        self.init_kwargs = kwargs
        self.restrict_actions = restrict_actions
        self.prestige_beta = prestige_beta
        self.prestige_scale = prestige_scale
        self.allow_negative_prestige = allow_negative_prestige
        self.spawn_delay = spawn_delay
        self.comm_dim = comm_dim
        self.comm_len = comm_len
        self.discrete_comm = discrete_comm
        self.n_agents = n_agents  # for communication space
        self.is_adversary = is_adversary
        self.observe_identity = observe_identity
        self.skill = skill

        # "stay" action by default
        self.env_act = 4
        if self.restrict_actions:
            env_act_dim = 5
        else:
            env_act_dim = 6

        if comm_dim > 0 and comm_len > 0:
            self.observe_comm = True
            if discrete_comm:
                comm_space = [gym.spaces.MultiDiscrete([int(
                    comm_dim) for _ in range(comm_len)]) for _ in range(
                    n_agents)]
                comm_act_space = comm_space[0]
                comm_space = gym.spaces.Tuple(comm_space)
                self.comm = np.zeros((comm_len,))
            else:
                comm_space = gym.spaces.Box(low=0.0,
                                            high=comm_dim,
                                            shape=(n_agents, comm_len),
                                            dtype=np.float32)
                comm_act_space = gym.spaces.Box(low=0.0,
                                                high=comm_dim,
                                                shape=(comm_len,),
                                                dtype=np.float32)
                self.comm = np.zeros((comm_len,), dtype=np.float32)
        else:
            self.observe_comm = False

        if self.prestige_beta > 1:
            self.prestige_beta = 0.95

        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(view_tile_size * view_size, view_tile_size * view_size, 3),
            dtype='uint8',
        )
        if observation_style == 'image':
            self.observation_space = image_space
        elif observation_style == 'dict':
            obs_space = {
                'pov': image_space,
            }
            if self.observe_rewards:
                obs_space['reward'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                     shape=(),
                                                     dtype=np.float32)
            if self.observe_position:
                # discrete position space is handled in base.py
                obs_space['position'] = gym.spaces.Box(low=0, high=1,
                                                       shape=(n_agents, 2),
                                                       dtype=np.float32)
            if self.observe_self_position:
                # discrete position space is handled in base.py
                obs_space['selfpos'] = gym.spaces.Box(low=0, high=1,
                                                      shape=(2,),
                                                      dtype=np.float32)
            if self.observe_done:
                obs_space['done'] = gym.spaces.Discrete(n=2)
            if self.observe_self_env_act:
                obs_space['self_env_act'] = gym.spaces.Discrete(n=env_act_dim)
            if self.observe_orientation:
                obs_space['orientation'] = gym.spaces.Discrete(n=4)
            if self.observe_t:
                obs_space['t'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(),
                                                dtype=np.float32)
            if self.observe_identity:
                obs_space['identity'] = gym.spaces.Discrete(n=2)
            if self.observe_comm:
                obs_space['comm'] = comm_space
            self.observation_space = gym.spaces.Dict(obs_space)
        elif observation_style == 'tuple':
            obs_space = [image_space]
            if self.observe_rewards:
                obs_space += gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(),
                                            dtype=np.float32)
            if self.observe_position:
                # discrete position space is handled in base.py
                obs_space += gym.spaces.Box(low=0, high=1,
                                            shape=(n_agents, 2),
                                            dtype=np.float32)
            if self.observe_self_position:
                # discrete position space is handled in base.py
                obs_space += gym.spaces.Box(low=0, high=1,
                                            shape=(2,),
                                            dtype=np.float32)
            if self.observe_done:
                obs_space += gym.spaces.Discrete(n=2)
            if self.observe_self_env_act:
                obs_space += gym.spaces.Discrete(n=env_act_dim)
            if self.observe_orientation:
                obs_space += gym.spaces.Discrete(n=4)
            if self.observe_t:
                obs_space += gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(),
                                            dtype=np.float32)
            if self.observe_identity:
                obs_space += gym.spaces.Discrete(n=2)
            if self.observe_comm:
                obs_space += comm_space
            self.observation_space = gym.spaces.Tuple(obs_space)
        else:
            raise ValueError(
                f"{self.__class__.__name__} kwarg 'observation_style' must be "
                f"one of 'image', 'dict', 'tuple'.")

        if comm_dim > 0 and comm_len > 0:
            # include communication action space
            self.action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(env_act_dim), comm_act_space])
        else:
            self.action_space = gym.spaces.Discrete(env_act_dim)

        self.metadata = {
            **self.metadata,
            'view_size': view_size,
            'view_tile_size': view_tile_size,
        }
        self.reset(new_episode=True)

    def render_post(self, tile):
        return tile

    def clone(self):
        ret = self.__class__(
            view_size=self.view_size,
            view_offset=self.view_offset,
            view_tile_size=self.view_tile_size,
            observation_style=self.observation_style,
            observe_rewards=self.observe_rewards,
            observe_position=self.observe_position,
            observe_self_position=self.observe_self_position,
            observe_done=self.observe_done,
            observe_self_env_act=self.observe_self_env_act,
            observe_orientation=self.observe_orientation,
            observe_t=self.observe_t,
            hide_item_types=self.hide_item_types,
            restrict_actions=self.restrict_actions,
            see_through_walls=self.see_through_walls,
            prestige_beta=self.prestige_beta,
            prestige_scale=self.prestige_scale,
            allow_negative_prestige=self.allow_negative_prestige,
            spawn_delay=self.spawn_delay,
            comm_dim=self.comm_dim,
            comm_len=self.comm_len,
            discrete_comm=self.discrete_comm,
            n_agents=self.n_agents,
            is_adversary=self.is_adversary,
            observe_identity=self.observe_identity,
            skill=self.skill,
            **self.init_kwargs
        )
        return ret

    def at_pos(self, pos):
        if self.pos[0] == pos[0] and self.pos[1] == pos[1]:
            return True
        return False

    def on_step(self, obj):
        if isinstance(obj, BonusTile):
            self.bonuses.append((obj.bonus_id, self.prestige))
        self.prestige *= self.prestige_beta

    def reward(self, rew):
        if self.allow_negative_prestige:
            self.rew += rew
        else:
            if rew >= 0:
                self.prestige += rew
            else:  # rew < 0
                self.prestige = 0

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def reset(self, new_episode=False):
        self.done = False
        self.active = False
        self.pos = None
        self.carrying = None
        self.mission = ''
        if new_episode:
            self.prestige = 0
            self.bonus_state = None
            self.bonuses = []

    def render(self, img):
        if self.active:
            super().render(img)

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert self.dir >= 0 and self.dir < 4
        return np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        return np.add(self.pos, self.dir_vec)

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        ax -= 2 * self.view_offset * dx
        ay -= 2 * self.view_offset * dy

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_pos(self):
        return self.view_size // 2, self.view_size - 1 - self.view_offset

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        if self.see_through_walls:
            # Make everything easier by using a fixed orientation
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size // 2
        else:
            dir = self.dir
            # Facing right
            if dir == 0:  # 1
                topX = self.pos[0] - self.view_offset
                topY = self.pos[1] - self.view_size // 2
            # Facing down
            elif dir == 1:  # 0
                topX = self.pos[0] - self.view_size // 2
                topY = self.pos[1] - self.view_offset
            # Facing left
            elif dir == 2:  # 3
                topX = self.pos[0] - self.view_size + 1 + self.view_offset
                topY = self.pos[1] - self.view_size // 2
            # Facing up
            elif dir == 3:  # 2
                topX = self.pos[0] - self.view_size // 2
                topY = self.pos[1] - self.view_size + 1 + self.view_offset
            else:
                assert False, 'invalid agent direction'

        botX = topX + self.view_size
        botY = topY + self.view_size

        return topX, topY, botX, botY

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and
        returns the corresponding coordinates
        """
        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """
        return self.relative_coords(x, y) is not None

    def sees(self, x, y):
        raise NotImplementedError

    def process_vis(self, opacity_grid):
        assert len(opacity_grid.shape) == 2
        if not self.see_through_walls:
            return occlude_mask(~opacity_grid, self.get_view_pos())
        else:
            return np.full(opacity_grid.shape, 1, dtype=np.bool)


@numba.njit
def occlude_mask(grid, agent_pos):
    mask = np.zeros(grid.shape[:2]).astype(numba.boolean)
    mask[agent_pos[0], agent_pos[1]] = True
    width, height = grid.shape[:2]

    for j in range(agent_pos[1] + 1, 0, -1):
        for i in range(agent_pos[0], width):
            if mask[i, j] and grid[i, j]:
                if i < width - 1:
                    mask[i + 1, j] = True
                if j > 0:
                    mask[i, j - 1] = True
                    if i < width - 1:
                        mask[i + 1, j - 1] = True

        for i in range(agent_pos[0] + 1, 0, -1):
            if mask[i, j] and grid[i, j]:
                if i > 0:
                    mask[i - 1, j] = True
                if j > 0:
                    mask[i, j - 1] = True
                    if i > 0:
                        mask[i - 1, j - 1] = True

    for j in range(agent_pos[1], height):
        for i in range(agent_pos[0], width):
            if mask[i, j] and grid[i, j]:
                if i < width - 1:
                    mask[i + 1, j] = True
                if j < height - 1:
                    mask[i, j + 1] = True
                    if i < width - 1:
                        mask[i + 1, j + 1] = True

        for i in range(agent_pos[0] + 1, 0, -1):
            if mask[i, j] and grid[i, j]:
                if i > 0:
                    mask[i - 1, j] = True
                if j < height - 1:
                    mask[i, j + 1] = True
                    if i > 0:
                        mask[i - 1, j + 1] = True

    return mask
