"""Multi-agent grid world.
Based on MiniGrid: https://github.com/maximecb/gym-minigrid.
"""

import gym
import gym_minigrid
import math
import numpy as np
import warnings
from enum import IntEnum
from gym_minigrid.rendering import fill_coords, point_in_rect, downsample, \
    highlight_img

from .agents import GridAgentInterface
from .objects import Wall, Goal, COLORS

TILE_PIXELS = 32


class ObjectRegistry:
    """
    This class contains dicts that map objects to numeric keys and vise versa.
    Used so that grid worlds can represent objects using numerical arrays rather
    than lists of lists of generic objects.
    """

    def __init__(self, objs=[], max_num_objects=1000):
        self.key_to_obj_map = {}
        self.obj_to_key_map = {}
        self.max_num_objects = max_num_objects
        for obj in objs:
            self.add_object(obj)

    def get_next_key(self):
        for k in range(self.max_num_objects):
            if k not in self.key_to_obj_map:
                break
        else:
            raise ValueError('Object registry full.')
        return k

    def __len__(self):
        return len(self.id_to_obj_map)

    def add_object(self, obj):
        new_key = self.get_next_key()
        self.key_to_obj_map[new_key] = obj
        self.obj_to_key_map[obj] = new_key
        return new_key

    def contains_object(self, obj):
        return obj in self.obj_to_key_map

    def contains_key(self, key):
        return key in self.key_to_obj_map

    def get_key(self, obj):
        if obj in self.obj_to_key_map:
            return self.obj_to_key_map[obj]
        else:
            return self.add_object(obj)

    def obj_of_key(self, key):
        return self.key_to_obj_map[key]


def rotate_grid(grid, rot_k):
    """
    Basically replicates np.rot90 (with the correct args for rotating images).
    But it's faster.
    """
    rot_k = rot_k % 4
    if rot_k == 3:
        return np.moveaxis(grid[:, ::-1], 0, 1)
    elif rot_k == 1:
        return np.moveaxis(grid[::-1, :], 0, 1)
    elif rot_k == 2:
        return grid[::-1, ::-1]
    else:
        return grid


class MultiGrid:
    tile_cache = {}

    def __init__(self, shape, obj_reg=None, orientation=0):
        self.orientation = orientation
        if isinstance(shape, tuple):
            self.width, self.height = shape
            self.grid = np.zeros((self.width, self.height),
                                 dtype=np.uint8)
        elif isinstance(shape, np.ndarray):
            self.width, self.height = shape.shape
            self.grid = shape
        else:
            raise ValueError('Must create grid from shape tuple or array.')

        if self.width < 3 or self.height < 3:
            raise ValueError('Grid needs width, height >= 3')

        self.obj_reg = ObjectRegistry(
            objs=[None]) if obj_reg is None else obj_reg

    @property
    def opacity(self):
        transparent_fun = np.vectorize(lambda k: (
            self.obj_reg.key_to_obj_map[k].see_behind() if hasattr(
                self.obj_reg.key_to_obj_map[k],
                'see_behind') else True))
        return ~transparent_fun(self.grid)

    def __getitem__(self, *args, **kwargs):
        return self.__class__(
            np.ndarray.__getitem__(self.grid, *args, **kwargs),
            obj_reg=self.obj_reg,
            orientation=self.orientation,
        )

    def rotate_left(self, k=1):
        return self.__class__(
            rotate_grid(self.grid, rot_k=k),  # np.rot90(self.grid, k=k),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - k) % 4,
        )

    def slice(self, topX, topY, width, height, rot_k=0):
        """
        Get a subset of the grid
        """
        sub_grid = self.__class__(
            (width, height),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - rot_k) % 4,
        )
        x_min = max(0, topX)
        x_max = min(topX + width, self.width)
        y_min = max(0, topY)
        y_max = min(topY + height, self.height)

        x_offset = x_min - topX
        y_offset = y_min - topY
        sub_grid.grid[
          x_offset: x_max - x_min + x_offset, y_offset: y_max - y_min + y_offset
        ] = self.grid[x_min:x_max, y_min:y_max]

        sub_grid.grid = rotate_grid(sub_grid.grid, rot_k)

        sub_grid.width, sub_grid.height = sub_grid.grid.shape

        return sub_grid

    def set(self, i, j, obj):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[i, j] = self.obj_reg.get_key(obj)

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height

        return self.obj_reg.key_to_obj_map[self.grid[i, j]]

    def test_get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height

        return self.grid[i, j]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h, obj_type=Wall):
        self.horz_wall(x, y, w, obj_type=obj_type)
        self.horz_wall(x, y + h - 1, w, obj_type=obj_type)
        self.vert_wall(x, y, h, obj_type=obj_type)
        self.vert_wall(x + w - 1, y, h, obj_type=obj_type)

    def __str__(self):
        render = (
            lambda x: '  '
            if x is None or not hasattr(x, 'str_render')
            else x.str_render(dir=self.orientation)
        )
        hstars = '*' * (2 * self.width + 2)
        return (
                hstars
                + '\n'
                + '\n'.join(
            '*' + ''.join(
                render(self.get(i, j)) for i in range(self.width)) + '*'
            for j in range(self.height)
        )
                + '\n'
                + hstars
        )

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        array[i, j, :] = 0
                    else:
                        array[i, j, :] = v.encode()
        return array

    @classmethod
    def decode(cls, array):
        raise NotImplementedError
        width, height, channels = array.shape
        assert channels == 3
        vis_mask[i, j] = np.ones(shape=(width, height), dtype=np.bool)
        grid = cls((width, height))

    @classmethod
    def cache_render_fun(cls, key, f, *args, **kwargs):
        if key not in cls.tile_cache:
            cls.tile_cache[key] = f(*args, **kwargs)
        return np.copy(cls.tile_cache[key])

    @classmethod
    def cache_render_obj(cls, obj, tile_size, subdivs):
        if obj is None:
            return cls.cache_render_fun((tile_size, None), cls.empty_tile,
                                        tile_size, subdivs)
        else:
            img = cls.cache_render_fun(
                (tile_size, obj.__class__.__name__, *obj.encode()),
                cls.render_object, obj, tile_size, subdivs
            )
            if hasattr(obj, 'render_post'):
                return obj.render_post(img)
            else:
                return img

    @classmethod
    def empty_tile(cls, tile_size, subdivs):
        alpha = max(0, min(20, tile_size - 10))
        img = np.full((tile_size, tile_size, 3), alpha, dtype=np.uint8)
        img[1:, :-1] = 0
        return img

    @classmethod
    def render_object(cls, obj, tile_size, subdivs):
        img = np.zeros((tile_size * subdivs, tile_size * subdivs, 3),
                       dtype=np.uint8)
        obj.render(img)
        return downsample(img, subdivs).astype(np.uint8)

    @classmethod
    def blend_tiles(cls, img1, img2):
        """
        This function renders one "tile" on top of another.
        Assumes img2 is a downscaled monochromatic with a (0, 0, 0) background.
        """
        alpha = img2.sum(2, keepdims=True)
        max_alpha = alpha.max()
        if max_alpha == 0:
            return img1
        return (
                ((img1 * (max_alpha - alpha)) + (img2 * alpha)
                 ) / max_alpha
        ).astype(img1.dtype)

    @classmethod
    def render_tile(cls, obj, tile_size=TILE_PIXELS, subdivs=3, top_agent=None):
        subdivs = 3

        if obj is None:
            img = cls.cache_render_obj(obj, tile_size, subdivs)
        else:
            if ('Agent' in obj.type) and (top_agent in obj.agents):
                # If the tile is a stack of agents that includes the top agent,
                # then just render the top agent.
                img = cls.cache_render_obj(top_agent, tile_size, subdivs)
            elif 'Goal' in obj.type:
                # `blend_tiles`
                # Do not render any agent on top of a goal tile
                img = cls.cache_render_obj(obj, tile_size, subdivs)
            else:
                # Otherwise, render (+ downsize) the item in the tile.
                img = cls.cache_render_obj(obj, tile_size, subdivs)
                # If the base obj isn't an agent but has agents on top, render
                # an agent and blend it in.
                if len(obj.agents) > 0 and 'Agent' not in obj.type:
                    if top_agent in obj.agents:
                        img_agent = cls.cache_render_obj(top_agent, tile_size,
                                                         subdivs)
                    else:
                        img_agent = cls.cache_render_obj(obj.agents[0],
                                                         tile_size, subdivs)
                    img = cls.blend_tiles(img, img_agent)

            # Render the tile border if any of the corners are black.
            if (img[([0, 0, -1, -1], [0, -1, 0, -1])] == 0).all(axis=-1).any():
                img = img + cls.cache_render_fun((tile_size, None),
                                                 cls.empty_tile, tile_size,
                                                 subdivs)
        return img

    def render(self, tile_size, highlight_mask=None, visible_mask=None,
               top_agent=None):
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px), dtype=np.uint8
                       )[..., None] + COLORS['shadow']

        for j in range(0, self.height):
            for i in range(0, self.width):
                if visible_mask is not None and not visible_mask[i, j]:
                    continue
                obj = self.get(i, j)

                tile_img = MultiGrid.render_tile(
                    obj,
                    tile_size=tile_size,
                    top_agent=top_agent
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size

                img[ymin:ymax, xmin:xmax, :] = rotate_grid(tile_img,
                                                           self.orientation)

        if highlight_mask is not None:
            hm = np.kron(highlight_mask.T,
                         np.full((tile_size, tile_size), 255, dtype=np.uint16)
                         )[..., None]
            img = np.right_shift(
                img.astype(np.uint16) * 8 + hm * 2, 3
            ).clip(0, 255).astype(np.uint8)

        return img


class MultiGridEnv(gym.Env):
    """Multi-agent grid world env interface that is compatible with RLlib.

    Agents are identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib agents.
    """

    def __init__(self, config, width=None, height=None):

        agents = config.get('agents')
        grid_size = config.get('grid_size')
        max_steps = config.get('max_steps')
        seed = config.get('seed')
        respawn = config.get('respawn')
        ghost_mode = config.get('ghost_mode')
        agent_spawn_kwargs = config.get('agent_spawn_kwargs', {})
        team_reward_multiplier = config.get('team_reward_multiplier')
        team_reward_type = config.get('team_reward_type')
        team_reward_freq = config.get('team_reward_freq')
        active_after_done = config.get('active_after_done')
        discrete_position = config.get('discrete_position')
        separate_rew_more = config.get('separate_rew_more')

        self.debug = config.get('debug', False)

        if grid_size and not width and not height:
            width, height = grid_size, grid_size

        self.respawn = respawn

        self.window = None

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.seed(seed=seed)
        self.agent_spawn_kwargs = agent_spawn_kwargs
        self.ghost_mode = ghost_mode
        self.team_reward_multiplier = team_reward_multiplier
        self.team_reward_type = team_reward_type
        self.team_reward_freq = team_reward_freq
        self.active_after_done = active_after_done
        self.max_dis = np.sqrt(np.sum(np.square([self.width, self.height])))
        self.discrete_position = discrete_position
        self.separate_rew_more = separate_rew_more

        self.agents = []

        for i, agent in enumerate(agents):
            self.add_agent(agent)

        self.num_adversaries = -1

        self.comm_dim = self.agents[0].comm_dim
        self.comm_len = self.agents[0].comm_len
        self.discrete_comm = self.agents[0].discrete_comm

        self.reset()

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):
        # return a single agent's action space
        # (assume the same action space for all agents)
        return self.agents[0].action_space

    @property
    def observation_space(self):
        # return a single agent's obs space
        # (assume the same observation space for all agents)
        agent_pos_space = gym.spaces.MultiDiscrete(
                    [self.width, self.height])
        pos_space = [gym.spaces.MultiDiscrete(
            [self.width, self.height]) for _ in range(self.num_agents)]
        pos_space = gym.spaces.Tuple(pos_space)

        obs_space = self.agents[0].observation_space
        obs_style = self.agents[0].observation_style

        if self.discrete_position and obs_style == 'dict':
            if 'position' in obs_space.spaces:
                obs_space.spaces['position'] = pos_space
            if 'selfpos' in obs_space.spaces:
                obs_space.spaces['selfpos'] = agent_pos_space
        return obs_space

    @property
    def num_agents(self):
        return len(self.agents)

    def set_team_reward_multiplier(self, x):
        self.team_reward_multiplier = x

    def add_agent(self, agent_interface):
        if isinstance(agent_interface, dict):
            self.agents.append(GridAgentInterface(**agent_interface))
        elif isinstance(agent_interface, GridAgentInterface):
            self.agents.append(agent_interface)
        else:
            raise ValueError(
                'To add an agent to a marlgrid environment, call add_agent'
                ' with either a GridAgentInterface object '
                ' or a dictionary that can be used to initialize one.')

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        for agent in self.agents:
            agent.agents = []
            agent.reset(new_episode=True)

        self.goal_pos = self._gen_grid(self.width, self.height)

        for agent in self.agents:
            if agent.spawn_delay == 0:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        self.step_count = 0
        obs = self.gen_obs()

        obs_dict = {f'agent_{i}': obs[i] for i in range(len(obs))}
        return obs_dict

    def gen_obs_grid(self, agent):
        if not agent.active:
            raise ValueError('Agent must be active to have obs grid.')

        topX, topY, botX, botY = agent.get_view_exts()

        if agent.see_through_walls:
            # return view without rotation
            grid = self.grid.slice(topX, topY, agent.view_size, agent.view_size)
        else:
            grid = self.grid.slice(topX, topY, agent.view_size, agent.view_size,
                                   rot_k=agent.dir + 1)

        # Process occluders and visibility
        # Note that this incurs some slight performance cost
        vis_mask = agent.process_vis(grid.opacity)

        # Warning about the rest of the function:
        # Allows masking away objects that the agent isn't supposed to see.
        # But breaks consistency between the states of the grid objects in the
        # parial views and the grid objects overall.
        if len(getattr(agent, 'hide_item_types', [])) > 0:
            for i in range(grid.width):
                for j in range(grid.height):
                    item = grid.get(i, j)
                    if (item is not None) and (item is not agent) and (
                            item.type in agent.hide_item_types):
                        if len(item.agents) > 0:
                            grid.set(i, j, item.agents[0])
                        else:
                            grid.set(i, j, None)

        return grid, vis_mask

    def gen_agent_comm_obs(self, agent):
        comm = []
        for a in self.agents:
            if a == agent:
                continue
            else:
                comm.append(a.comm)
        comm.append(agent.comm)

        comm = np.stack(comm, axis=0)  # (num_agents, comm_len)
        return comm

    def gen_agent_done_obs(self, agent):
        d = []
        for i, a in enumerate(self.agents):
            if a == agent:
                continue
            else:
                d.append(a.done)
        d.append(agent.done)
        return np.array(d).astype(int)  # (num_agents,)

    def get_agent_pos(self, agent):
        # return a valid and normalized agent position
        agent_pos = agent.pos if agent.pos is not None else (0, 0)
        if self.discrete_position:
            return np.array(agent_pos)
        agent_pos = np.array(agent_pos) / np.array(
            [self.width, self.height], dtype=np.float)
        return agent_pos

    def gen_agent_pos_obs(self, agent):
        pos = []
        for a in self.agents:
            if a == agent:
                continue
            else:
                pos.append(self.get_agent_pos(a))
        pos.append(self.get_agent_pos(agent))

        # shape (num_agents, 2)
        return np.stack(pos, axis=0)

    def gen_agent_obs(self, agent, image_only=False):
        """
        Generate agent's view (partially observable, low-resolution encoding)
        """
        if not agent.active:
            image_px = agent.view_size * agent.view_tile_size
            grid_image = np.zeros(shape=(image_px, image_px),
                                  dtype=np.uint8)[..., None] + COLORS['shadow']
        else:
            grid, vis_mask = self.gen_obs_grid(agent)
            grid_image = grid.render(tile_size=agent.view_tile_size,
                                     visible_mask=vis_mask,
                                     top_agent=agent)
        if agent.observation_style == 'image' or image_only:
            return grid_image
        elif agent.observation_style == 'dict':
            ret = {'pov': grid_image}
            if agent.observe_rewards:
                ret['reward'] = getattr(agent, 'step_reward', 0)
            if agent.observe_position:
                ret['position'] = self.gen_agent_pos_obs(agent)
            if agent.observe_self_position:
                ret['selfpos'] = self.get_agent_pos(agent)
            if agent.observe_done:
                ret['done'] = self.gen_agent_done_obs(agent)
            if agent.observe_self_env_act:
                ret['self_env_act'] = agent.env_act
            if agent.observe_orientation:
                agent_dir = agent.dir if agent.dir is not None else 0
                ret['orientation'] = agent_dir
            if agent.observe_t:
                ret['t'] = self.step_count / self.max_steps
            if agent.observe_identity:
                ret['identity'] = agent.is_adversary
            if agent.observe_comm:
                ret['comm'] = self.gen_agent_comm_obs(agent)
            return ret
        elif agent.observation_style == 'tuple':
            ret = (grid_image,)
            if agent.observe_rewards:
                rew = getattr(agent, 'step_reward', 0)
                ret += (rew,)
            if agent.observe_position:
                pos = self.gen_agent_pos_obs(agent)
                ret += (pos,)
            if agent.observe_self_position:
                ret += self.get_agent_pos(agent)
            if agent.observe_done:
                done_obs = self.gen_agent_done_obs(agent)
                ret += (done_obs,)
            if agent.observe_self_env_act:
                ret += (agent.env_act,)
            if agent.observe_orientation:
                agent_dir = agent.dir if agent.dir is not None else 0
                ret += (agent_dir,)
            if agent.observe_t:
                ret += (self.step_count / self.max_steps,)
            if agent.observe_identity:
                ret += (agent.is_adversary,)
            if agent.observe_comm:
                ret += (self.gen_agent_comm_obs(agent),)
            return ret
        else:
            raise ValueError('Unsupported observation style.')

    def gen_obs(self, image_only=False):
        obs = [self.gen_agent_obs(agent, image_only) for agent in self.agents]
        return obs

    def __str__(self):
        return self.grid.__str__()

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                '__all__' (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        # Spawn agents if it's time.
        for agent in self.agents:
            if not agent.active and not agent.done and \
                    self.step_count >= agent.spawn_delay:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        actions = [action_dict[f'agent_{i}'] for i in range(len(self.agents))]

        step_rewards = np.zeros((len(self.agents, )), dtype=np.float)
        env_rewards = np.zeros((len(self.agents, )), dtype=np.float)

        comm_rewards = np.zeros((len(self.agents, )), dtype=np.float)
        comm_strs = ['' for _ in range(self.num_agents)]

        self.step_count += 1

        iter_agents = list(enumerate(zip(self.agents, actions)))
        iter_order = np.arange(len(iter_agents))
        self.np_random.shuffle(iter_order)
        for shuffled_ix in iter_order:
            agent_no, (agent, action) = iter_agents[shuffled_ix]
            agent.step_reward = 0

            if self.comm_dim > 0 and self.comm_len > 0:
                assert len(action) == 2
                assert len(action[1]) == self.comm_len
                agent.env_act = action[0]
                agent.comm = action[1]
                action = agent.env_act
            else:
                agent.env_act = action

            if agent.active:
                cur_pos = agent.pos[:]
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos[:]
                fwd_cell = self.grid.get(*fwd_pos)
                agent_moved = False

                # right, down, left, up
                if action in {0, 1, 2, 3}:
                    # update direction
                    agent.dir = action

                    # move forward (if allowed)
                    can_move = fwd_cell is None or fwd_cell.can_overlap()
                    if can_move:
                        agent_moved = True
                        # Add agent to new cell
                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent)
                            agent.pos = fwd_pos
                        else:
                            fwd_cell.agents.append(agent)
                            agent.pos = fwd_pos

                        # Remove agent from old cell
                        if cur_cell == agent:
                            self.grid.set(*cur_pos, None)
                        else:
                            assert cur_cell.can_overlap()
                            cur_cell.agents.remove(agent)

                        # Add agent's agents to old cell
                        for left_behind in agent.agents:
                            cur_obj = self.grid.get(*cur_pos)
                            if cur_obj is None:
                                self.grid.set(*cur_pos, left_behind)
                            elif cur_obj.can_overlap():
                                cur_obj.agents.append(left_behind)
                            else:  # How was "agent" there in the first place?
                                raise ValueError('?!?!?!')

                        # After moving, the agent shouldn't
                        # contain any other agents.
                        agent.agents = []

                        # agent can only receive rewards if fwd_cell has a
                        # "get_reward" method && the agent is not already done
                        if not agent.done:
                            if hasattr(fwd_cell, 'get_reward'):
                                rwd = fwd_cell.get_reward(agent)
                                env_rew, step_rew = self._get_reward(
                                    rwd, agent_no)
                                env_rewards += env_rew
                                step_rewards += step_rew

                        if isinstance(fwd_cell, Goal):
                            agent.done = True

                # Pick up an object
                elif action == agent.actions.pickup:
                    if fwd_cell and fwd_cell.can_pickup():
                        if agent.carrying is None:
                            agent.carrying = fwd_cell
                            agent.carrying.cur_pos = np.array([-1, -1])
                            self.grid.set(*fwd_pos, None)
                    else:
                        pass

                # Drop an object
                elif action == agent.actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None
                    else:
                        pass

                # Toggle/activate an object
                elif action == agent.actions.toggle:
                    if fwd_cell:
                        fwd_cell.toggle(agent, fwd_pos)
                    else:
                        pass

                # Done action (same as "do nothing")
                elif action == agent.actions.done:
                    pass

                else:
                    raise ValueError(
                        f'Environment cannot handle action {action}.')

                agent.on_step(fwd_cell if agent_moved else None)

        # If any of the agents individually are "done" (hit Goal)
        # but the env requires respawning, respawn those agents.
        for agent in self.agents:
            if agent.done:
                if self.active_after_done:
                    assert agent.active
                elif self.respawn:
                    resting_place_obj = self.grid.get(*agent.pos)
                    if resting_place_obj == agent:
                        if agent.agents:
                            self.grid.set(*agent.pos, agent.agents[0])
                            agent.agents[0].agents += agent.agents[1:]
                        else:
                            self.grid.set(*agent.pos, None)
                    else:
                        resting_place_obj.agents.remove(agent)
                        resting_place_obj.agents += agent.agents[:]
                        agent.agents = []

                    agent.reset(new_episode=False)
                    self.place_obj(agent, **self.agent_spawn_kwargs)
                    agent.activate()
                else:  # if the agent shouldn't be respawned, deactivate it.
                    agent.deactivate()

        rew_dict = {'step_rewards': step_rewards,
                    'env_rewards': env_rewards,
                    'comm_rewards': comm_rewards}

        obs = self.gen_obs()
        obs_dict = {f'agent_{i}': obs[i] for i in range(len(obs))}

        info_dict = {'comm_strs': comm_strs}

        return obs_dict, rew_dict, {}, info_dict

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid.
        Replace anything that is already there.
        """
        self.grid.set(i, j, obj)
        if obj is not None:
            obj.set_position((i, j))
        return True

    def try_place_obj(self, obj, pos):
        """ Try to place an object at a certain position in the grid.
        If it is possible, then do so and return True.
        Otherwise do nothing and return False. """
        # grid_obj: whatever object is already at pos.
        grid_obj = self.grid.get(*pos)

        # If the target position is empty, the object can always be placed.
        if grid_obj is None:
            self.grid.set(*pos, obj)
            obj.set_position(pos)
            return True

        # Otherwise only agents can be placed, and only
        # if the target position can_overlap.
        if not (grid_obj.can_overlap() and obj.is_agent):
            return False

        if grid_obj.is_agent or len(grid_obj.agents) > 0:
            return False

        grid_obj.agents.append(obj)
        obj.set_position(pos)
        return True

    def place_obj(self, obj, top=(0, 0), size=None, reject_fn=None,
                  max_tries=1e5):
        max_tries = int(max(1, min(max_tries, 1e5)))
        top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        bottom = (min(top[0] + size[0], self.grid.width),
                  min(top[1] + size[1], self.grid.height))

        for _ in range(max_tries):
            pos = self.np_random.randint(top, bottom)
            if (reject_fn is not None) and reject_fn(pos):
                continue
            else:
                if self.try_place_obj(obj, pos):
                    break
        else:
            raise RecursionError('Rejection sampling failed in place_obj.')

        return pos

    def render(
            self,
            mode='human',
            close=False,
            highlight=True,
            tile_size=TILE_PIXELS,
            show_agent_views=True,
            max_agents_per_col=3,
            agent_col_width_frac=0.3,
            agent_col_padding_px=2,
            pad_grey=100,
            show_more=False,
    ):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            from gym.envs.classic_control.rendering import SimpleImageViewer

            self.window = SimpleImageViewer(caption='Marlgrid')

        # Compute which cells are visible to the agent
        highlight_mask = np.full((self.width, self.height), False,
                                 dtype=np.bool)
        for agent in self.agents:
            if agent.active:
                xlow, ylow, xhigh, yhigh = agent.get_view_exts()
                dxlow, dylow = max(0, 0 - xlow), max(0, 0 - ylow)
                dxhigh, dyhigh = max(0, xhigh - self.grid.width),\
                                 max(0, yhigh - self.grid.height)
                if agent.see_through_walls:
                    highlight_mask[xlow + dxlow:xhigh - dxhigh,
                    ylow + dylow:yhigh - dyhigh] = True
                else:
                    a, b = self.gen_obs_grid(agent)
                    highlight_mask[xlow + dxlow:xhigh - dxhigh,
                    ylow + dylow:yhigh - dyhigh] |= (
                        rotate_grid(b, a.orientation)[
                        dxlow:(xhigh - xlow) - dxhigh,
                        dylow:(yhigh - ylow) - dyhigh]
                    )

        # Render the whole grid
        img = self.grid.render(
            tile_size, highlight_mask=highlight_mask if highlight else None
        )
        rescale = lambda X, rescale_factor=2: np.kron(
            X, np.ones((int(rescale_factor), int(rescale_factor), 1))
        )

        if show_agent_views:

            target_partial_width = int(
                img.shape[0] * agent_col_width_frac - 2 * agent_col_padding_px)
            target_partial_height = (img.shape[1] - 2 * agent_col_padding_px
                                     ) // max_agents_per_col
            agent_views = self.gen_obs(image_only=True)
            agent_views = [
                rescale(view, min(target_partial_width / view.shape[0],
                                  target_partial_height / view.shape[1])) for
                view in
                agent_views]
            agent_views = [agent_views[pos:pos + max_agents_per_col] for pos in
                           range(0, len(agent_views), max_agents_per_col)]

            f_offset = lambda view: np.array(
                [target_partial_height - view.shape[1],
                 target_partial_width - view.shape[0]]) // 2

            cols = []
            for col_views in agent_views:
                if show_more:
                    col = np.full((img.shape[0],
                                   2 * target_partial_width +
                                   4 * agent_col_padding_px,
                                   3),
                                  pad_grey, dtype=np.uint8)
                else:
                    col = np.full((img.shape[0],
                                   target_partial_width +
                                   2 * agent_col_padding_px,
                                   3),
                                  pad_grey, dtype=np.uint8)
                for k, view in enumerate(col_views):
                    offset = f_offset(view) + agent_col_padding_px
                    offset[0] += k * target_partial_height
                    col[
                    offset[0]:offset[0] + view.shape[0],
                    offset[1]:offset[1] + view.shape[1], :] = view
                cols.append(col)

            img = np.concatenate((img, *cols), axis=1)

        if mode == 'human':
            if not self.window.isopen:
                self.window.imshow(img)
                self.window.window.set_caption('Marlgrid')
            else:
                self.window.imshow(img)

        return img
