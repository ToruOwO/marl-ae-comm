import numpy as np

from ..base import MultiGridEnv, MultiGrid
from ..objects import Goal, Wall


def dis_func(x, y, k=1):
    return np.linalg.norm(x - y) / k


class FindGoalMultiGrid(MultiGridEnv):
    """
    A single cluttered room with a green goal at random position.
    Each agent obtains a reward when reaching the goal.
    All agents must be reach the goal to obtain a team reward.
    """
    mission = 'get to the green square'
    metadata = {}

    def __init__(self, config):
        n_clutter = config.get('n_clutter')
        clutter_density = config.get('clutter_density')
        randomize_goal = config.get('randomize_goal')

        if (n_clutter is None) == (clutter_density is None):
            raise ValueError('Must provide n_clutter or clutter_density.')

        super().__init__(config)

        if clutter_density is not None:
            self.n_clutter = int(
                clutter_density * (self.width - 2) * (self.height - 2))
        else:
            self.n_clutter = n_clutter

        self.randomize_goal = randomize_goal

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        if getattr(self, 'randomize_goal', True):
            goal_pos = self.place_obj(Goal(color='green', reward=1),
                                      max_tries=100)
        else:
            goal_pos = np.asarray([width - 2, height - 2])
            self.put_obj(Goal(color='green', reward=1), width - 2, height - 2)

        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), max_tries=100)

        return goal_pos

    def gen_global_obs(self, agent_done=None):
        if agent_done is None:
            # an integer array storing agent's done info
            agent_done = np.zeros((len(self.agents, )), dtype=np.float)
        self.sees_goal = np.array([self.agents[i].in_view(
                self.goal_pos[0], self.goal_pos[1]) for i in range(
                self.num_agents)]) * 1

        obs = {
            'adv_indices': self.adv_indices,
            'agent_done': agent_done,  # (N,)
            'goal_pos': self.goal_pos,  # (2,)
            'sees_goal': self.sees_goal,  # (N,)
            'pos': np.stack([self.get_agent_pos(a) for a in self.agents],
                            axis=0),  # (N, 2)
            'comm_act': np.stack([a.comm for a in self.agents],
                                 axis=0),  # (N, comm_len)
            'env_act': np.stack([a.env_act for a in self.agents],
                                axis=0),  # (N, 1)
        }
        return obs

    def reset(self):
        obs_dict = MultiGridEnv.reset(self)

        if self.num_adversaries < 0:
            # need to count number of adversaries in the env
            self.adv_indices = set()
            for i, agent in enumerate(self.agents):
                if agent.is_adversary:
                    self.adv_indices.add(i)
            self.num_adversaries = len(self.adv_indices)

            obs_dict['global'] = self.gen_global_obs()
            return obs_dict

        else:
            # randomize adv indices each episode
            adv_indices = np.random.choice([i for i in range(self.num_agents)],
                                           self.num_adversaries,
                                           replace=False)
            for i, agent in enumerate(self.agents):
                if i in adv_indices:
                    agent.is_adversary = True
                else:
                    agent.is_adversary = False
            self.adv_indices = adv_indices

            obs_dict['global'] = self.gen_global_obs()
            return obs_dict

    def _get_reward(self, rwd, agent_no):
        step_rewards = np.zeros((len(self.agents, )), dtype=np.float)
        env_rewards = np.zeros((len(self.agents, )), dtype=np.float)
        if agent_no in self.adv_indices:
            # agent can only receive rewards if it is non-adversarial
            return env_rewards, step_rewards

        env_rewards[agent_no] += rwd
        if self.team_reward_type == 'share':
            # assign zero-sum rewards to both teams
            for agent_id in range(self.num_agents):
                if agent_id not in self.adv_indices:

                    step_rewards[agent_id] += rwd
                    self.agents[agent_id].reward(rwd)
                else:
                    step_rewards[agent_id] -= rwd
                    self.agents[agent_id].reward(-rwd)
        else:
            step_rewards[agent_no] += rwd
            self.agents[agent_no].reward(rwd)
        return env_rewards, step_rewards

    def update_reward(self, step_rewards):
        nonadv_done_n = []
        adv_rew = 0.0
        for i, agent in enumerate(self.agents):
            if i not in self.adv_indices:
                # zero-sum reward between adversaries and non-adversaries
                nonadv_done_n.append(agent.done)
                adv_rew -= step_rewards[i]
        nonadv_done = all(nonadv_done_n)

        timeout = (self.step_count >= self.max_steps)

        # normalized distance-to-goal to range [0, 1]
        ndis_to_goal = [dis_func(agent.pos, self.goal_pos, k=self.max_dis)
                        for agent in self.agents]

        if self.team_reward_type == 'const':
            # give constant team reward to non-adversaries
            if nonadv_done:
                team_rwd = self.team_reward_multiplier
                for i, a in enumerate(self.agents):
                    if i not in self.adv_indices:
                        a.reward(team_rwd)
                        step_rewards[i] += team_rwd

                        # keep zero-sum reward between
                        # adversaries and non-adversaries
                        adv_rew -= team_rwd
        else:
            # no team reward
            pass

        if len(self.adv_indices) > 0:
            adv_rew /= len(self.adv_indices)
            for i in self.adv_indices:
                step_rewards[i] += adv_rew
        return timeout, nonadv_done, step_rewards, ndis_to_goal

    def step(self, action_dict):
        obs_dict, rew_dict, _, info_dict = MultiGridEnv.step(self, action_dict)
        if self.active_after_done:
            done_n = [agent.at_pos(self.goal_pos) for agent in self.agents]
        else:
            done_n = [agent.done for agent in self.agents]

        step_rewards = rew_dict['step_rewards']
        env_rewards = rew_dict['env_rewards']
        comm_rewards = rew_dict['comm_rewards']
        comm_strs = info_dict['comm_strs']

        timeout, nonadv_done, step_rewards, ndis_to_goal = self.update_reward(
            step_rewards)

        # The episode overall is done if ALL non-adversarial agents are done,
        # or if it exceeds the step limit.
        done = timeout or nonadv_done
        if self.debug:
            done = any(done_n)

        step_rewards += comm_rewards

        rew_dict = {f'agent_{i}': step_rewards[i] for i in range(
            len(step_rewards))}
        done_dict = {'__all__': done}
        info_dict = {f'agent_{i}': {
            'done': done_n[i],
            'comm': self.agents[i].comm,
            'nonadv_done': nonadv_done,
            'posd': np.array([self.agents[i].pos[0], self.agents[i].pos[1],
                              done_n[i]]),
            'sees_goal': self.sees_goal[i],
            'comm_str': comm_strs[i],
        } for i in range(len(done_n))}

        info_dict['rew_by_act'] = {
            # env reward
            0: {f'agent_{i}': env_rewards[i] for i in range(len(env_rewards))},

            # designed comm reward
            'comm': {f'agent_{i}': comm_rewards[i] for i in range(len(
                comm_rewards))},
        }

        # team reward
        if self.separate_rew_more:
            info_dict['rew_by_act'][1] = {f'agent_{i}': (
                    step_rewards[i] - env_rewards[i]) for i in range(
                len(step_rewards))}
        else:
            info_dict['rew_by_act'][1] = {f'agent_{i}': (
                step_rewards[i]) for i in range(len(step_rewards))}

        obs_dict['global'] = self.gen_global_obs()
        return obs_dict, rew_dict, done_dict, info_dict
