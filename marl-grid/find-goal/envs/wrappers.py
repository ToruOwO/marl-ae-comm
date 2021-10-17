import gym
import cv2
import numpy as np


class DictObservationNormalizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        return

    def step(self, action):
        obs_dict, rew_dict, done_dict, info_dict = self.env.step(action)
        for k, v in obs_dict.items():
            if k == 'global':
                continue

            if isinstance(v, dict):
                obs_dict[k]['pov'] = (2. * ((v['pov'] / 255.) - 0.5))
            else:
                obs_dict[k] = (2. * ((v / 255.) - 0.5))
        return obs_dict, rew_dict, done_dict, info_dict


class GridWorldEvaluatorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.video_scale = 2
        self.show_reward = True
        return

    def get_raw_obs(self):
        frame = self.env.render(mode='rgb_array', show_more=self.show_reward)
        frame = frame.astype(np.uint8)
        frame = self.resize(frame)
        obs, rew, _, info = self._out

        frame = self.render_reward(frame, rew, info)
        return frame

    def step(self, action):
        self._out = self.env.step(action)
        return self._out

    def resize(self, frame):
        if self.video_scale != 1:
            frame = cv2.resize(frame, None,
                               fx=self.video_scale,
                               fy=self.video_scale,
                               interpolation=cv2.INTER_AREA)
        return frame

    def render_reward(self, frame, reward_dict, info_dict):
        if self.show_reward:
            # render reward
            to_render = ['env rew',
                         *[f'{k[0] + k[-1]}: {v:.3f}' for k, v in
                           reward_dict.items()],
                         'comm rew',
                         *[f'{k[0] + k[-1]}: {v:.3f}' for k, v in
                           info_dict['rew_by_act'][1].items()]
                         ]

            # render communication
            for k, v in info_dict.items():
                if k[-1].isdigit():
                    to_render += [str(k[0] + k[-1]) + ': ' + str(v['comm'])]

            for k, v in info_dict.items():
                if k[-1].isdigit() and v['comm_str'] != '':
                    to_render += [(str(k[0] + k[-1]) + ' ---'),
                                  *(v['comm_str'])]

            str_spacing = 30
            x_start = ((frame.shape[1] - frame.shape[0]
                        ) // 2) + frame.shape[0] + 10
            y_start = int(0.1 * frame.shape[0])
            for i, text_to_render in enumerate(to_render):
                cv2.putText(frame, text_to_render,
                            (x_start, y_start + (i * str_spacing)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
