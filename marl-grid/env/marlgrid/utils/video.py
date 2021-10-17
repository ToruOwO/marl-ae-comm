import gym
import numpy as np
import os
import tqdm
import cv2


def export_video(X, outfile, fps=30, rescale_factor=2):

    try:
        import moviepy.editor as mpy
    except:
        raise ImportError(
            "GridRecorder requires moviepy library. Try installing:\n $ pip install moviepy"
        )

    if isinstance(X, list):
        X = np.stack(X)

    if isinstance(X, np.float) and X.max() < 1:
        X = (X * 255).astype(np.uint8).clip(0, 255)

    if rescale_factor is not None and rescale_factor != 1:
        print('`rescale_factor` is deprecated in `export_video`. ' +\
              'use `env.video_scale` instead. ')

    def make_frame(i):
        out = X[i]
        return out

    getframe = lambda t: make_frame(min(int(t * fps), len(X) - 1))
    clip = mpy.VideoClip(getframe, duration=len(X) / fps)

    outfile = os.path.abspath(os.path.expanduser(outfile))
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    clip.write_videofile(outfile, fps=fps)


def render_frames(X, path, ext="png"):
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Error importing from PIL in export_frames. Try installing PIL:\n $ pip install Pillow"
        )

    # If the path has a file extension, dump frames in a new directory with = path minus extension
    if "." in os.path.basename(path):
        path = os.path.splitext(path)[0]
    if not os.path.isdir(path):
        os.makedirs(path)

    for k, frame in tqdm.tqdm(enumerate(X), total=len(X)):
        Image.fromarray(frame, "RGB").save(os.path.join(path, f"frame_{k}.{ext}"))


class GridRecorder(gym.core.Wrapper):
    default_max_len = 1000
    default_video_kwargs = {
        'fps': 20,
        'rescale_factor': 1,
    }
    def __init__(
            self,
            env,
            save_root,
            max_steps=1000,
            auto_save_images=True,
            auto_save_videos=True,
            auto_save_interval=None,
            render_kwargs={},
            video_kwargs={}
            ):
        super().__init__(env)

        self.frames = None
        self.ptr = 0
        self.reset_count = 0
        self.last_save = -10000
        self.recording = False
        self.save_root = self.fix_path(save_root)
        self.auto_save_videos = auto_save_videos
        self.auto_save_images = auto_save_images
        self.auto_save_interval = auto_save_interval
        self.render_kwargs = render_kwargs
        self.video_kwargs = {**self.default_video_kwargs, **video_kwargs}
        self.n_parallel = getattr(env, 'num_envs', 1)

        self.video_scale = 8 # default setting to see text clearly
        self.render_reward = True

        if max_steps is None:
            if hasattr(env, "max_steps") and env.max_steps != 0:
                self.max_steps = env.max_steps + 1
            else:
                self.max_steps = self.default_max_steps + 1
        else:
            self.max_steps = max_steps + 1

    @staticmethod
    def fix_path(path):
        return os.path.abspath(os.path.expanduser(path))

    @property
    def should_record(self):
        if self.recording:
            return True
        if self.auto_save_interval is None:
            return False
        return (self.reset_count - self.last_save) >= self.auto_save_interval

    def export_frames(self,  episode_id=None, save_root=None):
        if save_root is None:
            save_root = self.save_root
        if episode_id is None:
            episode_id = f'frames_{self.reset_count}'
        render_frames(self.frames[:self.ptr], os.path.join(self.fix_path(save_root), episode_id))

    def export_video(self, episode_id=None, save_root=None):
        if save_root is None:
            save_root = self.save_root
        if episode_id is None:
            episode_id = f'video_{self.reset_count}.mp4'
        export_video(self.frames[:self.ptr],
                     os.path.join(self.fix_path(save_root), episode_id),
                     **self.video_kwargs)

    def export_both(self, episode_id, save_root=None):
        self.export_frames(f'{episode_id}_frames', save_root=save_root)
        self.export_video(f'{episode_id}.mp4', save_root=save_root)

    def reset(self, **kwargs):
        if self.should_record and self.ptr>0:
            self.append_current_frame()
            if self.auto_save_images:
                self.export_frames()
            if self.auto_save_videos:
                self.export_video()
            self.last_save = self.reset_count
        del self.frames
        self.frames = None
        self.ptr = 0
        self.reset_count += self.n_parallel
        return self.env.reset(**kwargs)

    def append_current_frame(self, reward_dict=None, info_dict=None):
        if self.should_record:
            new_frame = self.env.render(mode="rgb_array", show_more=True,
                                        **self.render_kwargs)

            if isinstance(new_frame, list) or len(new_frame.shape) > 3:
                new_frame = new_frame[0]

            if self.video_scale != 1:
                new_frame = cv2.resize(new_frame, None,
                                       fx=self.video_scale,
                                       fy=self.video_scale,
                                       interpolation=cv2.INTER_AREA)

            if self.render_reward and reward_dict is not None:
                to_render = ['rew',
                             *[f'{k[0]+k[-1]}: {v:.3f}' for k, v in
                               reward_dict.items()]
                             ]
                str_spacing = 30
                for i, text_to_render in enumerate(to_render):
                    cv2.putText(new_frame, text_to_render,
                                (int(0.8 * new_frame.shape[1]),
                                 int(0.1 * new_frame.shape[0]) + (i * str_spacing)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if self.frames is None:
                self.frames = np.zeros(
                    (self.max_steps, *new_frame.shape), dtype=new_frame.dtype
                )

            self.frames[self.ptr] = new_frame
            self.ptr += 1

    def step(self, action):
        if self.ptr == 0:
            self.append_current_frame()
        obs, rew, done, info = self.env.step(action)
        self.append_current_frame(reward_dict=rew, info_dict=info)
        return obs, rew, done, info

