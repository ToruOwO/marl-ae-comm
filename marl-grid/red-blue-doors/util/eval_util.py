import numpy as np
import matplotlib.pyplot as plt


def plot_ents(ents, reward, max_time, save_path, max_ent=None, t_red_door=None):
    plt.clf()

    t = ents.shape[0]
    n = ents.shape[1]
    x = np.arange(1, t + 1)

    title = f't={max_time}, r={reward}'
    if t_red_door:
        title += f', t_r={t_red_door}'

    if len(ents.shape) == 2:
        # num_act == 1
        fig = plt.gcf()
        fig.set_size_inches(10, 5)

        for aid in range(n):
            plt.plot(x, ents[:, aid].flatten(), label=f'Agent{aid}')

        ax = plt.gca()
        if max_ent is not None:
            ax.set_ylim([-0.2, max_ent[0] + 0.2])
            ax.hlines(y=max_ent[0], xmin=0, xmax=t, colors='r', linestyles='--')

        if t_red_door and t_red_door > 0:
            plt.axvline(x=t_red_door, ymin=0, ymax=max_ent[0], color='r',
                        linestyle='--')
        plt.xlabel('t')
        plt.ylabel('ent')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(title)

    else:
        plt.cla()
        # (T, N, 1 + comm_len)
        assert len(ents.shape) == 3

        fig, axs = plt.subplots(ents.shape[-1],
                                figsize=(10, ents.shape[-1] * 5))

        for i in range(ents.shape[-1]):

            for aid in range(n):
                axs[i].plot(x, ents[:, aid, i].flatten(), label=f'Agent{aid}')

            if i == 0:
                axs[i].set_title('env-act')
            else:
                axs[i].set_title(f'comm-act-{i}')

            axs[i].set(xlabel='t', ylabel='ent')
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            if max_ent is not None:
                if i >= len(max_ent):
                    max_ent_idx = len(max_ent) - 1
                else:
                    max_ent_idx = i
                axs[i].set_ylim([-0.2, max_ent[max_ent_idx] + 0.2])
                axs[i].hlines(y=max_ent[max_ent_idx], xmin=0, xmax=t,
                              colors='r', linestyles='--')

        fig.suptitle(title)

    fig.tight_layout()
    plt.savefig(save_path)
