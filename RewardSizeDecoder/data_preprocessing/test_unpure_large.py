from collections import defaultdict
from matplotlib import cm
from RewardSizeDecoder.RewardSizeDecoder.data_preprocessing.trial_alignment import get_reward_size_labels
import matplotlib.pyplot as plt
import datajoint as dj
import numpy as np
import os


 # Connect to Datajoint
dj.config['database.host'] = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
dj.config['database.user'] = 'ShaniE'
dj.config['database.password'] = 'opala'
conn = dj.conn()
img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')
video_neural = dj.VirtualModule('VIDEONEURAL', "lab_videoNeuralAlignment")
exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')
dj_modules = {'img': img, 'tracking': tracking, 'exp2': exp2, 'video_neural': video_neural}
save_path_4pad = 'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/2vs4 pad'
os.makedirs(save_path_4pad, exist_ok=True)

subject_lst = [464724, 464725, 463189, 463190]
session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]
steps = [4]
for step in steps:
    plot_sessions = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    subject_colors = {subject_id: colors[i] for i, subject_id in enumerate(subject_lst)}
    subject_labels = []
    all_sessions = []
    loc_dict = defaultdict(lambda: defaultdict(list))
    for i, subject in enumerate(subject_lst):
        session_list = session_lists[i]
        for j, session in enumerate(session_list):
            print(f"subject {subject}, session {session}")
            count_session = 0
            reward_labels = get_reward_size_labels(subject, session, dj_modules, handle_omission='convert',
                                                   clean_ignore=True)
            reward_labels = np.array(reward_labels)
            bad_labels = []
            n_trials = len(reward_labels)
            trial_idx = np.arange(n_trials)
            n_trials_large = np.sum(reward_labels == 'large')
            for trial_num in range(n_trials):
                # Neighbor indices in [trial_num-step, trial_num+step] excluding self
                # if reward_labels[trial_num] == 'large':
                neighbors = np.concatenate([
                    np.arange(trial_num - step, trial_num),
                    np.arange(trial_num + 1, trial_num + step + 1)
                ])

                # Keep only valid neighbor indices
                neighbors = neighbors[(neighbors >= 0) & (neighbors < n_trials)]

                # Any neighbor labeled 'large'
                has_large_neighbor = np.any(reward_labels[neighbors] == 'large')
                if has_large_neighbor:
                    count_session += 1
                    bad_labels.append(trial_num)
            trial_idx = np.delete(trial_idx, bad_labels)
            loc_dict[subject][session].extend(trial_idx)
            np.save(os.path.join(save_path_4pad, f"{subject}_{session}_4pad_trial_idx.npy"), trial_idx)

    #         # unpure_large_per = count_session / n_trials_large
    #         trials_remain = n_trials - count_session
    #         all_sessions.append(trials_remain)
    #         plot_sessions.append(f'{subject}-{session}')
    #         subject_labels.append(subject)
    #
    # # Plot 1: Normalized trial counts per session
    # plt.figure(figsize=(12, 6))
    # bar_colors = [subject_colors[s] for s in subject_labels]
    # plt.bar(plot_sessions, all_sessions, color=bar_colors)
    # plt.xticks(rotation=90)
    # plt.xlabel('Subject-Session')
    # plt.ylabel('bad large trials')
    # plt.title(f'bad large trials normalized by number of large trials in session\nwindow of {step} trials each side')
    # plt.tight_layout()
    # plt.show()

    # Prepare plotting
    # fig, ax = plt.subplots(figsize=(18, 16))
    #
    # x_positions = []
    # x_labels = []
    # colors = cm.tab10.colors  # categorical color map
    #
    # x_counter = 0
    #
    # for subj_idx, (subject, sessions) in enumerate(loc_dict.items()):
    #     color = colors[subj_idx % len(colors)]
    #
    #     for session, trials in sessions.items():
    #         # Scatter trial points
    #         x_vals = np.full(len(trials), x_counter)
    #         ax.scatter(x_vals, trials, color=color, s=35, alpha=0.85)
    #
    #         # Vertical line spanning min → max trial
    #         if len(trials) > 1:
    #             ax.vlines(
    #                 x=x_counter,
    #                 ymin=min(trials),
    #                 ymax=max(trials),
    #                 colors=color,
    #                 linewidth=2,
    #                 alpha=0.8
    #             )
    #
    #         x_labels.append(f"{subject}-{session}")
    #         x_positions.append(x_counter)
    #         x_counter += 1
    #
    # # Formatting
    # ax.set_xticks(x_positions)
    # ax.set_xticklabels(x_labels, rotation=90)
    # ax.set_ylabel("Trial ID")
    # ax.set_xlabel("Subject-Session")
    # ax.set_title(f"unpure trials pad {step}\n(dots are trials id along session)")
    #
    # plt.tight_layout()
    # path = 'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results'
    # plt.savefig(f"{path}/4_padded_trials_idx.png")
    # plt.show()
