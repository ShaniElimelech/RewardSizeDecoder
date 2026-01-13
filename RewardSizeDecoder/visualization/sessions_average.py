import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_score_over_time(ave_session, stem_session, frame_rate, title, save_path, ylabel="Score"):
    time_array = [int(k) for k in ave_session.keys()]
    yvals = list(ave_session.values())
    stems = np.array(list(stem_session.values()))

    # Optional: sort by x to get ascending axis
    time_array, yvals = zip(*sorted(zip(time_array, yvals)))
    time_array = [t/frame_rate for t in time_array]
    yvals = np.array(yvals)

    time_array = np.asarray(time_array)
    x = np.arange(len(time_array))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, yvals, linestyle="-", label="mean")
    ax.fill_between(x, yvals - stems, yvals + stems, alpha=0.25)
    # Vertical line at time = 0
    zero_idx = np.where(time_array == 0)[0][0]
    ax.axvline(x=zero_idx, linestyle="--", linewidth=1, label="t = 0")
    # --- Compute mean and max in the -10 to -2s window ---
    mask = (time_array >= -10) & (time_array < -2)
    if np.any(mask):
        avg_val = np.mean(yvals[mask])
        max_val = np.max(yvals[mask])
        # Add horizontal lines
        ax.axhline(avg_val, linestyle="--", color='black', linewidth=1, label=f"Avg baseline (-10 to -2s)")
        ax.axhline(max_val, linestyle="--", color='red', linewidth=1, label=f"Max baseline (-10 to -2s)")

    # Ticks correspond 1:1 to list entries
    #ax.set_xticks(x[::tick_every])
    #ax.set_xticklabels([fmt.format(t) for t in time_array[::tick_every]], rotation=45, ha="right")
    int_indices = [i for i, t in enumerate(time_array) if float(t).is_integer()]
    int_times = [int(t) for t in time_array[int_indices]]

    # Apply integer ticks
    ax.set_xticks(int_indices)
    ax.set_xticklabels(int_times, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    fig.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)




def scores_ave(datapath, model, subject_lst, session_lists, frame_rate, param, metric):
    savepath = os.path.join(datapath, 'figures', f'Decoder {model} output', 'averages')
    os.makedirs(savepath, exist_ok=True)
    all_animals_all_time = {}
    for i, subject in enumerate(subject_lst):
        session_list = session_lists[i]
        all_sessions_all_time = {}
        for j, session in enumerate(session_list):
            print(subject, session)
            data = os.path.join(datapath, f'Decoder {model} output' ,f'{subject}', f'session{session}', f'{metric}.pkl')
            if not os.path.exists(data):
                continue
            params_dict = pickle.load(open(data, 'rb'))
            if param == 'PR-auc':
                for key, value in params_dict.items():
                    score = round(value, 2)
                    all_sessions_all_time[key] = all_sessions_all_time.setdefault(key, []) + [score]

            else:
                for outer_k, inner_dict in params_dict.items():
                    if param == 'roc_auc':
                        ave_score = round(inner_dict[param],2)
                    else:
                        score_folds = np.array(inner_dict[param])
                        ave_score = round(np.mean(score_folds),2)
                    all_sessions_all_time[outer_k] = all_sessions_all_time.setdefault(outer_k, []) + [ave_score]

        ave_session = {key: round(sum(item)/len(item),2) for key, item in all_sessions_all_time.items()}
        stem_sessions = {key: round(np.std(np.array(item)/ np.sqrt(len(item))),2) for key, item in all_sessions_all_time.items()}
        title = f'average {param} score - subject {subject}'
        save_path = os.path.join(savepath, f'average {param}- subject {subject}.png')
        plot_score_over_time(ave_session, stem_sessions, frame_rate, title, save_path, ylabel=f'{param} score')
        for key, mean_val in ave_session.items():
            all_animals_all_time.setdefault(key, []).append(mean_val)


    ave_animals = {key: round(sum(item)/len(item),2) for key, item in all_animals_all_time.items()}
    stem_animals = {key: round(np.std(np.array(item) / np.sqrt(len(item))), 2) for key, item in all_animals_all_time.items()}
    title = f'average {param} score - all subjects'
    save_path = os.path.join(savepath, f'average {param}-all subjects.png')
    plot_score_over_time(ave_animals, stem_animals, frame_rate, title, save_path, ylabel=f'{param} score')


metrics = ['scores', 'roc', 'pr_auc']
params = ['PR-auc', 'pr_auc_folds', 'roc_auc', 'roc_auc_folds', 'accuracy', 'precision', 'recall', 'f1_score']
subject_lst =  [464724, 464725, 463189, 463190]
session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]  # [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]
model = 'LR'    # ['LR', 'LDA', 'SVM']
frame_rate = 10
data_path = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/prediction - fps 10 Hz'
for param in params:
    if param == 'roc_auc':
        metric = 'roc'
    elif param == 'PR-auc':
        metric = 'pr_auc'
    else:
        metric = 'scores'
    scores_ave(data_path, model, subject_lst, session_lists, frame_rate, param=param, metric=metric)
