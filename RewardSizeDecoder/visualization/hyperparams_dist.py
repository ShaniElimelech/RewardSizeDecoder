import math
import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(params_lst, save_path, title):
    plt.hist(params_lst, bins=70)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def params_hist(data_path, model, subject_lst, session_lists):
    save_path = os.path.join(data_path, 'figures', f'Decoder {model} output', 'hyperparams_hist')
    os.makedirs(save_path, exist_ok=True)
    all_params = {}
    for i, subject in enumerate(subject_lst):
        session_list = session_lists[i]
        for j, session in enumerate(session_list):
            print(subject, session)
            data = os.path.join(data_path, f'Decoder {model} output' ,f'{subject}', f'session{session}', 'hyperparameters.pklq')
            if not os.path.exists(data):
                continue
            params_dict = pickle.load(open(data, 'rb'))
            for outer_k, inner_dict in params_dict.items():
                for inner_k, inner_v in inner_dict.items():
                    value_array = np.array(inner_v)
                    if any(isinstance(x, str) for x in inner_v):
                        all_params[inner_k] = all_params.get(inner_k, []) + inner_v
                    else:
                        if not np.any(np.isinf(value_array)):
                            all_params[inner_k] = all_params.get(inner_k, []) + inner_v

    for param in all_params:
        title = f'histogram of parameter {param} - all sessions'
        save = os.path.join(save_path, f'{title}.png')
        plot_hist(all_params[param], save, title)


TIME_BINS = [
    (-10, -2),
    (-2, 10),
    (10, 20),
    (20, 50)
]

TIME_BIN_LABELS = {
    (-10, -2): "[-10, -2]",
    (-2, 10): "(-2, 10]",
    (10, 20): "(10, 20]",
    (20, 50): "(20, 50]"
}

# soft "watercolor" palette
TIME_BIN_COLORS = {
    "[-10, -2]": "#8ecae6",   # light blue
    "(-2, 10]": "#ffb703",    # soft orange
    "(10, 20]": "#f4acb7",    # pastel cyan
    "(20, 50]": "#cdb4db"     # lavender
}


def plot_overlaid_hist(data_dict, save_path, title, bins=70):
    """
    data_dict: {label: list_of_values}
    """
    plt.figure(figsize=(8, 6))

    for label, values in data_dict.items():
        if len(values) == 0:
            continue

        plt.hist(
            values,
            bins=bins,
            alpha=0.55,
            label=label,
            color=TIME_BIN_COLORS.get(label, None),
            density=True  # helps comparison
        )

    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("Density")
    plt.legend(title="time bin [s]", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_time_bin_label(time_point):
    for tmin, tmax in TIME_BINS:
        if tmin == -10:
            if -10 <= time_point <= -2:
                return TIME_BIN_LABELS[(tmin, tmax)]
        else:
            if tmin < time_point <= tmax:
                return TIME_BIN_LABELS[(tmin, tmax)]
    return None


def params_hist_new(data_path, model, subject_lst, session_lists):
    """
    Creates:
    1) Global histograms (all sessions, all times, all folds)
    2) Per-session histograms
    3) Per-session overlaid histograms split by time bins
    """

    base_save_path = os.path.join(
        data_path,
        "figures",
        f"Decoder {model} output",
        "hyperparams_hist"
    )
    os.makedirs(base_save_path, exist_ok=True)

    # global aggregation
    global_params = {}   # param -> list of values

    # per-session + time-bin aggregation
    session_params = {}  # (subject, session) -> param -> time_bin -> list

    for i, subject in enumerate(subject_lst):
        for session in session_lists[i]:
            print(f"Processing {subject}, session {session}")

            pkl_path = os.path.join(
                data_path,
                f"Decoder {model} output",
                f'{subject}',
                f"session{session}",
                "hyperparameters.pklq"
            )

            if not os.path.exists(pkl_path):
                continue

            params_dict = pickle.load(open(pkl_path, "rb"))
            session_key = (subject, session)
            session_params.setdefault(session_key, {})

            for time_point, param_dict in params_dict.items():
                bin_label = get_time_bin_label(time_point)
                if bin_label is None:
                    continue

                for param, fold_values in param_dict.items():
                    fold_values = np.asarray(fold_values)

                    if fold_values.dtype.kind in {"U", "S"}:
                        continue
                    if np.any(np.isinf(fold_values)):
                        continue

                    # ---------- global ----------
                    global_params.setdefault(param, []).extend(fold_values.tolist())

                    # ---------- per session / time bin ----------
                    session_params[session_key].setdefault(param, {})
                    session_params[session_key][param].setdefault(bin_label, [])
                    session_params[session_key][param][bin_label].extend(fold_values.tolist())

    # Global histograms
    for param, values in global_params.items():
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=70, density=True, alpha=0.7)
        plt.title(f"Global histogram of {param}")
        plt.xlabel("Parameter value")
        plt.ylabel("Density")
        plt.tight_layout()

        save_path = os.path.join(base_save_path, f"global_{param}.png")
        plt.savefig(save_path)
        plt.close()

# Per-session, time-binned histograms (overlayed)
    for (subject, session), param_dict in session_params.items():
        for param, bin_dict in param_dict.items():
            title = f"{param} – subject{subject}-session {session}"
            save_path = os.path.join(base_save_path, f"{param}_{subject}_session{session}.png")

            plot_overlaid_hist(bin_dict, save_path, title)






subject_lst =  [464724, 464725, 463189, 463190]
session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 2, 3, 4, 9], [2, 3, 5, 6, 10]]
model = 'LR'
data_path = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/new-hparams search- fps 5 Hz'
params_hist_new(data_path, model, subject_lst, session_lists)



























