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













subject_lst =  [464724, 464725, 463189, 463190]
session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 2, 3, 4, 9], [2, 3, 5, 6, 10]]
model = 'SVM'
data_path = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/new-hparams search- fps 5 Hz'
params_hist(data_path, model, subject_lst, session_lists)



























