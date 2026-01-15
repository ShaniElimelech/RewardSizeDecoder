import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import cm
from pathlib import Path
from RewardSizeDecoder_class import RewardSizeDecoder


def num_pc_analysis(pc_analysis_saveroot, model, pc_lst, time_bin, frame_rate, subject_lst, session_lists, param, metric, log_scale=False):
    os.makedirs(pc_analysis_saveroot, exist_ok=True)

    time_window = np.arange(time_bin[0] * frame_rate, time_bin[1] * frame_rate + 1)
    # subject -> pc -> list of values (sessions scores)
    all_subjects_data = defaultdict(lambda: defaultdict(list))

    for subj_idx, subject in enumerate(subject_lst):
        session_list = session_lists[subj_idx]

        for session in session_list:
            for pc in pc_lst:
                data_path = os.path.join(
                    pc_analysis_saveroot,
                    f"{pc} features",
                    f"Decoder {model} output",
                    f"{subject}",
                    f"session{session}",
                    f"{metric}.pkl"
                )

                # if not os.path.exists(data_path):
                #     continue

                with open(data_path, "rb") as f:
                    params_dict = pickle.load(f)

                time_score_lst = []
                for time_point, inner_dict in params_dict.items():
                    if time_point in time_window:
                        scores = np.array(inner_dict[param])
                        mean_score = np.mean(scores)
                        time_score_lst.append(mean_score)

                ave_window_score = np.mean(time_score_lst)
                all_subjects_data[subject][pc].append(ave_window_score)

    # ==========================
    # PLOTTING PER SUBJECT
    # ==========================
    for subject, pc_dict in all_subjects_data.items():
        plt.figure(figsize=(8, 5))

        session_list = session_lists[subject_lst.index(subject)]
        colors = cm.tab10(np.linspace(0, 1, len(session_list)))

        for pc in pc_lst:
            if pc not in pc_dict:
                continue

            y = pc_dict[pc]  # session values
            x = [pc] * len(y)

            for val, sess, color in zip(y, session_list, colors):
                plt.scatter(x[0], val, color=color, alpha=0.7, label=f"Session {sess}")

            # mean marker
            plt.scatter(pc, np.mean(y), color="black", s=100, marker="D", label="Mean")

        # remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=8)

        plt.title(f"AUC vs number of PCs - subject {subject}\nAUC averaged over time bin {time_bin} s")
        plt.xlabel("Number of PCs")
        plt.ylabel("AUC")
        if log_scale:
            plt.xscale("log")
            plt.xticks(pc_lst, pc_lst)
        plt.tight_layout()
        plt.savefig(os.path.join(pc_analysis_saveroot, f"{subject}_pc_auc.png"))
        plt.close()


    # ==========================
    #  plot for all subjects
    # ==========================
    plt.figure(figsize=(8, 5))
    colors = cm.tab10(np.linspace(0, 1, len(subject_lst)))

    for pc in pc_lst:
        subjects_means = []
        for subject, color in zip(subject_lst, colors):
            if pc not in all_subjects_data[subject]:
                continue

            vals = np.array(all_subjects_data[subject][pc])
            mean_val = np.mean(vals)
            subjects_means.append(mean_val)

            plt.scatter(pc, mean_val, color=color, alpha=0.7, label=subject)

        # grand mean
        plt.scatter(pc, np.mean(subjects_means), color='red', s=100, marker='D')

    # remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8)

    plt.xlabel("Number of PCs")
    if log_scale:
        plt.xscale("log")
        plt.xticks(pc_lst, pc_lst)
    plt.ylabel("AUC")
    plt.title(f"AUC vs number of PCs Across Subjects\nAUC averaged over time bin {time_bin} s")
    plt.tight_layout()
    plt.savefig(os.path.join(pc_analysis_saveroot, "AUC_vs_number_of_PCs_across_subjects.png"))
    plt.close()





def cumulative_pc_auc_analysis(user_pipeline_params, subject_lst, session_lists, supported_resampling):
    try:
        base = Path(__file__).resolve().parent.parent.parent
    except NameError:
        base = Path.cwd().resolve().parent.parent

    pc_analysis_folder_name = 'num_pc_analysis'
    # Build saveroot path
    pc_analysis_saveroot = base / 'results' / user_pipeline_params['save_folder_name'] / pc_analysis_folder_name
    pc_analysis_saveroot.mkdir(parents=True, exist_ok=True)
    pc_lst = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]

    for pc in pc_lst:
        user_pipeline_params['save_folder_name'] = f'{pc} features'

        for i, subject in enumerate(subject_lst):
            session_list = session_lists[i]
            for j, session in enumerate(session_list):
                user_pipeline_params['subject_id'] = subject
                user_pipeline_params['session'] = session
                decoder = RewardSizeDecoder(**user_pipeline_params)

                decoder.validate_params(supported_models={"LR", "SVM", "LDA"}, supported_resampling=supported_resampling)
                decoder.define_saveroot(reference_path=pc_analysis_saveroot,       # data file path/ directory to save results, if None results will be save in the parent folder
                                        reference_path_video=None,
                                        log_to_file=False)  # dont save logs to file

                all_frames_scores = decoder.decoder()

    num_pc_analysis(
        pc_analysis_saveroot,
        user_pipeline_params['model'],
        pc_lst,
        user_pipeline_params['time_bin'],
        user_pipeline_params['frame_rate'],
        subject_lst,
        session_lists,
        param='roc_auc_folds',
        metric='scores',
        log_scale=True
    )


