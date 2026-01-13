from main import RewardSizeDecoder
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.patches import Patch






supported_resampling = ['No resample', 'combine undersample(random) and oversample(SMOTE)', 'simple undersample', 'undersample and ensemble']
supported_models = ['LDA', 'SVM', 'LR']
user_model_params = {'LDA': {}, 'SVM': {'probability': True}, 'LR': {'thresh':0.65}}
host = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
user = 'ShaniE'
password = 'opala'
dj_info = {'host_path': host, 'user_name': user, 'password': password}
subject_lst = [464724, 464725, 463189, 463190]  #[464724, 464725, 463189, 463190]
session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]    #  [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]
all_sessions = {}
frame_rate = 5
time_bin = (0, 5)
time_window = np.arange(time_bin[0] * frame_rate, time_bin[1] * frame_rate + 1)
pcs_remove = range(500)
pc_dict = {}
for pc_remove in pcs_remove:
    subjects_score_lst = []
    for i, subject in enumerate(subject_lst):
        sessions_score_lst = []
        session_list = session_lists[i]
        for j, session in enumerate(session_list):
            decoder = RewardSizeDecoder(
                pc_remove=pc_remove,
                subject_id=subject,                                 # subject id
                session=session,                                    # session number
                num_features=500,      # [1,2,5,10,20,50,100,200,300,400,500]                             # number of predictive features from video
                frame_rate=5,                                  # neural frame rate(Hz)
                time_bin=(-1, 8),                                 # trial bin duration(sec)
                original_video_path='D:/Arseny_behavior_video',     # path to raw original video data
                model="LR",                                         # type of classification model to apply on data - supported_models = ['LDA', 'SVM', 'LR']
                user_model_params=user_model_params,                # model hyperparameters, if not specify then the default will be set/ apply parameters search
                resample_method='simple undersample',               # choose resample method to handle unbalanced data
                dj_info=dj_info,                                    # data joint user credentials
                save_folder_name=f"None",      # choose new folder name for each time you run the model with different parameters
                save_video_folder = f"processed video - fps 5 Hz",  # save processed video outputs (downsampled video, svd)
                handle_omission='convert',                          # ['keep'(no change), 'clean'(throw omission trials), 'convert'(convert to regular)]
                clean_ignore=True,                                  # throw out ignore trials (trials in which the mouse was not responsive)
                find_parameters=False                               # enable hyperparameters search
            )

            decoder.validate_params(supported_models={"LR", "SVM", "LDA"}, supported_resampling=supported_resampling)
            decoder.define_saveroot(reference_path=None,            # data file path/ directory to save results, if None results will be save in the parent folder
                                    log_to_file=False)              # dont save logs to file
            # decoder.save_user_parameters(fmt="excel")


            score_dic = decoder.decoder()
            time_score_lst = []
            for time_point, inner_dict in score_dic.items():
                if time_point in time_window:
                    scores = np.array(inner_dict['roc_auc_folds'])
                    mean_score = np.mean(scores)
                    time_score_lst.append(mean_score)

            ave_window_score = np.mean(time_score_lst)
            sessions_score_lst.append(ave_window_score)

        ave_score_sesh = np.mean(np.array(sessions_score_lst))
        subjects_score_lst.append(ave_score_sesh)

    ave_score_subject = np.mean(np.array(subjects_score_lst))
    pc_dict[pc_remove] = ave_score_subject


fpath = "C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/num_pc_analysis/leave_one_pc_out.pkl"
with open(fpath, "wb") as f:
    pickle.dump(pc_dict, f)

# pc_dict = pickle.load(open(fpath, "rb"))

#############
# compute baseline
#############
metric='scores'
model = 'LR'
datapath = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results'
subjects_score_lst = []
for i, subject in enumerate(subject_lst):
    sessions_score_lst = []
    session_list = session_lists[i]
    for j, session in enumerate(session_list):
        data_path = os.path.join(
                            datapath,
                            f"LR-thresh=0.65 - fps 5 Hz",
                            f"Decoder {model} output",
                            f"{subject}",
                            f"session{session}",
                            f"{metric}.pkl"
                        )

        with open(data_path, "rb") as f:
            params_dict = pickle.load(f)

        time_score_lst = []
        for time_point, inner_dict in params_dict.items():
            if time_point in time_window:
                scores = np.array(inner_dict['roc_auc_folds'])
                mean_score = np.mean(scores)
                time_score_lst.append(mean_score)

        ave_window_score = np.mean(time_score_lst)
        sessions_score_lst.append(ave_window_score)

        ave_score_sesh = np.mean(np.array(sessions_score_lst))
        subjects_score_lst.append(ave_score_sesh)

ave_score_baseline = np.mean(np.array(subjects_score_lst))


pcs = sorted(pc_dict.keys())
scores = np.array([pc_dict[pc] for pc in pcs])
deltas = scores - ave_score_baseline

# Color logic
colors = ['green' if d < 0 else 'red' for d in deltas]

plt.figure(figsize=(15, 5))

# Bars start at baseline and extend up/down
plt.bar(
    pcs,
    deltas,
    bottom=ave_score_baseline,
    color=colors,
    edgecolor='black'
)

# Baseline
plt.axhline(
    float(ave_score_baseline),
    color='black',
    linestyle='--',
    linewidth=5,
    zorder=10,   # <-- important
    label='Baseline'
)
plt.xlabel("PC removed")
# plt.xticks(pcs, pcs)
plt.ylabel("Decoder performance (AUC)")
plt.title(f"Effect of Removing Each PC on Decoder Performance\nAUC averaged across subjects and over time bin {time_bin} s")

legend_elements = [
    Patch(facecolor='green', label='Important (performance ↓)'),
    Patch(facecolor='red', label='Not important (performance ↑)'),
]
plt.legend(handles=legend_elements, loc='best')

plt.tight_layout()
# plt.show()
plt.savefig("C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/num_pc_analysis/leave_one_pc_out.png")















