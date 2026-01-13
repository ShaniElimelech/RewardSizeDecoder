from numpy.ma.core import shape

from main import RewardSizeDecoder
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.patches import Patch
from scipy.stats import t as tdist





supported_resampling = ['No resample', 'combine undersample(random) and oversample(SMOTE)', 'simple undersample', 'undersample and ensemble']
supported_models = ['LDA', 'SVM', 'LR']
user_model_params = {'LDA': {}, 'SVM': {'probability': True}, 'LR': {'thresh':0.65}}
host = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
user = 'ShaniE'
password = 'opala'
dj_info = {'host_path': host, 'user_name': user, 'password': password}
subject_lst = [464724, 464725, 463189, 463190]  #[464724, 464725, 463189, 463190]
session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 9], [2, 3, 5, 6, 10]]    #  [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]
all_sessions = {}
frame_rate = 10
model = 'LR'
num_permutations = 100



def extract_auc_timeseries(score_dic):
    """
    Returns:
        times (np.array)
        auc (np.array)
    """
    times = np.array(sorted(score_dic.keys()))
    auc = np.array([
        np.mean(score_dic[t]['roc_auc_folds'])
        for t in times
    ])
    return times, auc


def extract_auc_sem(score_dic):
    """
     Returns:
         auc (np.array)
         sem (np.array)
     """
    times = np.array(sorted(score_dic.keys()))
    auc = np.array([
        np.mean(score_dic[t]['roc_auc_folds'])
        for t in times
    ])

    auc_sem = np.array([
        np.std(score_dic[t]['roc_auc_folds']) / np.sqrt(len(score_dic[t]['roc_auc_folds']))
        for t in times
    ])

    return auc, auc_sem


def collect_pvals_across_sessions(results, subject):
    return np.array([
        results[subject][session]["pvals"]
        for session in results[subject]
    ])

def permutation_pvalues(real_auc, perm_auc):
    """
    real_auc: shape (n_times,)
    perm_auc: shape (n_times, n_perm)
    """
    n_perm = perm_auc.shape[1]
    pvals = np.zeros(real_auc.shape[0])

    for t in range(len(real_auc)):
        pvals[t] = (np.sum(perm_auc[t] >= real_auc[t]) + 1) / (n_perm + 1)

    return pvals



def permutation_pvals_ttest(real_auc, perm_auc):
    """
    One-sided t-test p-values testing:
        H0: mean(perm_auc[t]) = real_auc[t]
        H1: real_auc[t] > mean(perm_auc[t])

    real_auc: shape (n_times,)
    perm_auc: shape (n_times, n_perm)

    Returns:
        pvals: shape (n_times,)
        tstats: shape (n_times,)
    """

    n_times, n_perm = perm_auc.shape

    mu = perm_auc.mean(axis=1)                 # mean of permutation null
    sd = perm_auc.std(axis=1, ddof=1)          # std of permutation null
    sem = sd / np.sqrt(n_perm)                 # standard error of the mean

    # t-statistic: (real - null_mean) / SEM
    tstats = (real_auc - mu) / sem

    # one-sided p-value: P(T >= t)
    pvals = 1.0 - tdist.cdf(tstats, df=n_perm - 1)

    return pvals


def baseline_pvals_ttest(real_auc, auc_sem, times, n):
    """
    One-sided t-test p-values testing:
        H0: mean(real_auc[t]) = baseline
        H1: real_auc[t] > baseline

    real_auc: shape (n_times,)
    auc_sem: shape (n_times,)

    Returns:
        pvals: shape (n_times,)
        tstats: shape (n_times,)
    """

    mask = (times >= -10) & (times < -2)
    baseline = np.mean(real_auc[mask])

    # t-statistic: (real - null_mean) / SEM
    tstats = (real_auc - baseline) / auc_sem

    # one-sided p-value: P(T >= t)
    pvals = 1.0 - tdist.cdf(tstats, df=n - 1)

    return pvals



def aggregate_subject(sessions_real_auc, sessions_perm_auc):
    """
    sessions_real_auc: list of (n_times,)
    sessions_perm_auc: list of (n_times, n_perm)
    """
    real_auc_subject = np.mean(sessions_real_auc, axis=0)
    perm_auc_subject = np.mean(sessions_perm_auc, axis=0)

    return real_auc_subject, perm_auc_subject

def aggregate_group(subject_real_auc, subject_perm_auc):
    real_auc_group = np.mean(subject_real_auc, axis=0)
    perm_auc_group = np.mean(subject_perm_auc, axis=0)
    return real_auc_group, perm_auc_group


def plot_pvalue_distribution(times, pvals_all, title, savepath):
    """
    pvals_all: shape (n_times)
    """
    pvals_all = np.asarray(pvals_all).squeeze()
    times = np.asarray(times).squeeze()
    mean_p = float(np.mean(pvals_all))

    plt.figure(figsize=(10, 4))
    plt.plot(times, pvals_all)
    plt.axhline(0.05, color='r', linestyle='--',label='threshold p-value: 0.05')
    plt.axhline(mean_p, color='b', linestyle='--', label=f'mean p-value: {round(mean_p,2)}')
    plt.xlabel("Time (s)")
    plt.ylabel("p-value")
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)
    plt.close()


def plot_auc_with_significance(times, auc, parm_auc, pvals, title, savepath):
    sig = pvals < 0.05
    ymax = auc.max() + 0.02
    ave_parm_auc = parm_auc.mean(axis=1)

    mask = (times >= -10) & (times < -2)
    avg_val = np.mean(auc[mask])
    perc_sig = int(round(100 * np.mean(sig)))
    plt.figure(figsize=(10, 4))
    # Vertical line at time = 0
    plt.axvline(x=0, linestyle="--", linewidth=1, label="t = 0")
    plt.axhline(avg_val, linestyle="--", color='black', linewidth=2, label=f"Avg baseline (-10 to -2s)")
    plt.plot(times, auc, label="AUC")
    plt.plot(times, ave_parm_auc, color='pink', linewidth=2 ,label="average permutations AUC")

    # plt.scatter(times[sig], ymax * np.ones(np.sum(sig)),
    #             s=10, color='red', label="p-value < 0.05")

    # Add bbox with percentage
    textstr = f"Significant frames:\n{perc_sig:.1f}%"
    plt.text(
        0.5, 0.95, textstr,
        transform=plt.gca().transAxes,  # axes coordinates (0–1)
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.xlabel("Time (s)")
    plt.ylabel("ROC AUC")
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)
    plt.close()


def plot_auc_with_baseline_pvalue(times, auc, auc_sem, pvals, title, savepath):
    sig = pvals < 0.05
    ymax = auc.max() + 0.02

    mask = (times >= -10) & (times < -2)
    avg_val = np.mean(auc[mask])
    perc_sig = int(round(100 * np.mean(sig)))
    plt.figure(figsize=(10, 4))
    # Vertical line at time = 0
    plt.axvline(x=0, linestyle="--", linewidth=1, label="t = 0")
    plt.axhline(avg_val, linestyle="--", color='black', linewidth=2, label=f"Avg baseline (-10 to -2s)")
    plt.plot(times, auc, label="AUC")
    plt.fill_between(times, auc - auc_sem, auc + auc_sem, alpha=0.25, label="±1 std")

    plt.scatter(times[sig], ymax * np.ones(np.sum(sig)), s=10, color='red', label="p-value < 0.05")

    # Add bbox with percentage
    # textstr = f"Significant frames:\n{perc_sig:.1f}%"
    # plt.text(
    #     0.5, 0.95, textstr,
    #     transform=plt.gca().transAxes,  # axes coordinates (0–1)
    #     fontsize=9,
    #     verticalalignment='top',
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    # )

    plt.xlabel("Time (s)")
    plt.ylabel("ROC AUC")
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)
    plt.close()




# results = defaultdict(dict)
# for i, subject in enumerate(subject_lst):
#     session_list = session_lists[i]
#     for j, session in enumerate(session_list):
#         print("Processing subject %d/%d" % (subject, session))
#         # real decoding
#         score_dic_path = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/prediction - fps 10 Hz/Decoder {model} output/{subject}/session{session}/scores.pkl'
#         score_dic = pickle.load(open(score_dic_path, "rb"))
#         times, real_auc = extract_auc_timeseries(score_dic)
#
#         n_times = len(times)
#         perm_auc = np.zeros((n_times, num_permutations))
#
#         for i_perm in range(num_permutations):
#             decoder = RewardSizeDecoder(
#                 subject_id=subject,                                 # subject id
#                 session=session,                                    # session number
#                 num_features=200,                                     # number of predictive features from video
#                 frame_rate=10,                                         # neural frame rate(Hz)
#                 time_bin=(-10, 50),                                 # trial bin duration(sec)
#                 original_video_path='D:/Arseny_behavior_video',     # path to raw original video data
#                 model="LR",                                         # type of classification model to apply on data - supported_models = ['LDA', 'SVM', 'LR']
#                 user_model_params=user_model_params,                # model hyperparameters, if not specify then the default will be set/ apply parameters search
#                 resample_method='simple undersample',               # choose resample method to handle unbalanced data
#                 dj_info=dj_info,                                    # data joint user credentials
#                 save_folder_name=f"None",                           # choose new folder name for each time you run the model with different parameters
#                 save_video_folder = f"processed video - fps 10 Hz",  # save processed video outputs (downsampled video, svd)
#                 handle_omission='convert',                          # ['keep'(no change), 'clean'(throw omission trials), 'convert'(convert to regular)]
#                 clean_ignore=True,                                  # throw out ignore trials (trials in which the mouse was not responsive)
#                 find_parameters=False                               # enable hyperparameters search
#             )
#
#             decoder.validate_params(supported_models={"LR", "SVM", "LDA"}, supported_resampling=supported_resampling)
#             decoder.define_saveroot(reference_path=None, log_to_file=False)
#
#             score_dic_perm = decoder.decoder()
#             _, auc_perm = extract_auc_timeseries(score_dic_perm)
#             perm_auc[:, i_perm] = auc_perm
#
#         # -------- SESSION P-VALUES --------
#         pvals = permutation_pvalues(real_auc, perm_auc)
#
#         # -------- SAVE SESSION RESULTS --------
#         results[subject][session] = {
#             "times": times,
#             "real_auc": real_auc,
#             "perm_auc": perm_auc,
#             "pvals": pvals
#         }
#
# # save results
# save_path = 'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/label shuffle 10 hz'
# os.makedirs(save_path, exist_ok=True)
# with open(os.path.join(save_path, 'decoding_permutation_results.pkl'), "wb") as f:
#     pickle.dump(results, f)
#
#
# # SUBJECT-level aggregation
#
# subject_results = {}
#
# for subject in results:
#     sessions = list(results[subject].keys())
#
#     real_auc_sessions = []
#     perm_auc_sessions = []
#
#     for session in sessions:
#         real_auc_sessions.append(results[subject][session]["real_auc"])
#         perm_auc_sessions.append(results[subject][session]["perm_auc"])
#
#     real_auc_subject = np.mean(real_auc_sessions, axis=0)
#     perm_auc_subject = np.mean(perm_auc_sessions, axis=0)
#
#     pvals_subject = permutation_pvalues(real_auc_subject, perm_auc_subject)
#
#     subject_results[subject] = {
#         "times": results[subject][sessions[0]]["times"],
#         "real_auc": real_auc_subject,
#         "perm_auc": perm_auc_subject,
#         "pvals": pvals_subject
#     }
#
#
#
# # GROUP-level aggregation
# real_auc_subjects = []
# perm_auc_subjects = []
#
# for subject in subject_results:
#     real_auc_subjects.append(subject_results[subject]["real_auc"])
#     perm_auc_subjects.append(subject_results[subject]["perm_auc"])
#
# real_auc_group = np.mean(real_auc_subjects, axis=0)
# perm_auc_group = np.mean(perm_auc_subjects, axis=0)
# pvals_group = permutation_pvalues(real_auc_group, perm_auc_group)
#
# group_results = {
#     "times": subject_results[subject_lst[0]]["times"],
#     "real_auc": real_auc_group,
#     "perm_auc": perm_auc_group,
#     "pvals": pvals_group
# }
#
# # Save subject + group results
# with open(os.path.join(save_path, "decoding_subject_results.pkl"), "wb") as f:
#     pickle.dump(subject_results, f)
#
# with open(os.path.join(save_path, "decoding_group_results.pkl"), "wb") as f:
#     pickle.dump(group_results, f)
#
#

# Paths
base_path = "C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/label shuffle 10 hz"

results_path = os.path.join(base_path, "decoding_permutation_results.pkl")
subject_results_path = os.path.join(base_path,"decoding_subject_results.pkl")
group_results_path = os.path.join(base_path,"decoding_group_results.pkl")

# Load raw session-level results
with open(results_path, "rb") as f:
    results = pickle.load(f)

# Load subject-level results
with open(subject_results_path, "rb") as f:
    subject_results = pickle.load(f)

# Load group-level results
with open(group_results_path, "rb") as f:
    group_results = pickle.load(f)

# Plot 1 — p-value distribution

save_path = 'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/label shuffle 10 hz'
os.makedirs(save_path, exist_ok=True)
# AUC + significant time points
for subject in results:
    all_sessions_auc = []
    for session in results[subject]:

        times = results[subject][session]["times"]
        time = np.array([i / frame_rate for i in times], dtype=float)
        # real_auc = results[subject][session]["real_auc"]
        pvals = results[subject][session]["pvals"]
        perm_auc = results[subject][session]["perm_auc"]

        score_dic_path = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/prediction - fps 10 Hz/Decoder {model} output/{subject}/session{session}/scores.pkl'
        score_dic = pickle.load(open(score_dic_path, "rb"))
        real_auc, auc_sem = extract_auc_sem(score_dic)
        all_sessions_auc.append(real_auc)
        # pvals_ttest =

        plot_auc_with_significance(
            time,
            real_auc,
            perm_auc,
            pvals,
            title=f"Subject {subject} – Session {session}",
            savepath=os.path.join(save_path, f"Subject {subject} – Session {session}.png")
        )

        plot_auc_with_baseline_pvalue(
            time,
            real_auc,
            auc_sem,
            pvals,
            title=f"Subject {subject} – Session {session}",
            savepath=os.path.join(save_path, f"Subject {subject} – Session {session}_baseline_pvalue.png")
        )


        plot_pvalue_distribution(
            time,
            pvals,
            title=f"Subject {subject} – Session {session} p-value distribution",
            savepath=os.path.join(save_path, f"Subject {subject} – Session {session}_pvalue_distribution.png")
        )

    subject_sem = np.std(np.array(all_sessions_auc), axis=0) / np.sqrt(len(all_sessions_auc))
    subject_results[subject]["subject_sem"] = subject_sem

# SUBJECT-LEVEL PLOTTING

for subject in subject_results:
    subject_pvals = subject_results[subject]["pvals"]
    times = subject_results[subject]["times"]
    time = np.array([i / frame_rate for i in times], dtype=float)

    plot_pvalue_distribution(
        time,
        subject_pvals,
        title=f"Subject {subject} p-value distribution (averaged over sessions)",
        savepath = os.path.join(save_path, f"Subject_{subject}_pvalue_distribution.png")
    )

all_subjects_auc = []
for subject in subject_results:
    times = subject_results[subject]["times"]
    time = np.array([i / frame_rate for i in times], dtype=float)
    real_auc = subject_results[subject]["real_auc"]
    auc_sem = subject_results[subject]["subject_sem"]
    pvals = subject_results[subject]["pvals"]
    perm_auc = subject_results[subject]["perm_auc"]
    all_subjects_auc.append(real_auc)

    plot_auc_with_significance(
        time,
        real_auc,
        perm_auc,
        pvals,
        title=f"Subject {subject} – Average over sessions",
        savepath = os.path.join(save_path, f"Subject_{subject}_Average_over_sessions.png")
    )

    plot_auc_with_baseline_pvalue(
        time,
        real_auc,
        auc_sem,
        pvals,
        title=f"Subject {subject} – Average over sessions",
        savepath=os.path.join(save_path, f"Subject {subject}_baseline_pvalue.png")
    )

group_sem = np.std(np.array(all_subjects_auc), axis=0) / np.sqrt(len(all_subjects_auc))
# GROUP-LEVEL PLOTTING
group_pvals = np.expand_dims(group_results["pvals"], axis=0)

plot_pvalue_distribution(
    time,
    group_pvals,
    title="All subjects p-value distribution",
    savepath = os.path.join(save_path, f"All_subjects_pvalue_distribution.png")
)

plot_auc_with_significance(
    time,
    group_results["real_auc"],
    group_results["perm_auc"],
    group_results["pvals"],
    title="All subjects decoding (average over subjects)",
    savepath=os.path.join(save_path, f"All_subjects_decoding_average_over_subjects.png")
)

plot_auc_with_baseline_pvalue(
    time,
    group_results["real_auc"],
    group_sem,
    pvals,
    title=f"All subjects decoding (average over subjects)",
    savepath=os.path.join(save_path, f"All subjects decoding (average over subjects).png")
)

