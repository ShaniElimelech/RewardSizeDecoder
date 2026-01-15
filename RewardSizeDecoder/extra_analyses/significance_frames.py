from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from RewardSizeDecoder_class import RewardSizeDecoder


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


def extract_auc_folds(score_dic):
    """
    Returns:
        times: shape (n_times,)
        auc_folds: shape (n_times, n_folds)
    """
    times = np.array(sorted(score_dic.keys()))
    auc_folds = np.array([
        score_dic[t]['roc_auc_folds']
        for t in times
    ])
    return times, auc_folds


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



from scipy.stats import t as tdist

def paired_pvals_vs_baseline_window(score_dic, baseline_window=(-10, -2)):
    times, auc_folds = extract_auc_folds(score_dic)
    # auc_folds: (n_times, n_folds)

    baseline_mask = (times >= baseline_window[0]) & (times < baseline_window[1])
    if not np.any(baseline_mask):
        raise ValueError("No baseline points in the specified window.")

    # baseline per fold
    baseline_per_fold = auc_folds[baseline_mask].mean(axis=0)  # (n_folds,)

    # paired differences
    diffs = auc_folds - baseline_per_fold[None, :]             # (n_times, n_folds)

    # t-test vs 0 for each time
    mean_diff = diffs.mean(axis=1)
    sem_diff  = diffs.std(axis=1, ddof=1) / np.sqrt(diffs.shape[1])

    tstats = mean_diff / sem_diff
    pvals = 1.0 - tdist.cdf(tstats, df=diffs.shape[1] - 1)  # one-sided: >

    return pvals


def paired_pvals_vs_single_time(score_dic, t0=-5):
    times, auc_folds = extract_auc_folds(score_dic)

    t0_idx = np.argmin(np.abs(times - t0))
    baseline_fold = auc_folds[t0_idx]  # (n_folds,)

    diffs = auc_folds - baseline_fold[None, :]

    mean_diff = diffs.mean(axis=1)
    sem_diff  = diffs.std(axis=1, ddof=1) / np.sqrt(diffs.shape[1])

    tstats = mean_diff / sem_diff
    pvals = 1.0 - tdist.cdf(tstats, df=diffs.shape[1] - 1)

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


def pvals_across_sessions(all_sessions_auc, times, baseline_window=(-10, -2)):
    """
    all_sessions_auc: np.array, shape (n_sessions, n_times)
    times: np.array, shape (n_times,)
    baseline_window: tuple (tmin, tmax)

    Returns:
        pvals: shape (n_times,)
        tstats: shape (n_times,)
        diffs: baseline-corrected data, shape (n_sessions, n_times)
    """
    all_sessions_auc = np.asarray(all_sessions_auc)
    if all_sessions_auc.ndim != 2:
        raise ValueError("all_sessions_auc must have shape (n_sessions, n_times)")

    n_sessions, n_times = all_sessions_auc.shape

    # find baseline timepoints
    mask = (times >= baseline_window[0]) & (times < baseline_window[1])
    if not np.any(mask):
        raise ValueError("No baseline points found in the specified window.")

    # baseline per session
    baseline_per_session = all_sessions_auc[:, mask].mean(axis=1)  # (n_sessions,)

    # baseline-correct
    diffs = all_sessions_auc - baseline_per_session[:, None]  # (n_sessions, n_times)

    # one-sample t-test across sessions vs 0 (one-sided)
    mean_diff = diffs.mean(axis=0)
    sem_diff = diffs.std(axis=0, ddof=1) / np.sqrt(n_sessions)

    tstats = mean_diff / sem_diff
    pvals = 1.0 - tdist.cdf(tstats, df=n_sessions - 1)

    return pvals


def plot_auc_with_significance(times, auc, auc_sem, parm_auc, pvals, p_value, title, savepath):
    sig = pvals < p_value
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
    plt.fill_between(times, auc - auc_sem, auc + auc_sem, alpha=0.25, label="±1 std")
    plt.plot(times, ave_parm_auc, color='pink', linewidth=2 ,label="average permutations AUC")

    # plt.scatter(times[sig], ymax * np.ones(np.sum(sig)),
    #             s=10, color='red', label="p-value < 0.05")

    # Add bbox with percentage
    textstr = f"Significant frames:\n{perc_sig:.1f}% with p-value < {p_value}"
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


def plot_auc_with_baseline_pvalue(times, auc, auc_sem, pvals, p_value, title, savepath):
    sig = pvals < p_value
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

    plt.scatter(times[sig], ymax * np.ones(np.sum(sig)), s=10, color='red', label=f"p-value < {p_value}")

    plt.xlabel("Time (s)")
    plt.ylabel("ROC AUC")
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)
    plt.close()



def compute_label_permutation(num_permutations, user_pipeline_params, subject_lst, session_lists, supported_resampling, data_path, sig_save_path):
    results = defaultdict(dict)
    for i, subject in enumerate(subject_lst):
        session_list = session_lists[i]
        for j, session in enumerate(session_list):
            # real decoding
            model_name = user_pipeline_params['model']
            score_dic_path = os.path.join(data_path, f'Decoder {model_name} output', f'{subject}', f'session{session}', 'scores.pkl')
            score_dic = pickle.load(open(score_dic_path, "rb"))
            times, real_auc = extract_auc_timeseries(score_dic)

            n_times = len(times)
            perm_auc = np.zeros((n_times, num_permutations))

            for i_perm in range(num_permutations):
                user_pipeline_params['subject_id'] = subject
                user_pipeline_params['session'] = session
                user_pipeline_params['save_results'] = False
                decoder = RewardSizeDecoder(**user_pipeline_params)

                decoder.validate_params(supported_models={"LR", "SVM", "LDA"},
                                        supported_resampling=supported_resampling)
                decoder.define_saveroot(reference_path=None,    # data file path/ directory to save results, if None results will be save in the parent folder
                                        reference_path_video=None,
                                        log_to_file=False)  # dont save logs to file

                score_dic_perm = decoder.decoder(permute_labels=True)
                _, auc_perm = extract_auc_timeseries(score_dic_perm)
                perm_auc[:, i_perm] = auc_perm

            # -------- SESSION P-VALUES --------
            pvals = permutation_pvalues(real_auc, perm_auc)

            # -------- SAVE SESSION RESULTS --------
            results[subject][session] = {
                "times": times,
                "real_auc": real_auc,
                "perm_auc": perm_auc,
                "pvals": pvals
            }

    # save results
    with open(os.path.join(sig_save_path, 'decoding_permutation_results.pkl'), "wb") as f:
        pickle.dump(results, f)


    # SUBJECT-level aggregation

    subject_results = {}

    for subject in results:
        sessions = list(results[subject].keys())

        real_auc_sessions = []
        perm_auc_sessions = []

        for session in sessions:
            real_auc_sessions.append(results[subject][session]["real_auc"])
            perm_auc_sessions.append(results[subject][session]["perm_auc"])

        real_auc_subject = np.mean(real_auc_sessions, axis=0)
        perm_auc_subject = np.mean(perm_auc_sessions, axis=0)

        pvals_subject = permutation_pvalues(real_auc_subject, perm_auc_subject)

        subject_results[subject] = {
            "times": results[subject][sessions[0]]["times"],
            "real_auc": real_auc_subject,
            "perm_auc": perm_auc_subject,
            "pvals": pvals_subject
        }

    # GROUP-level aggregation
    real_auc_subjects = []
    perm_auc_subjects = []

    for subject in subject_results:
        real_auc_subjects.append(subject_results[subject]["real_auc"])
        perm_auc_subjects.append(subject_results[subject]["perm_auc"])

    real_auc_group = np.mean(real_auc_subjects, axis=0)
    perm_auc_group = np.mean(perm_auc_subjects, axis=0)
    pvals_group = permutation_pvalues(real_auc_group, perm_auc_group)

    group_results = {
        "times": subject_results[subject_lst[0]]["times"],
        "real_auc": real_auc_group,
        "perm_auc": perm_auc_group,
        "pvals": pvals_group
    }

    # Save subject + group results
    with open(os.path.join(sig_save_path, "decoding_subject_results.pkl"), "wb") as f:
        pickle.dump(subject_results, f)

    with open(os.path.join(sig_save_path, "decoding_group_results.pkl"), "wb") as f:
        pickle.dump(group_results, f)



def plot_significant_frames_perm(num_permutations, p_value, user_pipeline_params, subject_lst, session_lists, supported_resampling, data_path, sig_save_path):
    frame_rate = user_pipeline_params['frame_rate']
    model = user_pipeline_params['model']

    # Paths
    results_path = os.path.join(sig_save_path, "decoding_permutation_results.pkl")
    subject_results_path = os.path.join(sig_save_path,"decoding_subject_results.pkl")
    group_results_path = os.path.join(sig_save_path,"decoding_group_results.pkl")

    # check if permutation process is already been done - if the outputs exist load them, if not, call compute permutation
    if not os.path.exists(results_path) and not os.path.exists(subject_results_path) and not os.path.exists(group_results_path):
        compute_label_permutation(num_permutations, user_pipeline_params, subject_lst, session_lists, supported_resampling, data_path, sig_save_path)

    # Load raw session-level results
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Load subject-level results
    with open(subject_results_path, "rb") as f:
        subject_results = pickle.load(f)

    # Load group-level results
    with open(group_results_path, "rb") as f:
        group_results = pickle.load(f)


    # plotting

    os.makedirs(sig_save_path, exist_ok=True)
    # AUC + significant time points
    for subject in results:
        all_sessions_auc = []
        for session in results[subject]:
            times = results[subject][session]["times"]
            time = np.array([i / frame_rate for i in times], dtype=float)
            pvals = results[subject][session]["pvals"]
            perm_auc = results[subject][session]["perm_auc"]

            score_dic_path = f'{data_path}/Decoder {model} output/{subject}/session{session}/scores.pkl'
            score_dic = pickle.load(open(score_dic_path, "rb"))
            real_auc, auc_sem = extract_auc_sem(score_dic)
            all_sessions_auc.append(real_auc)

            plot_auc_with_significance(
                time,
                real_auc,
                auc_sem,
                perm_auc,
                pvals,
                p_value,
                title=f"Subject {subject} – Session {session}",
                savepath=os.path.join(sig_save_path, f"Subject {subject} – Session {session}_perm.png")
            )

        subject_sem = np.std(np.array(all_sessions_auc), axis=0) / np.sqrt(len(all_sessions_auc))
        subject_results[subject]["subject_sem"] = subject_sem

    # SUBJECT-LEVEL PLOTTING

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
            auc_sem,
            perm_auc,
            pvals,
            p_value,
            title=f"Subject {subject} – Average over sessions",
            savepath = os.path.join(sig_save_path, f"Subject_{subject}_Average_over_sessions_perm.png")
        )

    group_sem = np.std(np.array(all_subjects_auc), axis=0) / np.sqrt(len(all_subjects_auc))
    # GROUP-LEVEL PLOTTING
    group_pvals = np.expand_dims(group_results["pvals"], axis=0)

    plot_auc_with_significance(
        time,
        group_results["real_auc"],
        group_sem,
        group_results["perm_auc"],
        group_results["pvals"],
        p_value,
        title="All subjects decoding (average over subjects)",
        savepath=os.path.join(sig_save_path, f"All_subjects_pvalue_permutation.png")
    )


def plot_significant_frames_baseline(subject_lst,session_lists, p_value, user_pipeline_params, data_path, sig_save_path):
    frame_rate = user_pipeline_params['frame_rate']
    model = user_pipeline_params['model']
    subject_results = defaultdict(dict)

    # plotting

    os.makedirs(sig_save_path, exist_ok=True)
    # AUC + significant time points
    for i, subject in enumerate(subject_lst):
        session_list = session_lists[i]
        all_sessions_auc = []
        for j, session in enumerate(session_list):
            score_dic_path = f'{data_path}/Decoder {model} output/{subject}/session{session}/scores.pkl'
            score_dic = pickle.load(open(score_dic_path, "rb"))
            real_auc, auc_sem = extract_auc_sem(score_dic)
            times = np.array(sorted(score_dic.keys()))
            time = np.array([i / frame_rate for i in times], dtype=float)
            all_sessions_auc.append(real_auc)
            pvals_ttest = paired_pvals_vs_baseline_window(score_dic)

            plot_auc_with_baseline_pvalue(
                time,
                real_auc,
                auc_sem,
                pvals_ttest,
                p_value,
                title=f"Subject {subject} – Session {session}",
                savepath=os.path.join(sig_save_path, f"Subject {subject} – Session {session}_pvalue_baseline_window.png")
            )

        real_auc = np.mean(np.array(all_sessions_auc), axis=0)
        subject_sem = np.std(np.array(all_sessions_auc), axis=0) / np.sqrt(len(all_sessions_auc))
        subject_results[subject]["real_auc"] = real_auc
        subject_results[subject]["subject_sem"] = subject_sem
        subject_results[subject]["all_sessions_auc"] = all_sessions_auc


    # SUBJECT-LEVEL PLOTTING

    all_subjects_auc = []
    for subject in subject_lst:
        real_auc = subject_results[subject]["real_auc"]
        auc_sem = subject_results[subject]["subject_sem"]
        all_sessions_auc = subject_results[subject]["all_sessions_auc"]
        all_subjects_auc.append(real_auc)
        pvals_ttest = pvals_across_sessions(all_sessions_auc, times)

        plot_auc_with_baseline_pvalue(
            time,
            real_auc,
            auc_sem,
            pvals_ttest,
            p_value,
            title=f"Subject {subject} – Average over sessions",
            savepath=os.path.join(sig_save_path, f"Subject {subject}_pvalue_baseline_window.png")
        )

    group_auc = np.mean(np.array(all_subjects_auc), axis=0)
    group_sem = np.std(np.array(all_subjects_auc), axis=0) / np.sqrt(len(all_subjects_auc))
    pvals_ttest = pvals_across_sessions(all_subjects_auc, times)

    # GROUP-LEVEL PLOTTING
    plot_auc_with_baseline_pvalue(
        time,
        group_auc,
        group_sem,
        pvals_ttest,
        p_value,
        title=f"All subjects decoding (average over subjects)",
        savepath=os.path.join(sig_save_path, f"All_subjects_pvalue_baseline_window.png")
    )

