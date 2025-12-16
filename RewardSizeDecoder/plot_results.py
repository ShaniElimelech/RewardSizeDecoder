import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def _stack_fold_lists(series):
    """
    series: pd.Series where each element is a list/array of per-fold values.
    returns: 2D np.ndarray of shape (n_time, n_folds)
    """
    # Convert each element to np.array and stack rows
    rows = []
    for v in series:
        a = np.asarray(v, dtype=float)
        rows.append(a)

    return np.vstack(rows)  # (n_time, n_folds)


def plot_mean_with_band(time_array, mean_vals, err_vals, title, savepath, ylabel="Score", err_label="±1 std", tick_every=1, fmt="{:.1f}"):
    time_array = np.asarray(time_array)
    x = np.arange(len(time_array))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mean_vals, linestyle="-", label="mean")
    ax.fill_between(x, mean_vals - err_vals, mean_vals + err_vals, alpha=0.25, label=err_label)
    # Vertical line at time = 0
    zero_idx = np.where(time_array == 0)[0][0]
    ax.axvline(x=zero_idx, linestyle="--", linewidth=1, label="t = 0")
    # --- Compute mean and max in the -10 to -2s window ---
    mask = (time_array >= -10) & (time_array < -2)
    if np.any(mask):
        avg_val = np.mean(mean_vals[mask])
        max_val = np.max(mean_vals[mask])
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
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_score_over_time(yvals, time_array, title, savepath, ylabel="Score", tick_every=1, fmt="{:.1f}"):
    time_array = np.asarray(time_array)
    x = np.arange(len(time_array))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, yvals, linestyle="-", label="mean")
    # Vertical line at time = 0
    zero_idx = np.where(time_array == 0)[0][0]
    ax.axvline(x=zero_idx, linestyle="--", linewidth=1, label="t = 0")
    # --- Compute mean and max in the -10 to -2s window ---
    mask = (time_array >= -10) & (time_array < -2)
    if np.any(mask):
        avg_val = np.mean(yvals[mask])
        max_val = np.max(yvals[mask])
        # Add horizontal lines
        ax.axhline(avg_val, linestyle="--", color='black',linewidth=1, label=f"Avg baseline (-10 to -2s)")
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
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_all_scores_over_time(savedir, scores_dic, roc_dic, pr_auc_dic, subject_id, session, model, frame_bin, frame_rate,
                              error_type="sem", also_plot_per_fold=False):
    """
    error_type: "sem" (standard error of the mean)
    also_plot_per_fold: if True, plots each fold as a thin line (separate files, same directory)
    """
    savedir = os.path.join(savedir, f'Decoder {model} output', 'all_scores_over_time')
    os.makedirs(savedir, exist_ok=True)

    # time axis
    time_array = np.array([i / frame_rate for i in frame_bin], dtype=float)

    # Build scores_df: index=time bins, columns=metrics; cell=list of per-fold values
    scores_df = pd.DataFrame.from_dict(scores_dic, orient='index')

    # add scalar AUC per time bin (no folds)
    auc_series = pd.Series({k: v['auc'] for k, v in roc_dic.items()})
    # align to scores_df index order
    scores_df['auc'] = auc_series.reindex(scores_df.index).values

    # add scalar PR auc per time bin to scores_df
    pr_auc_series = pd.Series(auc for auc in pr_auc_dic.values())
    scores_df['pr_auc'] = pr_auc_series.reindex(scores_df.index).values

    # sanity check: same length
    assert len(scores_df) == len(time_array), "time_array and number of time bins must match."

    # Iterate metrics (columns)
    for col in scores_df.columns:
        col_series = scores_df[col]
        title = f'{col} score over time,subject{subject_id},session{session}'
        savepath = os.path.join(savedir, title + '.png')

        if col in ['auc', 'pr_auc']:
            # scalar per time bin → simple line plot
            yvals = col_series.astype(float).values
            plot_score_over_time(yvals, time_array, title, savepath, ylabel=f"{col} score")

        else:
            mat = _stack_fold_lists(scores_df[col])  # (n_time, n_folds)

            # mean across folds (ignore NaN if padding happened)
            mean_vals = np.nanmean(mat, axis=1)

            # sem = std / sqrt(n_effective)
            std_vals = np.nanstd(mat, axis=1, ddof=1)
            n_eff = np.sum(~np.isnan(mat), axis=1).clip(min=1)
            err_vals = std_vals / np.sqrt(n_eff)
            err_label = "±1 SEM"

            plot_mean_with_band(time_array, mean_vals, err_vals, title, savepath, ylabel=f"{col} score", err_label=err_label)

            # optional: per-fold lines (can be helpful for debugging/QA)
            if also_plot_per_fold:
                plt.figure(figsize=(10, 6))
                # plot each fold as a thin line
                for j in range(mat.shape[1]):
                    plt.plot(time_array, mat[:, j], alpha=0.35, linewidth=1)
                # overlay mean
                plt.plot(time_array, mean_vals, linewidth=2)
                plt.title(title + " (per-fold)")
                plt.xlabel("Time [s]")
                plt.ylabel("Score")
                plt.tight_layout()
                pf_path = os.path.join(savedir, title + ' (per-fold).png')
                plt.savefig(pf_path)
                plt.close()


def plot_roc_curve(savedir, roc_dic, subject_id, session, model):
    savedir = os.path.join(savedir, 'figures', f'Decoder {model} output', 'roc_curve')
    os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, f'roc curve,subject{subject_id},session{session}.png')
    plt.figure(figsize=(10, 6))
    for key in roc_dic.keys():
        tpr = roc_dic[key]['tpr']
        fpr = roc_dic[key]['fpr']
        plt.plot(fpr, tpr, label=key)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve all frames,subject{subject_id},session{session}')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.close()


def plot_results(savedir, scores_dic, roc_dic, pr_auc_dic, subject_id, session, model, frame_bin, frame_rate):
    """
    Generate all figures (scores over time + ROC curves) and save them under `savedir`.

    Parameters
    ----------
    savedir : str
        Base directory to save figures.
    scores_dic : dict
        Nested dict keyed by frame (time bin) with scalar metrics per key (e.g., accuracy, precision, recall, etc.).
    roc_dic : dict
        Nested dict keyed by frame (time bin), each holding {'fpr': array, 'tpr': array, 'auc': float}.
    pr_auc_dic : dict
        dict keyed by frame (time bin) such that {frame: pr_auc}
    subject_id : str or int
    session : str or int
    model : str
    frame_bin : list[int]
        Frame indices that are being plotted.
    frame_rate : float
        Frames per second; used to convert frames to time (seconds).
    """
    # 1) Plot all scalar scores over time (this function also injects AUC)
    plot_all_scores_over_time(
        savedir=savedir,
        scores_dic=scores_dic,
        roc_dic=roc_dic,
        pr_auc_dic=pr_auc_dic,
        subject_id=subject_id,
        session=session,
        model=model,
        frame_bin=frame_bin,
        frame_rate=frame_rate,
    )

    # 2) Plot the ROC curves for each frame/bin
    plot_roc_curve(
        savedir=savedir,
        roc_dic=roc_dic,
        subject_id=subject_id,
        session=session,
        model=model,
    )

