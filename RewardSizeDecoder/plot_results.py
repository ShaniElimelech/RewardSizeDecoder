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


def plot_mean_with_band(time_array, mean_vals, err_vals, title, savepath, ylabel="Score", err_label="±1 std"):
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, mean_vals, label="mean")
    plt.fill_between(time_array, mean_vals - err_vals, mean_vals + err_vals, alpha=0.25, label=err_label)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_score_over_time(yvals, time_array, title, savepath, ylabel="Score"):
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, yvals)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_all_scores_over_time(savedir, scores_dic, roc_dic, subject_id, session, model, frame_bin, frame_rate,
                              error_type="sem", also_plot_per_fold=False):
    """
    error_type: "sem" (standard error of the mean)
    also_plot_per_fold: if True, plots each fold as a thin line (separate files, same directory)
    """
    savedir = os.path.join(savedir, 'figures', f'Decoder {model} output', 'all_scores_over_time')
    os.makedirs(savedir, exist_ok=True)

    # time axis
    time_array = np.array([i / frame_rate for i in frame_bin], dtype=float)

    # Build scores_df: index=time bins, columns=metrics; cell=list of per-fold values
    scores_df = pd.DataFrame.from_dict(scores_dic, orient='index')

    # add scalar AUC per time bin (no folds)
    auc_series = pd.Series({k: v['auc'] for k, v in roc_dic.items()})
    # align to scores_df index order
    scores_df['auc'] = auc_series.reindex(scores_df.index).values

    # sanity check: same length
    assert len(scores_df) == len(time_array), "time_array and number of time bins must match."

    # Iterate metrics (columns)
    for col in scores_df.columns:
        col_series = scores_df[col]
        base_name = f'{col} score over time,subject{subject_id},session{session}'
        savepath = os.path.join(savedir, base_name, '.png')

        if col == 'auc':
            # scalar per time bin → simple line plot
            yvals = col_series.astype(float).values
            plot_score_over_time(yvals, time_array, base_name, savepath, ylabel="Score")

        else:
            mat = _stack_fold_lists(scores_df[col])  # (n_time, n_folds)

            # mean across folds (ignore NaN if padding happened)
            mean_vals = np.nanmean(mat, axis=1)

            # sem = std / sqrt(n_effective)
            std_vals = np.nanstd(mat, axis=1, ddof=1)
            n_eff = np.sum(~np.isnan(mat), axis=1).clip(min=1)
            err_vals = std_vals / np.sqrt(n_eff)
            err_label = "±1 SEM"

            plot_mean_with_band(time_array, mean_vals, err_vals, base_name, savepath, ylabel="Score", err_label=err_label)

            # optional: per-fold lines (can be helpful for debugging/QA)
            if also_plot_per_fold:
                plt.figure(figsize=(10, 6))
                # plot each fold as a thin line
                for j in range(mat.shape[1]):
                    plt.plot(time_array, mat[:, j], alpha=0.35, linewidth=1)
                # overlay mean
                plt.plot(time_array, mean_vals, linewidth=2)
                plt.title(base_name + " (per-fold)")
                plt.xlabel("Time [s]")
                plt.ylabel("Score")
                plt.tight_layout()
                pf_path = os.path.join(savedir, base_name + ' (per-fold).png')
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


def plot_results(savedir, scores_dic, roc_dic, subject_id, session, model, frame_bin, frame_rate):
    """
    Generate all figures (scores over time + ROC curves) and save them under `savedir`.

    Parameters
    ----------
    savedir : str
        Base directory to save figures.
    scores_dic : dict
        Nested dict keyed by frame (or time bin) with scalar metrics per key (e.g., accuracy, precision, recall, etc.).
    roc_dic : dict
        Nested dict keyed by frame (or time bin), each holding {'fpr': array, 'tpr': array, 'auc': float}.
    subject_id : str or int
    session : str or int
    model : str
    frame_bin : list[int]
        Frame indices (or bins) in the same order you want plotted over time.
    frame_rate : float
        Frames per second; used to convert frames to time (seconds).
    """
    # 1) Plot all scalar scores over time (this function also injects AUC)
    plot_all_scores_over_time(
        savedir=savedir,
        scores_dic=scores_dic,
        roc_dic=roc_dic,
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

