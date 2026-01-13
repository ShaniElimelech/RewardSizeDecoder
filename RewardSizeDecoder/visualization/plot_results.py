import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
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


def confusion_to_time_percentage_df(all_frames_confusion):
    """
    index   -> time_idx
    columns -> TP, FP, TN, FN
    values  -> percentage per time point (row sums to 1)
    """
    rows = []
    for time_idx, d in all_frames_confusion.items():
        total = sum(d.values())
        if total == 0:
            continue

        rows.append({
            "time_idx": time_idx,
            "TP": d["TP"] / total,
            "FP": d["FP"] / total,
            "TN": d["TN"] / total,
            "FN": d["FN"] / total,
            "n_trials": total
        })

    df = (
        pd.DataFrame(rows)
        .set_index("time_idx")
        .sort_index()
    )
    return df


#### stacked bar plot (integer ticks only)

def plot_stacked_confusion_over_time_seconds(
    confusion_df,
    time_array,
    subject_id,
    session,
    savepath,
    title
):
    """
    df         : output of confusion_to_time_percentage_df
    time_array : array mapping time_idx -> seconds

    Generates time-resolved stacked bar plots showing per–time-point normalized
    classification outcome distributions (TP/FP/TN/FN) and a zoomed-in view of
    false predictions (FP/FN) over real time.
    """

    # x axis in seconds
    x = time_array
    bottom = np.zeros(len(confusion_df))

    colors = {
        "TP": "green",
        "FP": "orange",
        "TN": "blue",
        "FN": "red"
    }

    # bar width based on temporal resolution
    if len(x) > 1:
        width = np.diff(x).mean()
    else:
        width = 1.0

    # full distribution figure - all metrics are display ("TP", "FP", "TN", "FN")
    fig, ax = plt.subplots(figsize=(14, 5))
    for label in ["TP", "FP", "TN", "FN"]:
        ax.bar(
            x,
            confusion_df[label].values,
            bottom=bottom,
            width=width,
            color=colors[label],
            edgecolor="none",
            label=label
        )
        bottom += confusion_df[label].values

    # ---- integer seconds only ----
    int_mask = np.isclose(x, np.round(x))
    ax.set_xticks(x[int_mask])
    ax.set_xticklabels([int(t) for t in x[int_mask]])

    ax.set_ylim(0, 1)
    ax.set_ylabel("Percentage (per time point)")
    ax.set_xlabel("Time [s]")
    ax.set_title(
        f"{title}\n"
        f"number of trials = {int(confusion_df['n_trials'].mean())}"
    )

    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2)
    )

    ax.set_xlim(x.min() - width / 2, x.max() + width / 2)

    fig.tight_layout()
    save_im = os.path.join(savepath, f"{title}_full_metrics_{subject_id}_session{session}.png")
    fig.savefig(save_im, dpi=150)
    plt.close(fig)

    # zoom in to false prediction - only false prediction distribution is display
    den = confusion_df['FP'] + confusion_df['FN']
    valid = den > 0

    confusion_df['FP_zoom'] = 0.0
    confusion_df['FN_zoom'] = 0.0
    confusion_df.loc[valid, 'FP_zoom'] = confusion_df.loc[valid, 'FP'] / den[valid]
    confusion_df.loc[valid, 'FN_zoom'] = confusion_df.loc[valid, 'FN'] / den[valid]

    colors = {
        "FP_zoom": "#FDB462",
        "FN_zoom": "#FB8072"
    }
    bottom = np.zeros(len(confusion_df))
    fig, ax = plt.subplots(figsize=(14, 5))
    for label in ["FP_zoom", "FN_zoom"]:
        ax.bar(
            x,
            confusion_df[label].values,
            bottom=bottom,
            width=width,
            color=colors[label],
            edgecolor="none",
            label='FP' if label=='FP_zoom' else 'FN'
        )
        bottom += confusion_df[label].values

    # ---- integer seconds only ----
    int_mask = np.isclose(x, np.round(x))
    ax.set_xticks(x[int_mask])
    ax.set_xticklabels([int(t) for t in x[int_mask]])

    ax.set_ylim(0, 1)
    ax.set_ylabel("Percentage (per time point)")
    ax.set_xlabel("Time [s]")
    ax.set_title(
        f"{title}\n"
        f"number of trials = {int(confusion_df['n_trials'].mean())}"
    )

    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2)
    )

    ax.set_xlim(x.min() - width / 2, x.max() + width / 2)

    fig.tight_layout()
    save_im = os.path.join(savepath, f"{title}_fase_predictions_{subject_id}_session{session}.png")
    fig.savefig(save_im, dpi=150)
    plt.close(fig)


############ Extract FP / FN trial-level data
def extract_false_predictions(all_frames_confusion_idx, f_rate):
    """
    Returns DataFrame with:
    time_idx | trial_id | error (FP/FN)
    """
    rows = []
    for time_idx, d in all_frames_confusion_idx.items():
        for err in ("FP", "FN"):
            for trial_id in d.get(err, []):
                rows.append({
                    "time_idx": time_idx / float(f_rate),
                    "trial_id": trial_id,
                    "error": err
                })
    return pd.DataFrame(rows)


########### Baseline threshold computation
def compute_baseline_thresholds(yvals, time_array):
    """
    Baseline window: -10 to -2 seconds
    """
    mask = (time_array >= -10) & (time_array < -2)
    avg_baseline = float(np.mean(yvals[mask]))
    max_baseline = float(np.max(yvals[mask]))
    return avg_baseline, max_baseline

########### Filter FP/FN trials above a baseline threshold
def filter_false_predictions_above_threshold(
    df_false,
    scores_over_time,
    time_array,
    threshold,
    time_col="time_idx"  # df_false column holding real seconds
):
    """
    Keeps FP/FN rows whose time bin score > threshold.
    Assumes df_false[time_col] values exist in time_array.
    """
    # map: time_in_seconds -> score
    score_map = dict(zip(time_array, scores_over_time))

    scores_at_events = df_false[time_col].map(score_map)
    return df_false[scores_at_events > threshold]


########## Trial-index FP/FN scatter with baseline bbox
def plot_false_predictions_with_baseline_bbox(
    df,
    baseline_err,
    baseline_type,      # average / max
    title,
    savepath
):
    fig, ax = plt.subplots(figsize=(10,15))

    color_map = {"FP": "#80B1D3", "FN": "#FB8072"}
    # df = df.iloc[:2000]
    for err in ("FP", "FN"):
        reward_size = 'Regular' if err == 'FP' else 'Large'
        sub = df[df["error"] == err]
        ax.scatter(
            sub["time_idx"],
            sub["trial_id"],
            s=15,
            alpha=0.9,
            c=color_map[err],
            label=f'{err} - {reward_size} reward'
        )
    if baseline_err:
        # ---- Baseline description box ----
        bbox_text = (
            f"{baseline_type} = {baseline_err:.3f}\n"
        )

        ax.text(
            0.02, 0.98,
            bbox_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="black",
                alpha=0.55
            )
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Trial ID")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


########## Export trials above baseline (dict + Excel)
def export_trials_above_baseline(df, savedir, file_name):
    os.makedirs(savedir, exist_ok=True)

    trial_dict = (
        df.groupby("trial_id")["time_idx"]
        .apply(list)
        .to_dict()
    )

    with open(os.path.join(savedir, f"{file_name}.pkl"), "wb") as f:
        pickle.dump(trial_dict, f)

    df.to_excel(
        os.path.join(savedir, f"{file_name}.xlsx"),
        index=False
    )

    return trial_dict


def extract_consistently_false_trials(
    df_false,
    threshold=0.8
):
    """
    Identify trials that are falsely predicted (FP or FN) in more than
    `threshold` fraction of time points.

    Parameters
    ----------
    df_false : pd.DataFrame
        Required columns:
        - time_idx       : real time (seconds or bins)
        - trial_id   : trial index
        - error      : 'FP' or 'FN'

    threshold : float
        Consistency threshold (e.g. 0.8 for 80%).

    Returns
    -------
    pd.DataFrame with columns:
        - trial_id
        - false_fraction
        - error
    """

    # total number of unique time points
    total_timepoints = df_false["time_idx"].nunique()

    # count false predictions per trial
    counts = (
        df_false
        .groupby(["trial_id", "error"])["time_idx"]
        .nunique()
        .reset_index(name="n_false_timepoints")
    )

    # compute false fraction per trial
    counts["false_fraction"] = (
        counts["n_false_timepoints"] / total_timepoints
    )

    # filter by consistency threshold
    consistent_trials = counts[
        counts["false_fraction"] >= threshold
    ].reset_index(drop=True)

    return consistent_trials



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D


def plot_consistently_false_trials_1d(
    consistent_trials_df,
    title,
    savepath,
    jitter=0.00
):
    """
    1D scatter plot of consistently false-predicted trials.

    - x-axis: trial ID
    - color hue: FP vs FN
    - color intensity: false_fraction (scaled to data min/max)
    """

    if consistent_trials_df.empty:
        print("No consistent false trials to plot.")
        return

    fig, ax = plt.subplots(figsize=(13, 3))

    x = consistent_trials_df["trial_id"].values
    y = np.random.uniform(-jitter, jitter, size=len(x))
    frac = consistent_trials_df["false_fraction"].values
    err = consistent_trials_df["error"].values

    # ---- data-driven intensity limits ----
    frac_min = frac.min()
    frac_max = frac.max()

    if frac_min == frac_max:
        # avoid zero-range normalization
        frac_min = max(0.0, frac_min - 1e-3)
        frac_max = min(1.0, frac_max + 1e-3)

    # ---- strong base colors ----
    base_colors = {
        "FP": np.array([230, 97, 1]) / 255.0,   # strong orange
        "FN": np.array([202, 0, 32]) / 255.0    # strong red
    }

    # ---- build colors: hue = error, intensity = false_fraction ----
    colors = []
    for e, f in zip(err, frac):
        base = base_colors[e]

        # normalize f to [0, 1] based on data range
        alpha = (f - frac_min) / (frac_max - frac_min)
        alpha = np.clip(alpha, 0, 1)

        # interpolate between light gray and base color
        light = base * 0.5 + np.array([1.0, 1.0, 1.0]) * 0.5
        c = (1 - alpha) * light + alpha * base
        colors.append(c)

    ax.scatter(
        x,
        y,
        c=colors,
        s=80,
        edgecolor="none"
    )

    # ---- axis formatting ----
    ax.set_yticks([])
    ax.set_xlabel("Trial ID")
    ax.set_title(title)
    ax.set_xlim(x.min() - 1, x.max() + 1)

    # ---- legend (upper right) ----
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=base_colors["FP"], markersize=10, label='FP'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=base_colors["FN"], markersize=10, label='FN')
    ]

    ax.legend(
        handles=legend_elements,
        title="False classification",
        loc="upper right",
        frameon=True
    )

    # ---- colorbar: scaled to data min/max ----
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "consistency",
        [(0.9, 0.9, 0.9), (0, 0, 0)]
    )

    norm = mcolors.Normalize(vmin=frac_min, vmax=frac_max)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        pad=0.25,
        fraction=0.05
    )
    cbar.set_label("False prediction fraction")
    cbar.set_ticks([frac_min, frac_max])
    cbar.set_ticklabels([f"{frac_min:.2f}", f"{frac_max:.2f}"])

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)




def plot_all_scores_over_time(saveroot, scores_dic, roc_dic, pr_auc_dic, confusion_metrics, confusion_idx, subject_id, session, model, frame_bin, frame_rate,
                              error_type="sem", also_plot_per_fold=False):
    """
    error_type: "sem" (standard error of the mean)
    also_plot_per_fold: if True, plots each fold as a thin line (separate files, same directory)
    """
    savedir = os.path.join(saveroot, 'figures', f'Decoder {model} output', 'all_scores_over_time')
    os.makedirs(savedir, exist_ok=True)

    # time axis
    time_array = np.array([i / frame_rate for i in frame_bin], dtype=float)

    # Build scores_df: index=time bins, columns=metrics; cell=list of per-fold values
    scores_df = pd.DataFrame.from_dict(scores_dic, orient='index')

    # add scalar AUC per time bin (no folds)
    auc_series = pd.Series({k: v['roc_auc'] for k, v in roc_dic.items()})
    # align to scores_df index order
    scores_df['roc_auc_global'] = auc_series.reindex(scores_df.index).values

    # add scalar PR auc per time bin to scores_df
    pr_auc_series = pd.Series(pr_auc_dic)
    scores_df['pr_auc_global'] = pr_auc_series.reindex(scores_df.index).values

    # sanity check: same length
    assert len(scores_df) == len(time_array), "time_array and number of time bins must match."

    # Per time point normalized classification outcome
    df = confusion_to_time_percentage_df(confusion_metrics)
    save_path = os.path.join(saveroot, 'figures', f'Decoder {model} output', 'false predictions analysis')
    os.makedirs(save_path, exist_ok=True)
    plot_stacked_confusion_over_time_seconds(df,
                                             time_array,
                                             subject_id,
                                             session,
                                             save_path,
                                             title='Per time point normalized classification outcome')

    # get indexes of false predicted trials
    df_false = extract_false_predictions(confusion_idx, frame_rate)

    # Iterate metrics (columns)
    for col in scores_df.columns:
        col_series = scores_df[col]
        title = f'{col} score over time,subject{subject_id},session{session}'
        savepath = os.path.join(savedir, title + '.png')

        if col in ['roc_auc_global', 'pr_auc_global']:
            # scalar per time bin → simple line plot
            yvals = col_series.astype(float).values
            plot_score_over_time(yvals, time_array, title, savepath, ylabel=f"{col} score")

            ave_baseline, max_baseline = compute_baseline_thresholds(yvals, time_array)
            # plot false prediction trial indexes

            # all false trials id's
            plot_false_predictions_with_baseline_bbox(
                df_false,
                baseline_err=None,
                baseline_type=None,
                title="False predictions trials id's",
                savepath=os.path.join(save_path, f"False_predictions_trials_ids_{col}_{subject_id}_session{session}.png")
            )
            export_trials_path = os.path.join(save_path, 'false_trials_id')
            os.makedirs(export_trials_path, exist_ok=True)
            export_trials_above_baseline(df_false, export_trials_path, file_name=f'False_predictions_trials_ids_{col}_{subject_id}_session{session}')

            # plot trials that are consistently false-predicted (≥80%)
            threshold = 0.7
            consistent_trials = extract_consistently_false_trials(
                df_false,
                threshold=threshold
            )

            plot_consistently_false_trials_1d(
                consistent_trials_df=consistent_trials,
                title=f"Consistently false predicted trials (≥{int(threshold*100)}%)",
                savepath=os.path.join(save_path, f"consistent_false_trials_{col}_{subject_id}_session{session}.png")
            )

            # false trials that are filtered according to the average baseline
            df_filtered = filter_false_predictions_above_threshold(df_false, yvals, time_array, ave_baseline)
            plot_false_predictions_with_baseline_bbox(
                df_filtered,
                ave_baseline,
                'average baseline',
                title='False predictions above averaged baseline score',
                savepath=os.path.join(save_path, f'False_predictions_ids_ave_baseline_{col}_{subject_id}_session{session}.png')
            )
            export_trials_above_baseline(df_false, export_trials_path, file_name=f'False_predictions_ids_ave_baseline_{col}_{subject_id}_session{session}')



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


def plot_pc_activity_groups(savedir, pc_activity_dic, subject_id, session, model, frame_bin, frame_rate):
    savedir = os.path.join(savedir, 'figures', f'Decoder {model} output', 'pc_activity_groups')
    os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, f'pc_activity_groups,subject{subject_id},session{session}.png')

    # time axis
    time = np.array([i / frame_rate for i in frame_bin], dtype=float)

    # Color & style mapping
    style_map = {
        'TP': {'color': 'red', 'linestyle': '-'},
        'FN': {'color': 'red', 'linestyle': '--'},
        'TN': {'color': 'blue', 'linestyle': '-'},
        'FP': {'color': 'blue', 'linestyle': '--'}
    }

    plt.figure(figsize=(10, 5))

    for cls, style in style_map.items():
        mean_vals = []
        sem_vals = []

        for t in frame_bin:
            values = pc_activity_dic[t].get(cls, [])

            if len(values) > 0:
                mean_vals.append(np.mean(values))
                sem_vals.append(np.std(values) / np.sqrt(len(values)))
            else:
                mean_vals.append(np.nan)
                sem_vals.append(np.nan)

        mean_vals = np.array(mean_vals)
        sem_vals = np.array(sem_vals)

        # plot mean line
        plt.plot(
            time,
            mean_vals,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=2,
            label=cls
        )

        # shaded SEM
        plt.fill_between(
            time,
            mean_vals - sem_vals,
            mean_vals + sem_vals,
            color=style['color'],
            alpha=0.25
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Mean PC activity")
    plt.title(f"First PC activity across trial classification groups\nSubject {subject_id}, Session {session}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()







def plot_results(
        savedir,
        scores_dic,
        roc_dic,
        pr_auc_dic,
        all_frames_confusion,
        all_frames_confusion_idx,
        all_frames_pc_separated,
        subject_id,
        session,
        model,
        frame_bin,
        frame_rate
):
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
        saveroot=savedir,
        scores_dic=scores_dic,
        roc_dic=roc_dic,
        pr_auc_dic=pr_auc_dic,
        confusion_metrics = all_frames_confusion,
        confusion_idx = all_frames_confusion_idx,
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

    # 3) Plot pc activity separated to four classification groups
    plot_pc_activity_groups(
        savedir=savedir,
        pc_activity_dic=all_frames_pc_separated,
        subject_id=subject_id,
        session=session,
        model=model,
        frame_bin=frame_bin,
        frame_rate=frame_rate,
    )




