import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_score_over_time(column_df, time_array, title, savepath):
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, column_df)
    plt.title(title)
    plt.xlabel('Time[s]')
    plt.ylabel('Score')
    plt.savefig(savepath)
    plt.close()


def plot_all_scores_over_time(savedir, scores_dic, roc_dic, subject_id, session, model, frame_bin, frame_rate):
    savedir = os.path.join(savedir, 'figures', f'Decoder {model} output', 'all_scores_over_time')
    os.makedirs(savedir, exist_ok=True)
    time_array = [i / frame_rate for i in frame_bin]
    scores_df = pd.DataFrame.from_dict(scores_dic, orient='index')
    auc = [v['auc'] for v in roc_dic.values()]
    scores_df['auc'] = auc
    for col in scores_df.columns:
        savepath = os.path.join(savedir, f'{col} score over time,subject{subject_id},session{session}.png')
        title = f'{col} score over time,subject{subject_id},session{session}'
        plot_score_over_time(scores_df[col], time_array, title, savepath)


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

