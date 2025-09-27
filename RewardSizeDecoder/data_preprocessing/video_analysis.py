import cv2   ### please install opencv-python package with pip and not conda otherwise you will have plugins problems
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
from scipy import stats as sc
import pickle
import datajoint as dj
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
from ExtractVideoNeuralAlignment import get_trials_data_table_for_mouse_session, get_dff_table_for_mouse_session, get_trial_video_frames_groups, get_trial_neural_frames
from trial_alignment import align_trials_and_get_lickrate, get_reward_size_labels


def plot_video_groups(
    video_groups,         # list/array of 2D frames
    frames_ave,           # 2D numpy array (average frame)
    frame_time,
    trial_num,
    reward_size,
    session,
    subject_id,
    savep,
    *,
    cmap="gray",
    dpi=300,
    thumb_per_inch=0.9,   # thumbnail display size (inches)
    avg_width_in_thumbs=4 # average panel width (in "thumb equivalents")
):

    # ---------- paths & title ----------
    savepath = os.path.join(savep, 'video_groups')
    os.makedirs(savepath, exist_ok=True)
    fname = f"frame time_{frame_time}_trial_{trial_num}_reward size_{reward_size}_session_{session}_subject_{subject_id}.png"
    saveimg = os.path.join(savepath, fname)
    title  = f"frame time: {frame_time} | trial: {trial_num} | reward size: {reward_size} | session: {session} | subject: {subject_id}"

    # ---------- prep data ----------
    frames = [np.asarray(f) for f in video_groups]
    ave = np.asarray(frames_ave)

    # Shared intensity range
    stacked_for_scale = np.stack(frames + [ave], axis=0)
    vmin = np.percentile(stacked_for_scale, 2)
    vmax = np.percentile(stacked_for_scale, 98)

    # ---------- grid shape for thumbnails ----------
    n = len(frames)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    # ---------- figure size ----------
    thumbs_w_in = ncols * thumb_per_inch
    thumbs_h_in = nrows * thumb_per_inch
    avg_w_in = avg_width_in_thumbs * thumb_per_inch
    fig_w = avg_w_in + thumbs_w_in + 0.3
    fig_h = max(thumbs_h_in, avg_w_in)

    # ---------- build layout ----------
    plt.close('all')
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=True)
    fig.suptitle(title, fontsize=10)

    gs = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[avg_w_in, thumbs_w_in])

    # --- Average frame ---
    ax_avg = fig.add_subplot(gs[0, 0])
    ax_avg.imshow(ave, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax_avg.set_title("Average", fontsize=11, pad=6)
    ax_avg.axis('off')
    for spine in ax_avg.spines.values():
        spine.set_linewidth(1.2)

    # --- Thumbnails grid ---
    gs_right = gs[0, 1].subgridspec(nrows, ncols, wspace=0.02, hspace=0.02)

    for i, f in enumerate(frames):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs_right[r, c])
        ax.imshow(f, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # ---------- save ----------
    fig.savefig(saveimg, dpi=dpi, bbox_inches='tight')
    #fig.savefig(saveimg.replace(".png", ".pdf"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_video_ave_for_pairs(
    trial_video_frames_ave_current,   # list of 2D arrays (current trial averages per frame time)
    trial_video_frames_ave_next,      # list of 2D arrays (next trial averages per frame time)
    existing_video_frames,            # list/array of frame-time stamps (len = number of columns)
    trial_num,
    reward_size_current,              # acts as the "type" for current trial
    reward_size_next,                 # acts as the "type" for next trial
    session,
    subject_id,
    saveroot,
    *,
    cmap="gray",
    dpi=300,
    thumb_size_in=1.1,                # width of each column (inches)
    row_height_in=1.1,                # height per image (inches)
    percentile_range=(2, 98),         # robust intensity scaling
    vmin=None,
    vmax=None
):
    """
    Layout:
        Row 1: current trial average frames (one per frame time)
        Row 2: next   trial average frames (aligned by column)
        Under each column (on the bottom row), the frame time is shown as an x-label.

    Notes:
      - Uses shared vmin/vmax across both rows for comparability.
      - If lengths differ, uses the minimum across the three inputs.
      - Any missing/None frames are shown as 'missing'.
    """
    # ---------- ensure lengths align ----------
    n = min(len(trial_video_frames_ave_current),
            len(trial_video_frames_ave_next),
            len(existing_video_frames))
    if n == 0:
        raise ValueError("No frames to plot: one of the input lists is empty.")

    cur = [trial_video_frames_ave_current[i] for i in range(n)]
    nxt = [trial_video_frames_ave_next[i] for i in range(n)]
    times = [existing_video_frames[i] for i in range(n)]

    # ---------- compute shared intensity range ----------
    if vmin is None or vmax is None:
        valid_arrays = []
        for arr in cur + nxt:
            if arr is not None:
                a = np.asarray(arr)
                if a.size > 0:
                    valid_arrays.append(a)
        if len(valid_arrays) == 0:
            raise ValueError("All frames are None or empty; cannot compute intensity scale.")
        stack = np.stack(valid_arrays, axis=0)
        p_lo, p_hi = percentile_range
        auto_vmin = np.percentile(stack, p_lo)
        auto_vmax = np.percentile(stack, p_hi)
        if vmin is None: vmin = auto_vmin
        if vmax is None: vmax = auto_vmax

    # ---------- figure size ----------
    fig_w = max(3.5, n * thumb_size_in)      # keep some minimum width for very small n
    fig_h = 2 * row_height_in + 0.6          # extra for row titles / xlabels

    # ---------- build figure ----------
    plt.close('all')
    fig, axes = plt.subplots(
        nrows=2, ncols=n, figsize=(fig_w, fig_h), dpi=dpi,
        constrained_layout=True
    )

    # If n == 1, matplotlib returns 1D axes; normalize to 2D indexable shape
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    # ---------- titles ----------
    # Row titles use reward size as "type"
    row1_title = f"Trial {trial_num} | type: {reward_size_current}"
    row2_title = f"Trial {trial_num + 1} | type: {reward_size_next}"

    # Put the row titles above the first column of each row (left-aligned)
    axes[0, 0].set_title(row1_title, loc='left', fontsize=11, pad=6)
    axes[1, 0].set_title(row2_title, loc='left', fontsize=11, pad=6)

    # Figure-level title with metadata
    fig.suptitle(
        f"average frames session: {session} | subject: {subject_id}",
        fontsize=10, y=1.02
    )

    # ---------- plot images ----------
    def _imshow(ax, img):
        if img is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
            ax.set_axis_off()
            return
        a = np.asarray(img)
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_axis_off()
        # subtle border
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    for c in range(n):
        _imshow(axes[0, c], cur[c])
        _imshow(axes[1, c], nxt[c])
        # Put the frame timestamp under the bottom image in each column
        axes[1, c].text(
            0.5, -0.15,  # X=center, Y just below the axes
            str(times[c]),  # the label
            transform=axes[1, c].transAxes,
            ha="center", va="top",
            fontsize=9)


    # Slightly tighten vertical spacing between rows
    plt.subplots_adjust(hspace=0.06, wspace=0.02)

    # ---------- save ----------
    out_dir = os.path.join(saveroot, "video_pairs")
    os.makedirs(out_dir, exist_ok=True)
    fname = (
        f"pairs_trial_{trial_num}_{trial_num+1}"
        f"_types_{reward_size_current}_{reward_size_next}"
        f"_session_{session}_subject_{subject_id}.png"
    )
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def get_session_trials_aligned_frames(subject_id, session, camera_num, dj_modules, original_video_path, time_bin, frame_rate, saveroot, clean_ignore=False,
                                      clean_omission=False, compute_neural_data=False, drop_neural_frames_with_no_video=True):

    video_neural = dj_modules['video_neural']
    img = dj_modules['img']
    tracking = dj_modules['tracking']
    exp2 = dj_modules['exp2']

    session_string = f"session{session}"
    all_videos_path = original_video_path
    if camera_num not in [0, 1]:
        raise ValueError("Camera number must be 0 or 1!")

    # get trial new start and end frames - global alignment of trials to the first lick after go cue
    start_trials, end_trials, _, _ = align_trials_and_get_lickrate(subject_id, session, frame_rate, time_bin,
                                                                   dj_modules, clean_ignore, clean_omission,
                                                                   flag_electric_video=True)
    # get reward size labels
    handle_omission = 'keep'
    #reward_labels = get_reward_size_labels(subject_id, session, dj_modules, handle_omission, clean_ignore)
    key = {'subject_id': subject_id, 'session': session}
    reward_labels = ((exp2.TrialRewardSize & key) - tracking.TrackingTrialBad - tracking.VideoGroomingTrial).fetch('reward_size_type')
    large_labels_idx = np.where(reward_labels == 'large')[0]

    key = {'subject_id': subject_id, 'session': session}
    outcome_table = ((exp2.BehaviorTrial & key) - tracking.TrackingTrialBad - tracking.VideoGroomingTrial).fetch('outcome')
    ignore_trails_idx = np.where(outcome_table == 'ignore')[0]

    # Get data from DataJoint
    trials_data = get_trials_data_table_for_mouse_session(subject_id, session, camera_num, tracking, video_neural, exp2,
                                                          clean_ignore, clean_omission)
    if trials_data.empty:
        raise ValueError(f'There is no neural data for subject{subject_id} session{session}')

    dff_data = None
    if compute_neural_data:
        dff_data = get_dff_table_for_mouse_session(subject_id, session, img)
        if dff_data.empty:
            raise ValueError(f'There is no neural data for subject{subject_id} session{session}')


    for large_idx in ignore_trails_idx:
        batch_idx = range(large_idx,large_idx+1)
        all_session_video_frames_groups = []
        all_session_neural_frames = []
        for index, row in trials_data.iterrows():
            if index not in batch_idx:
                continue
            print(row['trial'])
            trial_video_frames = get_trial_video_frames_groups(row, all_videos_path, subject_id, session_string, camera_num)
            trial_neural_frames_indexes_with_video = row["trial_neural_frames_indexes"][:len(trial_video_frames)]

            if compute_neural_data:
                trial_neural_frames = get_trial_neural_frames(dff_data, row["trial_neural_frames_indexes"],
                                                              len(trial_video_frames), drop_neural_frames_with_no_video)
            else:
                trial_neural_frames = None

            all_session_video_frames_groups.extend(trial_video_frames)
            all_session_neural_frames.extend(trial_neural_frames_indexes_with_video)

        time_frames = range(time_bin[0]* frame_rate, time_bin[1]* frame_rate +1)
        for trial_idx in batch_idx:
            if trial_idx != batch_idx[0]:
                continue
            trial_num = trials_data.iloc[trial_idx]['trial']
            if trial_idx not in ignore_trails_idx:
                if reward_labels[trial_idx+1] == 'large' and trial_idx+1 not in ignore_trails_idx:
                    reward_size_current = reward_labels[trial_idx]
                    reward_size_next = reward_labels[trial_idx+1]
                    print(f'trial {trial_num}- {reward_size_current} -> {reward_size_next}')

                    trial_video_frames_ave_current = []
                    trial_video_frames_ave_next = []
                    existing_video_frames = []
                    for i, time_frame in enumerate(time_frames):
                        # current trial frame
                        frame_current = start_trials[trial_idx] + i
                        neural_frame_idx_current = np.where(np.array(all_session_neural_frames) == frame_current)[0]
                        if len(neural_frame_idx_current) == 0:
                            continue
                        neural_frame_idx_current = neural_frame_idx_current[0]
                        video_frames_group_current = all_session_video_frames_groups[neural_frame_idx_current]
                        video_frame_ave_current = np.average(video_frames_group_current, axis=0)

                        # next trial frame
                        frame_next = start_trials[trial_idx+1] + i
                        neural_frame_idx_next = np.where(np.array(all_session_neural_frames) == frame_next)[0]
                        if len(neural_frame_idx_next) == 0:
                            continue
                        neural_frame_idx_next = neural_frame_idx_next[0]
                        video_frames_group_next = all_session_video_frames_groups[neural_frame_idx_next]
                        video_frame_ave_next = np.average(video_frames_group_next, axis=0)
                        existing_video_frames.append(time_frame)
                        trial_video_frames_ave_current.append(video_frame_ave_current)
                        trial_video_frames_ave_next.append(video_frame_ave_next)

                        plot_video_groups(video_frames_group_current, video_frame_ave_current, time_frame, trial_num, reward_size_current, session, subject_id, saveroot)
                        plot_video_groups(video_frames_group_next, video_frame_ave_next, time_frame, trial_num+1, reward_size_next, session, subject_id, saveroot)

                    # plot pairs of consecutive trials video averages frames (regular\omission -> large)
                    plot_video_ave_for_pairs(trial_video_frames_ave_current, trial_video_frames_ave_next, existing_video_frames, trial_num, reward_size_current, reward_size_next,
                                             session, subject_id, saveroot)
                    break

            else:

                # plot ignore trials frames
                reward_size = reward_labels[trial_idx]
                print(f'trial {trial_num}- {reward_size}')
                trial_video_frames_ave = []
                saveroot_ignore = os.path.join(saveroot, 'ignore trials')
                os.makedirs(saveroot_ignore, exist_ok=True)
                for i in range(len(all_session_video_frames_groups)):
                    if i > 15:
                        break
                    video_frames_group = all_session_video_frames_groups[i]
                    video_frame_ave = np.average(video_frames_group, axis=0)
                    trial_video_frames_ave.append(video_frame_ave)

                    plot_video_groups(video_frames_group, video_frame_ave, i, trial_num, reward_size, session, subject_id, saveroot_ignore)














subject_id = 464724
session = 4
time_bin = (-2,5)
frame_rate = 2
camera_num = 0

# Connect to Datajoint and load modules
dj.config['database.host'] = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
dj.config['database.user'] = 'ShaniE'
dj.config['database.password'] = 'opala'
conn = dj.conn()
img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')
video_neural = dj.VirtualModule('VIDEONEURAL', "lab_videoNeuralAlignment")
exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')
dj_modules = {'img': img, 'tracking': tracking, 'exp2': exp2, 'video_neural': video_neural}

original_video_path = 'E:/Arseny_behavior_video'
saveroot = 'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/video_analysis'
os.makedirs(saveroot, exist_ok=True)

get_session_trials_aligned_frames(subject_id, session, camera_num, dj_modules, original_video_path, time_bin, frame_rate, saveroot)