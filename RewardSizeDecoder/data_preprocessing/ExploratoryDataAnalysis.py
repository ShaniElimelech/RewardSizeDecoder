import pandas as pd
import datajoint as dj
import cv2   ### please install opencv-python package with pip and not conda otherwise you will have plugins problems
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def proportion_ignore_trials():
    ignore_df = pd.DataFrame(exp2.BehaviorTrial.fetch('subject_id', 'session', 'outcome', as_dict=True))
    grouped = ignore_df.groupby(['subject_id', 'session', 'outcome']).size().reset_index(name='count')
    session_totals = ignore_df.groupby(['subject_id', 'session']).size().reset_index(name='trials_total')
    result = pd.merge(grouped, session_totals, on=['subject_id', 'session'])
    result['proportion'] = result['count'] / result['trials_total']
    final_df = result[((result['subject_id'] == 464724) | (result['subject_id'] == 464725) | (
                result['subject_id'] == 463189) | (result['subject_id'] == 463190)) & (result['outcome'] == 'ignore')]
    final_df['subject_id'] = final_df['subject_id'].astype(str)
    plt.figure(figsize=(6, 6))
    sns.stripplot(
        data=final_df,
        x='subject_id',
        y='proportion',
        jitter=False,
        palette='Set2',
        alpha=0.7
    )

    # Calculate and plot means
    means = final_df.groupby('subject_id')['proportion'].mean().reset_index()
    sns.scatterplot(
        data=means,
        x='subject_id',
        y='proportion',
        color='black',
        marker='D',
        s=150,
        label='Mean'
    )
    # Customize plot
    plt.title('Proportion of ignore trials')
    plt.xlabel('Subject')
    plt.ylabel('Proportion')
    plt.legend(title='Legend')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def proportion_rewardsize():
    RewardSize_df = pd.DataFrame(exp2.TrialRewardSize.fetch('subject_id', 'session', 'reward_size_type', as_dict=True))
    grouped = RewardSize_df.groupby(['subject_id', 'session', 'reward_size_type']).size().reset_index(name='count')
    session_totals = RewardSize_df.groupby(['subject_id', 'session']).size().reset_index(name='trials_total')
    result = pd.merge(grouped, session_totals, on=['subject_id', 'session'])
    result['proportion'] = result['count'] / result['trials_total']
    final_df = result[
        (result['subject_id'] == 464724) | (result['subject_id'] == 464725) | (result['subject_id'] == 463189) | (
                    result['subject_id'] == 463190)]
    plt.figure(figsize=(6, 6))
    sns.stripplot(
        data=final_df,
        x='reward_size_type',
        y='proportion',
        jitter=False,
        palette='Set2',
        alpha=0.5
    )
    # Calculate and plot means
    means = final_df.groupby('reward_size_type')['proportion'].mean().reset_index()
    sns.scatterplot(
        data=means,
        x='reward_size_type',
        y='proportion',
        color='black',
        marker='D',
        s=150,
        label='Mean'
    )
    # Customize plot
    plt.title('Proportion by Reward Size Type')
    plt.xlabel('Reward Size Type')
    plt.ylabel('Proportion')
    plt.legend(title='Legend')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def overlapp_trial_frames():
    frame_rate = 2
    time_bin = [-2, 5]
    subject_lst = [464724, 464725, 463189, 463190]
    session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 5, 6, 7, 8, 9], [1, 2, 3, 4, 9], [2, 3, 4, 5, 6, 10]]
    all_pos_diff = []
    num_trials = 0
    for i, subject_id in enumerate(subject_lst):
        sessions = session_lists[i]
        for session in sessions:
            start_file, end_file, _, _ = align_trials_and_get_lickrate(subject_id, session, frame_rate, time_bin)
            num_trials += len(start_file[~np.isnan(start_file)])
            diff_files = end_file[:-1] - start_file[1:]
            pos_diff = diff_files[diff_files >= 0]
            all_pos_diff.extend(pos_diff)

    # Count the occurrences of each unique length
    unique_diff, counts = np.unique(all_pos_diff, return_counts=True)
    normalized_counts = counts / num_trials
    # Plotting the bar chart
    plt.bar(unique_diff, normalized_counts)
    plt.xlabel('number of overlapping frames')
    plt.ylabel('Count (normalized by number of trials)')
    plt.title('Overlapping trials using bin= [-2,5] trial alignment')
    plt.show()


def missing_video_frames():
    camera_num = 0
    all_missing_frames = []
    all_tot_frames = []
    for i, subject_id in enumerate(subject_lst):
        sessions = session_lists[i]
        for session in sessions:
            # generate for all trials aligned video and neural data and assemble it into a full session
            df_ndata = pd.DataFrame()
            all_video_session = []

            for trail_index, trial_data in get_session_trials_aligned_frames(subject_id, session, camera_num,
                                                                             dj_modules, clean_ignore=True, clean_omission=False):
                length = len(trial_data['trial_neural_frames_indexes_wo_video'])
                all_missing_frames.append(int(length))
                tot_frames = trial_data['trial_neural_total_frames']
                all_tot_frames.append(tot_frames)

    # Count the occurrences of each unique length
    unique_lengths, counts = np.unique(all_missing_frames, return_counts=True)

    # Plotting the bar chart
    plt.bar(unique_lengths, counts)
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.title('Histogram of Missing Frame Lengths')
    plt.show()
    # Plotting scatter chart
    plt.scatter(all_missing_frames, all_tot_frames)
    plt.xlabel('Number of frames with no video')
    plt.ylabel('Full trial length')
    plt.title('Full trial length vs frames with no video')
    plt.show()


def find_missing_video_frames(*args, **kwargs):
    no_video_neural_frames_session, start_trials, end_trials = video2neural2reward_labels_alignment(*args, **kwargs)
    bin_frame_dic = {f'{frame}': 0 for frame in range(-4, 11)}
    trial_missing_time_count_dic = defaultdict(int)
    for no_video_frame in no_video_neural_frames_session:
        for trial_frame_idx, _ in enumerate(start_trials):
            if start_trials[trial_frame_idx] <= no_video_frame <= end_trials[trial_frame_idx]:
                frame = int(-4 + no_video_frame - start_trials[trial_frame_idx])
                bin_frame_dic[f'{frame}'] += 1
                trial_missing_time_count_dic[f'{int(trial_frame_idx)}'] += 1
                break
            elif start_trials[trial_frame_idx] > no_video_frame:
                break
            elif end_trials[trial_frame_idx] < no_video_frame:
                continue

    trial_count_norm = len(trial_missing_time_count_dic.keys()) / len(start_trials)
    return trial_count_norm, bin_frame_dic, len(start_trials)


def alignment_missing_video_frames():
    dj_info = ("arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com", 'ShaniE', 'opala')
    frame_rate = 2
    time_bin = (-2, 5)
    save_path = None
    trial_count_all_sessions = []
    plot_sessions = []
    bin_frame_dic_all = {f'{frame}': 0 for frame in range(-4, 11)}
    subject_lst = [464724, 464725, 463189, 463190]
    session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 2, 3, 4, 9], [2, 3, 5, 6, 10]]
    # Optional: color mapping per subject
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    subject_colors = {subject_id: colors[i] for i, subject_id in enumerate(subject_lst)}
    subject_labels = []
    num_trials_tot = 0
    for i, subject_id in enumerate(subject_lst):
        sessions = session_lists[i]
        for session in sessions:
            print(f'subject:{subject_id}, session:{session} started')
            trial_count_norm, bin_frame_dic, num_trials = find_missing_video_frames(subject_id, session, frame_rate,
                                                                                    time_bin, dj_info,
                                                                                    save_path,
                                                                                    handle_omission='convert',
                                                                                    clean_ignore=True)
            trial_count_all_sessions.append(trial_count_norm)
            plot_sessions.append(f'{subject_id}-{session}')
            subject_labels.append(subject_id)
            bin_frame_dic_all = {k: bin_frame_dic[k] + bin_frame_dic_all[k] for k in bin_frame_dic_all}
            num_trials_tot += num_trials

    bin_frame_dic_all = {k: bin_frame_dic_all[k] / num_trials_tot for k in bin_frame_dic_all}

    # Plot 1: Normalized trial counts per session
    plt.figure(figsize=(12, 6))
    bar_colors = [subject_colors[s] for s in subject_labels]
    plt.bar(plot_sessions, trial_count_all_sessions, color=bar_colors)
    plt.xticks(rotation=90)
    plt.xlabel('Subject-Session')
    plt.ylabel('Normalized Trial Count')
    plt.title('Count of missing video frames trials per Session')
    plt.tight_layout()
    plt.show()

    # Plot 2: Summed missing video frame counts across all sessions
    plt.figure(figsize=(10, 5))
    frames = list(bin_frame_dic_all.keys())
    counts = list(bin_frame_dic_all.values())
    plt.bar(frames, counts, color='gray')
    plt.xlabel('Frame (relative to event)')
    plt.ylabel('Missing Frame Count')
    plt.title('Missing Video Frames (All Sessions Combined)')
    plt.tight_layout()
    plt.show()