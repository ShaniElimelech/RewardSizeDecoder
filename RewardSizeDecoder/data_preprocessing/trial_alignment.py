import pandas as pd
import datajoint as dj
import cv2   ### please install opencv-python package with pip and not conda otherwise you will have plugins problems
import os
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from .ExtractVideoNeuralAlignment import get_session_trials_aligned_frames, save_as_video
from collections import defaultdict


def get_restricted_table(restricted_table, restriction_list):
    if len(restriction_list) > 0:
        for restriction in restriction_list:
            restricted_table = restricted_table - restriction
    return restricted_table


def align_trials_and_get_lickrate(subject_id, session, frame_rate, time_bin, dj_modules, clean_ignore, clean_omission,
                                  flag_electric_video=True):
    # unpacking data joint modules
    exp2 = dj_modules['exp2']
    img = dj_modules['img']
    tracking = dj_modules['tracking']
    key = {'subject_id': subject_id, 'session': session}  # specify key for dj fetching
    # Base exclusion - removing bad trials and grooming trials
    restriction_list = [tracking.TrackingTrialBad, tracking.VideoGroomingTrial]
    # Conditional exclusions - ignore and omission trials
    if clean_ignore:
        restriction_list.append(exp2.BehaviorTrial & 'outcome="ignore"')
    if clean_omission:
        restriction_list.append(exp2.TrialRewardSize & 'reward_size_type="omission"')

    # Apply restrictions to base table
    img_FrameStartTrial = get_restricted_table((img.FrameStartTrial & key), restriction_list)
    TrialsStartFrame = img_FrameStartTrial.fetch('session_epoch_trial_start_frame', order_by='trial')
    trial_num = img_FrameStartTrial.fetch('trial', order_by='trial')

    if len(TrialsStartFrame) == 0:
        img_FrameStartFile = get_restricted_table((img.FrameStartFile & key), restriction_list)
        TrialsStartFrame = img_FrameStartFile.fetch('session_epoch_file_start_frame', order_by='session_epoch_file_num')
        trial_num = img_FrameStartFile.fetch('trial', order_by='trial')
        TrialsStartFrame = TrialsStartFrame[trial_num]

    if flag_electric_video:
        # We align based on electric lickport, even if video does not exist
        exp2_ActionEvent = get_restricted_table((exp2.ActionEvent & key), restriction_list)
        LICK_ELECTRIC = exp2_ActionEvent.fetch()

    else:
        # We align based on video if it exists
        # We align to the first video-detected lick after lickport movement
        tracking_VideoNthLickTrial = get_restricted_table((tracking.VideoNthLickTrial & key), restriction_list)
        LICK_VIDEO = tracking_VideoNthLickTrial.fetch('lick_time_onset_relative_to_trial_start')

    BehaviorTrial_Event = get_restricted_table((exp2.BehaviorTrial.Event & key & 'trial_event_type="go"'), restriction_list)
    go_time = BehaviorTrial_Event.fetch('trial_event_time')
    start_file = np.zeros(len(trial_num))
    end_file = np.zeros(len(trial_num))
    lick_tr_times_relative_to_first_lick_after_go = []
    lick_tr_total = []

    for i_tr in range(len(trial_num)):
        if flag_electric_video:
            all_licks = LICK_ELECTRIC[LICK_ELECTRIC['trial'] == trial_num[i_tr]]['action_event_time']
            licks_after_go = all_licks[all_licks > go_time[i_tr]]
        else:
            all_licks = LICK_VIDEO[LICK_VIDEO['trial'] == trial_num[i_tr]]['lick_time_onset_relative_to_trial_start']
            licks_after_go = all_licks[all_licks > go_time[i_tr]]

        if len(licks_after_go) > 0:
            start_file[i_tr] = int(TrialsStartFrame[i_tr]) + int(float(licks_after_go[0]) * frame_rate) + int(
                time_bin[0] * frame_rate)
            end_file[i_tr] = start_file[i_tr] + int(float(time_bin[1] - time_bin[0]) * frame_rate) - 1
            lick_tr_times_relative_to_first_lick_after_go.append(all_licks - licks_after_go[0])
            lick_tr_total.append(np.sum((lick_tr_times_relative_to_first_lick_after_go[i_tr] >= float(time_bin[0])) & (
                        lick_tr_times_relative_to_first_lick_after_go[i_tr] <= float(time_bin[-1]))))
            if start_file[i_tr] <= 0:
                start_file[i_tr] = float('nan')
                end_file[i_tr] = float('nan')

        else:
            start_file[i_tr] = float('nan')
            end_file[i_tr] = float('nan')
            lick_tr_total.append(0)
            lick_tr_times_relative_to_first_lick_after_go.append([])

    return start_file, end_file, lick_tr_times_relative_to_first_lick_after_go, lick_tr_total


def get_reward_size_labels(subject_id, session, dj_modules, handle_omission, clean_ignore):
    key = {'subject_id': subject_id, 'session': session}
    exp2 = dj_modules['exp2']
    tracking = dj_modules['tracking']
    # Base restriction
    restriction = (exp2.TrialRewardSize & key)
    full_reward_labels = restriction.fetch('reward_size_type')
    restriction = restriction - tracking.TrackingTrialBad - tracking.VideoGroomingTrial

    if clean_ignore:
        restriction -= (exp2.BehaviorTrial & 'outcome="ignore"')

    if handle_omission == 'clean':
        restriction &= ['reward_size_type="large"', 'reward_size_type="regular"']

    clean_reward_labels = restriction.fetch('reward_size_type')
    percentage_lost = round((1 - len(clean_reward_labels)/len(full_reward_labels)) * 100)

    if percentage_lost > 80:
        if percentage_lost > 95:
            raise ValueError(f'After cleaning bad trials/ ignore trials more than {percentage_lost}% of trials are lost, skipping this session')

        else:
            print(f'Warning!!! There are {len(full_reward_labels - clean_reward_labels)} trials that were lost after cleaning out of {len(full_reward_labels)}'
                  f'{percentage_lost} percent of trials were lost')

    large_reward_perc = round((sum(x == 'large' for x in clean_reward_labels) / len(clean_reward_labels)) * 100)
    if large_reward_perc < 2:
        raise ValueError(f'After cleaning bad trials/ ignore trials only {large_reward_perc}% of trials are large, skipping this session')


    if handle_omission == 'convert':
        clean_reward_labels[clean_reward_labels == 'omission'] = 'regular'

    return clean_reward_labels


def video2neural2reward_labels_alignment(subject_id, session, frame_rate, time_bin: tuple, dj_info: tuple, save_path,
                                         handle_omission='convert', clean_ignore=True):

    """
    A multi alignment function that gets subject id and session and align all datasets (video, neural activity, reward size labels) that goes into the reward decoder.
    handle_omission: user can choose to throw out omission trials ('clean'), convert label to regular trial ('convert') or keep omission trials ('keep') in case of multi classification. clean and convert options are for binary classification.
    clean_ignore: if true, ignore trials are thrown out.
    bad trials and grooming trials are automatically thrown out
    time_bin: global time period for all trials to be aligned to according to first lick ('trial_start', 'trial_end')
    frame rate: neural frame rate
    """

    # Connect to Datajoint
    host_path, user_name, password = dj_info
    dj.config['database.host'] = host_path
    dj.config['database.user'] = user_name
    dj.config['database.password'] = password
    conn = dj.conn()
    img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
    tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')
    exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')
    video_neural = dj.VirtualModule('VIDEONEURAL', "lab_videoNeuralAlignment")
    dj_modules = {'img': img, 'tracking': tracking, 'exp2': exp2, 'video_neural': video_neural}

    assert handle_omission in ['keep', 'clean', 'convert'], ('Invalid handle_omission, '
                                                             'value should be "keep" or "clean" or "convert"')

    clean_omission = True if handle_omission == 'clean' else False
    # get trial new start and end frames - global alignment of trials to the first lick after go cue
    start_trials, end_trials, _, _ = align_trials_and_get_lickrate(subject_id, session, frame_rate, time_bin,
                                                                   dj_modules, clean_ignore, clean_omission,
                                                                   flag_electric_video=True)
    '''   
    # get reward size labels
    reward_labels = get_reward_size_labels(subject_id, session, dj_modules, handle_omission, clean_ignore)

    assert len(reward_labels) == len(start_trials), ('Lengths of reward_labels and start_trials do not match, '
                                                  'please compare the lengths of original datasets in datajoint')
    '''
    # get aligned and downsampled video and neural activity
    #try:
    for camera_num in [0, 1]:  # align data for each camera
        # generate for all trials aligned video and neural data and assemble it into a full session
        df_ndata = pd.DataFrame()
        all_video_session = []
        no_video_neural_frames_session = []

        for trail_index, trial_data in get_session_trials_aligned_frames(subject_id, session, camera_num,
                                                                         dj_modules, clean_ignore, clean_omission):
            #df_ndata = pd.concat([df_ndata, trial_data['trial_neural_frames']], ignore_index=True, axis=1)
            # take the mean of all video frames that correspond to their matching neural frame
            #trial_video_frames = [np.array(frames).mean(axis=0) for frames in
                                  #trial_data['trial_video_frames_groups']]
            #all_video_session.extend(trial_video_frames)
            no_video_neural_frames_session.extend(trial_data['trial_neural_frames_indexes_wo_video'])
        '''
        # short_ndata = df_ndata.to_numpy()  # convert df to npy
        short_vdata = np.array(all_video_session)  # convert df to npy
        save_downsampled_video_path = os.path.join(save_path,'downsampled_n_v_data', f'{subject_id}', f'session{session}')
        os.makedirs(save_downsampled_video_path, exist_ok=True)
        save_as_video(short_vdata.astype(np.uint8),savepath=f'{save_downsampled_video_path}\downsampled_video_cam{camera_num}.avi')
        # np.save(f'{save_downsampled_video_path}\short_ndata.npy', short_ndata)
        '''
    #except (ValueError, RuntimeError, Exception) as e:
        #print(f"Handled error: {e}\nskipping subject{subject_id} session{session}")
        #return

    return no_video_neural_frames_session, start_trials, end_trials


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


if __name__ == "__main__":
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
            trial_count_norm, bin_frame_dic, num_trials = find_missing_video_frames(subject_id, session, frame_rate, time_bin, dj_info,
                                                                        save_path, handle_omission='convert',
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
    