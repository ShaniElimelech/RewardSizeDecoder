import datajoint as dj
import pandas as pd
import numpy as np
from sqlalchemy import false


def get_restricted_table(restricted_table, restriction_list):
    if len(restriction_list) > 0:
        for restriction in restriction_list:
            restricted_table = restricted_table - restriction
    return restricted_table


def align_video_trials(subject_id, session, frame_rate, time_bin, dj_modules, clean_ignore=True, clean_omission=False, flag_electric_video=True):
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

    TrackingTrial = get_restricted_table((tracking.TrackingTrial & key & {'tracking_device_id': 3}), restriction_list)
    trial_video_data = TrackingTrial.fetch()
    trial_duration = trial_video_data["tracking_duration"].astype(float)
    t0_video = trial_video_data["tracking_start_time"].astype(float)
    TrialsStartFrame = np.zeros(trial_video_data.shape[0])
    for i in range(1, len(TrialsStartFrame)):
        TrialsStartFrame[i] = TrialsStartFrame[i-1] + int(np.ceil(trial_duration[i-1] * frame_rate))

    # Apply restrictions to base table
    img_FrameStartTrial = get_restricted_table((img.FrameStartTrial & key), restriction_list)
    trial_num = img_FrameStartTrial.fetch('trial', order_by='trial')

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

    for i_tr in range(len(trial_num)):
        if flag_electric_video:
            all_licks = LICK_ELECTRIC[LICK_ELECTRIC['trial'] == trial_num[i_tr]]['action_event_time']
            licks_after_go = all_licks[all_licks > go_time[i_tr]]
        else:
            all_licks = LICK_VIDEO[LICK_VIDEO['trial'] == trial_num[i_tr]]['lick_time_onset_relative_to_trial_start']
            licks_after_go = all_licks[all_licks > go_time[i_tr]]

        if len(licks_after_go) > 0:
            start_file[i_tr] = int(TrialsStartFrame[i_tr]) + int((float(licks_after_go[0])- t0_video[i_tr]) * frame_rate) + int(time_bin[0] * frame_rate)

            if start_file[i_tr] <= 0:
                start_file[i_tr] = float('nan')

        else:
            start_file[i_tr] = float('nan')

    return start_file, TrialsStartFrame


def align_video_trials_test(trial_num_lst, subject_id, session, frame_rate, time_bin, dj_modules, clean_ignore=True, clean_omission=False, flag_electric_video=True):
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

    TrackingTrial = get_restricted_table((tracking.TrackingTrial & key & {'tracking_device_id': 3} & [{'trial': t} for t in trial_num_lst]), restriction_list)
    trial_video_data = TrackingTrial.fetch()
    trial_duration = trial_video_data["tracking_duration"].astype(float)
    t0_video = trial_video_data["tracking_start_time"].astype(float)
    TrialsStartFrame = np.zeros(trial_video_data.shape[0])
    for i in range(1, len(TrialsStartFrame)):
        TrialsStartFrame[i] = TrialsStartFrame[i-1] + int(np.ceil(trial_duration[i-1] * frame_rate))

    # Apply restrictions to base table

    if flag_electric_video:
        # We align based on electric lickport, even if video does not exist
        exp2_ActionEvent = get_restricted_table((exp2.ActionEvent & key& [{'trial': t} for t in trial_num_lst]), restriction_list)
        LICK_ELECTRIC = exp2_ActionEvent.fetch()

    else:
        # We align based on video if it exists
        # We align to the first video-detected lick after lickport movement
        tracking_VideoNthLickTrial = get_restricted_table((tracking.VideoNthLickTrial & key& [{'trial': t} for t in trial_num_lst]), restriction_list)
        LICK_VIDEO = tracking_VideoNthLickTrial.fetch('lick_time_onset_relative_to_trial_start')

    BehaviorTrial_Event = get_restricted_table((exp2.BehaviorTrial.Event & key & 'trial_event_type="go"' & [{'trial': t} for t in trial_num_lst]), restriction_list)
    go_time = BehaviorTrial_Event.fetch('trial_event_time')
    start_file = np.zeros(len(trial_num_lst))

    for i_tr in range(len(trial_num_lst)):
        if flag_electric_video:
            all_licks = LICK_ELECTRIC[LICK_ELECTRIC['trial'] == trial_num_lst[i_tr]]['action_event_time']
            licks_after_go = all_licks[all_licks > go_time[i_tr]]
        else:
            all_licks = LICK_VIDEO[LICK_VIDEO['trial'] == trial_num_lst[i_tr]]['lick_time_onset_relative_to_trial_start']
            licks_after_go = all_licks[all_licks > go_time[i_tr]]

        if len(licks_after_go) > 0:
            start_file[i_tr] = int(TrialsStartFrame[i_tr]) + int((float(licks_after_go[0])- t0_video[i_tr]) * frame_rate) + int(time_bin[0] * frame_rate)

            if start_file[i_tr] <= 0:
                start_file[i_tr] = float('nan')

        else:
            start_file[i_tr] = float('nan')

    return start_file



