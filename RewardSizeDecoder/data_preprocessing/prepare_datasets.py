import os.path
from .trial_alignment import align_trials_and_get_lickrate, get_reward_size_labels
from .VideoPipeline import Video, VideoPair
import numpy as np
import datajoint as dj
import logging
from .video_binning import align_video_trials


def keep_pure_trials(video_start_trials, video_og_start_trials, reward_labels, video_data, step=1, preserve_large=False):
    # Ensure numpy arrays
    video_start_trials = np.asarray(video_start_trials).copy()
    video_og_start_trials = np.asarray(video_og_start_trials)
    reward_labels = np.asarray(reward_labels).copy()

    n_trials = len(reward_labels)
    bad_trials_idx = []
    bad_large_idx = []
    bad_l = []

    for trial_num in range(n_trials):
        # Neighbor indices in [trial_num-step, trial_num+step] excluding self
        neighbors = np.concatenate([
            np.arange(trial_num - step, trial_num),
            np.arange(trial_num + 1, trial_num + step + 1)
        ])

        # Keep only valid neighbor indices
        if neighbors.size:
            neighbors = neighbors[(neighbors >= 0) & (neighbors < n_trials)]

        # Any neighbor labeled 'large'
        has_large_neighbor = neighbors.size > 0 and np.any(reward_labels[neighbors] == 'large')

        if has_large_neighbor:
            if reward_labels[trial_num] == 'large' and preserve_large:
                # If current and previous trials are 'large', delete from video data
                left_neighbors = np.arange(trial_num - step, trial_num)
                left_neighbors = left_neighbors[left_neighbors >= 0]
                has_previous_large_neighbor = left_neighbors.size > 0 and np.any(reward_labels[left_neighbors] == 'large')
                if has_previous_large_neighbor:
                    '''
                    start_original_trial = video_og_start_trials[trial_num]
                    # Compute end index
                    if len(video_og_start_trials) > trial_num + 1:
                        end_original_trial = video_og_start_trials[trial_num + 1]
                    else:
                        end_original_trial = len(video_data)

                    bad_large_idx.extend(range(int(start_original_trial), int(end_original_trial)))
                    trial_duration = int(end_original_trial - start_original_trial)
                    # Shift all subsequent trial start indices
                    if trial_num + 1 < len(video_start_trials):
                        video_start_trials[trial_num + 1:] = video_start_trials[trial_num + 1:] - trial_duration
                    '''
                    bad_trials_idx.append(trial_num)
                    bad_l.append(trial_num)

                else:
                    continue

            else:
                bad_trials_idx.append(trial_num)
                if reward_labels[trial_num] == 'large':
                    bad_l.append(trial_num)

    binary_rewards = reward_labels == 'large'
    count_large_tot = np.sum(binary_rewards)
    count_bad_large = len(bad_l)
    if bad_large_idx:
        bad_large_idx = np.unique(np.clip(np.asarray(bad_large_idx, dtype=int), 0, len(video_data) - 1))
        video_data = np.delete(video_data, bad_large_idx, axis=0)  # axis=0 for safety

    if bad_trials_idx:
        bad_trials_idx = np.unique(np.asarray(bad_trials_idx, dtype=int))
        video_start_trials = np.delete(video_start_trials, bad_trials_idx, axis=0)
        reward_labels = np.delete(reward_labels, bad_trials_idx, axis=0)

    return video_start_trials, reward_labels, video_data


def get_t_slice_video(start_trials, t_idx, video_array, neural_indexes, reward_labels):
    t_trials = start_trials + t_idx
    # indices in neural_indexes whose values appear in t_trials
    t_video = np.flatnonzero(np.isin(neural_indexes, t_trials))
    # take video frames that correspond to neural frames in trial range (t_idx point)
    video_data = video_array[t_video]
    # indices in t_trials whose values appear in neural_indexes
    idx_trial_with_video_t_frame = np.flatnonzero(np.isin(t_trials, neural_indexes))
    # remove trials labels that do not have video frame correspond to t point
    labels_data = reward_labels[idx_trial_with_video_t_frame]

    return video_data, labels_data


def load_clean_align_data(subject_id, session, num_features, frame_rate, time_bin: tuple, original_video_path: str , dj_info: dict, saveroot, logger,
                     handle_omission: str ='convert', clean_ignore=True):
    """
    A multi alignment function that gets subject id and session and align all datasets (video, neural activity, reward size labels) that goes into the reward decoder.
    handle_omission: user can choose to throw out omission trials ('clean'), convert label to regular trial ('convert') or keep omission trials ('keep') in case of multi classification. clean and convert options are for binary classification.
    clean_ignore: if true, ignore trials are thrown out.
    bad trials and grooming trials are automatically thrown out
    time_bin: global time period for all trials to be aligned to according to first lick ('trial_start', 'trial_end')
    frame rate: neural frame rate
    num_features: number of first principal components to consider as video predictors of reward trial type
    """

    # Connect to Datajoint
    try:
        # Try to connect; will raise on failure
        dj.config['database.host'] = dj_info['host_path']
        dj.config['database.user'] = dj_info['user_name']
        dj.config['database.password'] = dj_info['password']
        conn = dj.conn()
    except Exception as e:
        raise ValueError(f"DataJoint connection failed with provided dj_info: {e}, "
                         f"please check if credentials are correct.")

    img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
    tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')
    exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')
    video_neural = dj.VirtualModule('VIDEONEURAL', "lab_videoNeuralAlignment")
    dj_modules = {'img': img, 'tracking': tracking, 'exp2': exp2, 'video_neural': video_neural}

    assert handle_omission in ['keep', 'clean', 'convert'], ('Invalid handle_omission, '
                                                             'value should be "keep" or "clean" or "convert"')

    clean_omission = True if handle_omission == 'clean' else False
    logger.debug('start trial alignment')
    # get trial new start and end frames - global alignment of trials to the first lick after go cue
    #start_trials, end_trials, _, _ = align_trials_and_get_lickrate(subject_id, session, frame_rate, time_bin,
                                                                   #dj_modules, clean_ignore, clean_omission, flag_electric_video=True)
    logger.info('finish trial alignment')
    # get reward size labels
    reward_labels = get_reward_size_labels(subject_id, session, dj_modules, handle_omission, clean_ignore)

    #assert len(reward_labels) == len(start_trials), ('Lengths of reward_labels and start_trials do not match, '
                                                  #'please compare the lengths of original datasets in datajoint')

    svd_path = os.path.join(saveroot, 'video_svd', f'{subject_id}', f'session{session}', f'v_temporal_dynamics_2cameras.npy')
    #svd_path = os.path.join('C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/my_run', 'video_svd', f'{subject_id}', f'session{session}', f'v_temporal_dynamics_2cameras.npy')
    neural_indexes_path = os.path.join(saveroot, 'downsampled_n_v_data', f'{subject_id}', f'session{session}', f'neural_indexes.npy')
    #neural_indexes_path = os.path.join('C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/my_run', 'downsampled_n_v_data', f'{subject_id}', f'session{session}', f'neural_indexes.npy')

    # check if neural indexes and video svd already exist
    #if os.path.exists(svd_path) and os.path.exists(neural_indexes_path):
    if os.path.exists(svd_path):
        #neural_indexes = np.load(neural_indexes_path)
        video_features = np.load(svd_path)[:, : num_features]
        logger.info('video svd and neural_indexes already exist -> finish loading svd')

    else:
        logger.debug('start video downsample and alignment')
        logger.info('start video downsample and alignment')
        # get aligned and downsampled video and neural frames indexes
        # Initialize videos
        video0 = Video(subject_id, session, camera_num=0, video_path=None)
        video1 = Video(subject_id, session, camera_num=1, video_path=None)
        # Align neural data to video data and downsample video
        #video0_array, neural_indexes = video0.align_with_neural_data(dj_modules, original_video_path, clean_ignore, clean_omission, save_root=saveroot, compute_neural_data=False)
        # once you computed neural array for one camera you dont need to repeat for the second
        #video1_array, neural_indexes = video1.align_with_neural_data(dj_modules, original_video_path, clean_ignore, clean_omission, save_root=saveroot, compute_neural_data=False)

        video0.custom_video_downsampling(frame_rate, dj_modules, original_video_path, clean_ignore, clean_omission, save_root=saveroot)
        video1.custom_video_downsampling(frame_rate, dj_modules, original_video_path, clean_ignore, clean_omission, save_root=saveroot)

        logger.info('finish video downsample and alignment')

        logger.debug('start video svd')
        # get video features - compute svd for combined cameras
        pair = VideoPair(subject_id, session, video0, video1)
        video_features = pair.compute_svd(frame_rate, saveroot)[:, : num_features]  # shape: (frames, features) num all session frames x 200 first pc's
        logger.info('finish video svd')

    video_start_trials, video_og_start_trials = align_video_trials(subject_id, session, frame_rate, time_bin, dj_modules, clean_ignore=True, clean_omission=False)
    # keep trials that are padded from both sides with at least one regular trial
    # eliminate large trials that succeed large trials from video data
    video_start_trials, reward_labels, video_features = keep_pure_trials(video_start_trials, video_og_start_trials, reward_labels, video_features, step=2, preserve_large=False)

    #return start_trials, reward_labels, neural_indexes, video_features
    return video_start_trials, reward_labels, video_features






if __name__ == '__main__':
    """
    running example of load_clean_align_data
    """
    subject_id = 464724
    session = 1
    num_features = 50
    time_bin = (-2, 5)
    frame_rate = 2
    host = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
    user = 'ShaniE'
    password = 'opala'
    dj_info = {'host_path': host, 'user_name': user, 'password': password}
    start_trials, reward_labels, neural_indexes, video_features = load_clean_align_data(subject_id,
                                                                                        session,
                                                                                        num_features,
                                                                                        frame_rate,
                                                                                        time_bin,
                                                                                        dj_info,
                                                                                        saveroot=None,
                                                                                        handle_omission='convert',
                                                                                        clean_ignore=True)
    t_idx = 1
    video_data, labels_data = get_t_slice_video(start_trials, t_idx, video_features, neural_indexes, reward_labels)

