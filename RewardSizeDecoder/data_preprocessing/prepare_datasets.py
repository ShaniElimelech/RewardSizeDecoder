from .trial_alignment import align_trials_and_get_lickrate, get_reward_size_labels
from .VideoPipeline import Video, VideoPair
import numpy as np
import datajoint as dj
import logging


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


def load_clean_align_data(subject_id, session, num_features, frame_rate, time_bin: tuple, dj_info: dict, saveroot, logger,
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
    start_trials, end_trials, _, _ = align_trials_and_get_lickrate(subject_id, session, frame_rate, time_bin,
                                                                   dj_modules, clean_ignore, clean_omission,
                                                                   flag_electric_video=True)
    logger.info('finish trial alignment')
    # get reward size labels
    reward_labels = get_reward_size_labels(subject_id, session, dj_modules, handle_omission, clean_ignore)

    assert len(reward_labels) == len(start_trials), ('Lengths of reward_labels and start_trials do not match, '
                                                  'please compare the lengths of original datasets in datajoint')

    logger.debug('start video downsample and alignment')
    # get aligned and downsampled video and neural frames indexes
    # Initialize videos
    video0 = Video(subject_id, session, camera_num=0, video_path=None)
    video1 = Video(subject_id, session, camera_num=1, video_path=None)
    # Align neural data to video data and downsample video
    video0_array, neural_indexes = video0.align_with_neural_data(dj_modules, clean_ignore, clean_omission,
                                                               save_root=saveroot, compute_neural_data=False)
    # once you computed neural array for one camera you dont need to repeat for the second
    video1_array, neural_indexes = video1.align_with_neural_data(dj_modules, clean_ignore, clean_omission,
                                                 save_root=saveroot, compute_neural_data=False)

    logger.info('finish video downsample and alignment')
    logger.debug('start video svd')
    # get video features - compute svd for combined cameras
    pair = VideoPair(subject_id, session, video0, video1)
    video_features = pair.compute_svd(saveroot)[:, : num_features]  # shape: (frames, features) num all session frames x 50 first pc's
    logger.info('finish video svd')

    return start_trials, reward_labels, neural_indexes, video_features


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

