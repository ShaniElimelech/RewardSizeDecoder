import datajoint as dj
import pandas as pd
import numpy as np


def get_behav_epoch_neural_timestamps(key):

    # Get all session timestamps
    session_neural_timestamps_data = (img.FrameTime & key).fetch1()
    session_neural_timestamps = session_neural_timestamps_data["frame_timestamps"][0]

    # Get behav epoch timestamps
    session_epoch_frame_data = (img.SessionEpochFrame & key).fetch1()
    behav_epoch_neural_start_frame = int(session_epoch_frame_data["session_epoch_start_frame"])
    behav_epoch_neural_end_frame = int(session_epoch_frame_data["session_epoch_end_frame"])
    behav_epoch_neural_timestamps = session_neural_timestamps[
                                    behav_epoch_neural_start_frame: behav_epoch_neural_end_frame + 1]

    return behav_epoch_neural_timestamps


def get_trial_neural_timestamps_and_indexes(key, behav_epoch_neural_timestamps):
    # Get trials start and end frames index
    trial_neural_frames_data = (img.FrameStartTrial & key).fetch()
    trials_neural_start_frame = trial_neural_frames_data["session_epoch_trial_start_frame"].astype(int)
    trials_neural_end_frame = trial_neural_frames_data["session_epoch_trial_end_frame"].astype(int)

    # Get trials timestamps (relative to trial start) and indexes (relative to session epoch start)

    trial_neural_timestamps = [
        behav_epoch_neural_timestamps[trials_neural_start_frame[trial]: trials_neural_end_frame[trial] + 1] for trial in
        range(trials_neural_start_frame.shape[0])]
    trial_neural_timestamps_zero = [trial_neural_timestamps[trial] - trial_neural_timestamps[trial][0] for trial in
                                    range(trials_neural_start_frame.shape[0])]
    trial_neural_frames_indexes = [list(range(trials_neural_start_frame[trial], trials_neural_end_frame[trial] + 1)) for
                                   trial in range(trials_neural_start_frame.shape[0])]

    # returns a 2D arrays: list of ndarray of neural timestamps and neural_frames_indexes for each trial
    return trial_neural_timestamps_zero, trial_neural_frames_indexes


def get_trial_timestamps_and_indexes(key, camera_num):
    # Get video timestamps
    trial_video_data = (tracking.TrackingTrial & key & {'tracking_device_id': camera_num}).fetch()
    n = trial_video_data["tracking_num_samples"]
    fps = trial_video_data["tracking_sampling_rate"].astype(float)
    t0 = trial_video_data["tracking_start_time"].astype(float)
    trials_video_timestamps = [
        [t0[trial] + i / fps[trial] for i in range(n[trial])]
        for trial in range(n.shape[0])
    ]
    # returns a 2D array: lists of video timestamps for each trial
    return trials_video_timestamps


def get_grouped_trial_video_timestamps_and_indexes_by_neural_timestamps(trial_video_timestamps,
                                                                        trial_neural_timestamps_zero):
    # Group trial video timestamps and indexes
    trial_video_frames_indexes_groups = []
    trial_video_timestamps_groups = []
    for i in range(len(trial_neural_timestamps_zero) - 1):
        lower, upper = trial_neural_timestamps_zero[i], trial_neural_timestamps_zero[i + 1]
        # Get values from list_a that fall within (lower, upper)
        timestamps_group = [x for x in trial_video_timestamps if lower <= x < upper]
        indexes_group = [idx for idx, val in enumerate(trial_video_timestamps) if lower <= val < upper]
        if len(timestamps_group) > 0:
            timestamps_group_edges = [timestamps_group[0], timestamps_group[-1]]
            indexes_group_edges = [indexes_group[0], indexes_group[-1]]
        else:
            timestamps_group_edges = []
            indexes_group_edges = []
        trial_video_timestamps_groups.append(timestamps_group_edges)
        trial_video_frames_indexes_groups.append(indexes_group_edges)

    # Add last group - video timestamps larger than last neural timestamp
    upper = trial_neural_timestamps_zero[-1]
    timestamps_group = [x for x in trial_video_timestamps if upper <= x]
    indexes_group = [idx for idx, val in enumerate(trial_video_timestamps) if upper <= val]
    if len(timestamps_group) > 0:
        timestamps_group_edges = [timestamps_group[0], timestamps_group[-1]]
        indexes_group_edges = [indexes_group[0], indexes_group[-1]]
    else:
        timestamps_group_edges = []
        indexes_group_edges = []
    trial_video_timestamps_groups.append(timestamps_group_edges)
    trial_video_frames_indexes_groups.append(indexes_group_edges)

    return trial_video_timestamps_groups, trial_video_frames_indexes_groups



########################  Create Table NeuralVideoAlignment  #######################

#this script creates and populates the alignment table using lab user that has permissions
#table img.FrameStartTrial is used as a template with its primary keys for the alignment table



# Connect to Datajoint
dj.config['database.host'] = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
dj.config['database.user'] = "lab"
dj.config['database.password'] = "sababalab"
conn = dj.conn()

# Get all relevant schemas
img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')



# Table definition
schema = dj.Schema("lab_videoNeuralAlignment")


@schema
class NeuralVideoAlignment(dj.Computed):
    definition = """
    -> img.FrameStartTrial
 ---
    trial_neural_frames_indexes: longblob              # (frames) Relative to session epoch start
    trial_video_frames_indexes_groups: longblob        # (frames) Relative to trial start
    trial_neural_timestamps: longblob                  # (s) Relative to trial start
    trial_video_timestamps_groups: longblob            # (s) Relative to trial start
    """

    key_source = dj.U('subject_id','session','session_epoch_type','session_epoch_number') & img.FrameStartTrial

    def make(self, key):
        camera_num = 3  # Change this to select the camera number (3 or 4) they should provide the same data unless one camera has no trial video
        trials = (img.FrameStartTrial & key).fetch('trial', order_by='trial')
        behav_epoch_neural_timestamps = get_behav_epoch_neural_timestamps(key)
        trial_neural_timestamps_zero, trial_neural_frames_indexes = get_trial_neural_timestamps_and_indexes(key, behav_epoch_neural_timestamps)
        trial_video_timestamps = get_trial_timestamps_and_indexes(key, camera_num)

        df = pd.DataFrame({
        # include the other PK fields from `key` (subject_id, session, epoch...)
        'subject_id'                 : key['subject_id'],
        'session'                    : key['session'],
        'session_epoch_type'         : key['session_epoch_type'],
        'session_epoch_number'       : key['session_epoch_number'],
        # new key fields
        'trial'                      : trials,
        'trial_neural_frames_indexes'      : trial_neural_frames_indexes,
        'trial_neural_timestamps'          : trial_neural_timestamps_zero,
        'trial_video_timestamps'     : trial_video_timestamps
        })

        df[['trial_video_timestamps_groups', 'trial_video_frames_indexes_groups']] = (df.apply(
            lambda r: get_grouped_trial_video_timestamps_and_indexes_by_neural_timestamps(
                r['trial_video_timestamps'], r['trial_neural_timestamps']),
            axis=1,
            result_type='expand'))

        df = df.drop(columns=['trial_video_timestamps'])

        # Bulk insert: DataJoint accepts list-of-dicts
        self.insert(df.to_dict('records'))



NeuralVideoAlignment.populate(display_progress=True, reserve_jobs=True)



