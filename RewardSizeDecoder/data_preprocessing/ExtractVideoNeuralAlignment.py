import pandas as pd
import datajoint as dj
import cv2   ### please install opencv-python package with pip and not conda otherwise you will have plugins problems
import os
import numpy as np
import time
import matplotlib.pyplot as plt


def get_all_trial_video_frames(trial_video_file_path, video_file_trial_num, camera_num):
    #print("strting get_all_trial_video_frames")
    for attempt in range(3):
        cap = cv2.VideoCapture(trial_video_file_path)
        if cap.isOpened():
            break
        else:
            print(f"Attempt {attempt + 1} failed to open video.")
            cap.release()
            time.sleep(1)
    else:
        raise RuntimeError(f'Could not open video after 3 attempts - video file trial num:{video_file_trial_num}, camera:{camera_num}, file_video_path:{trial_video_file_path}-'
                           f' please check if video file trial num exist, if it does then check opencv package')

    frames_lst = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = np.squeeze(np.mean(frame, axis=2))   # remove color channels
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.convertScaleAbs(gray_frame)  # make into shorter format
        frames_lst.append(gray_frame)
    cap.release()

    return frames_lst


def aligned_trial_video_frames(trial_video_frames_indexes, frames_list):
    #print("strting aligned_trial_video_frames")
    aligned_trial_video_frames_ = []
    for index_range in trial_video_frames_indexes:
        if index_range == []:
            continue
        if index_range[0] < len(frames_list) <= index_range[1] + 1:
            aligned_trial_video_frames_.append(frames_list[index_range[0]:])
            break
        aligned_trial_video_frames_.append(frames_list[index_range[0]:index_range[1] + 1])

    return aligned_trial_video_frames_


def get_trials_data_table_for_mouse_session(subject_id, session, camera_num, tracking, video_neural, exp2, clean_ignore, clean_omission):
    #print("start get_trials_data_table_for_mouse_session")
    # This is due to a discrepancy between the camera numbers in the video files and the camera numbers in Datajoint
    if camera_num == 0:
        camera_num = 3
    if camera_num == 1:
        camera_num = 4

    key = {'subject_id': subject_id, 'session': session}
    tracking_trials = tracking.TrackingTrial
    video_neural_alignment_table = video_neural.NeuralVideoAlignment
    # Base exclusion - removing bad trials and grooming trials
    restricted_table = (tracking_trials * video_neural_alignment_table) - tracking.TrackingTrialBad - tracking.VideoGroomingTrial
    # Conditional exclusions - ignore and omission trials
    if clean_ignore:
        restricted_table -= (exp2.BehaviorTrial & 'outcome="ignore"')
    if clean_omission:
        restricted_table -= (exp2.TrialRewardSize & 'reward_size_type="omission"')

    return pd.DataFrame((restricted_table & key & {'tracking_device_id': 3}).fetch())


def get_trial_video_frames_groups(row, all_videos_path, subject_id, session_string, camera_num):
    #print("strting get_trial_video_frames_groups")
    video_file_trial_num = row["tracking_datafile_num"]
    video_file_name = f"video_cam_{camera_num}_v{video_file_trial_num:03d}.avi"
    trial_video_file_path = os.path.join(all_videos_path, f'{subject_id}', session_string, video_file_name)
    trial_frame_list = get_all_trial_video_frames(trial_video_file_path, video_file_trial_num, camera_num)
    trial_video_frames_indexes = row["trial_video_frames_indexes_groups"]
    return aligned_trial_video_frames(trial_video_frames_indexes, trial_frame_list)


def get_dff_table_for_mouse_session(subject_id, session, img):
    #print("strting get_dff_table_for_mouse_session")
    key = {"subject_id": subject_id, "session": session, "session_epoch_type": "behav_only"}
    ROIdeltaF = pd.DataFrame(((img.ROIdeltaF & key)).fetch())
    dff_trace_matrix = pd.DataFrame([x[0] for x in ROIdeltaF["dff_trace"]])
    df_trace = pd.concat([ROIdeltaF.drop(["dff_trace", "session_epoch_type", "subject_id", "session"], axis='columns'), dff_trace_matrix], axis=1)
    return df_trace


def get_trial_neural_frames(dff_data, trial_neural_frames_indexes, trial_video_length, drop_frames_with_no_video=True):
    #print("strting get_trial_neural_frames")
    if drop_frames_with_no_video:
        if trial_video_length == None:
            raise Exception("Can't drop neural frames with no data if video length is not provided")
        trial_neural_frames_indexes = trial_neural_frames_indexes[:trial_video_length]
        neural_frames = dff_data[trial_neural_frames_indexes]
    else:
        neural_frames = dff_data[trial_neural_frames_indexes]
    
    return neural_frames


def get_session_trials_aligned_frames(subject_id, session, camera_num, dj_modules, original_video_path, clean_ignore, clean_omission, compute_neural_data=True,
                                      take_only_first_video_frame=False, drop_neural_frames_with_no_video=True):
    """
    This function yields a dictionary with the neural frames and aligned video frames for each trial
    take_only_first_video_frame: You can set this True if you want to take only the first video frame for each neural frame
    drop_neural_frames_with_no_video: You can set this False if you want to keep neural frames that have no video frames associated with them
    camera_num: The camera number you want to use. has to be 0 or 1!!!. Note that cameras may differ in the number of frames they recorded.
    """
    video_neural = dj_modules['video_neural']
    img = dj_modules['img']
    tracking = dj_modules['tracking']
    exp2 = dj_modules['exp2']

    session_string = f"session{session}"
    all_videos_path = original_video_path
    if camera_num not in [0, 1]:
        raise ValueError("Camera number must be 0 or 1!")
    
    # Get data from DataJoint
    trials_data = get_trials_data_table_for_mouse_session(subject_id, session, camera_num, tracking, video_neural, exp2, clean_ignore, clean_omission)
    if trials_data.empty:
        raise ValueError(f'There is no neural data for subject{subject_id} session{session}')

    dff_data = None
    if compute_neural_data:
        dff_data = get_dff_table_for_mouse_session(subject_id, session, img)
        if dff_data.empty:
            raise ValueError(f'There is no neural data for subject{subject_id} session{session}')

    for index, row in trials_data.iterrows():
        trial_video_frames = get_trial_video_frames_groups(row, all_videos_path, subject_id, session_string, camera_num)
        trial_neural_frames_indexes_with_video = row["trial_neural_frames_indexes"][:len(trial_video_frames)]

        if compute_neural_data:
            trial_neural_frames = get_trial_neural_frames(dff_data, row["trial_neural_frames_indexes"], len(trial_video_frames), drop_neural_frames_with_no_video)
        else:
            trial_neural_frames = None

        if take_only_first_video_frame:
            trial_video_frames = [video_frames_group[0] for video_frames_group in trial_video_frames]

        new_row = {'trial_neural_frames': trial_neural_frames, 'trial_video_frames_groups': trial_video_frames,
                   'trial_neural_frames_indexes_with_video': trial_neural_frames_indexes_with_video}

        yield index, new_row


def save_as_video(array, savepath, framerate=2):
    """
    Save a 3D NumPy array as a video file.

    Parameters:
        array (np.ndarray): The 3D NumPy array with dimensions (time, X, Y).
        output_file (str): The name of the output video file.
        fps (int, optional): Frames per second (default is 30).
        codec (str, optional): FourCC codec identifier (default is 'mp4v').
    """
    # Get the dimensions of the array
    codec = 'XVID'
    num_frames, x_dim, y_dim = array.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(savepath, fourcc, framerate, (y_dim, x_dim))

    # Loop through the frames and write each frame to the video
    for frame in array:
        # Ensure the frame is in the correct format (e.g., 8-bit BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Write the frame to the video
        out.write(frame)

    # Release the VideoWriter to save the video file
    out.release()


if __name__ == '__main__':
    # Connect to Datajoint
    dj.config['database.host'] = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
    dj.config['database.user'] = 'ShaniE'
    dj.config['database.password'] = 'opala'
    conn = dj.conn()
    img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
    tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')
    video_neural = dj.VirtualModule('VIDEONEURAL', "lab_videoNeuralAlignment")
    exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')
    dj_modules = {'img': img, 'tracking': tracking, 'exp2': exp2, 'video_neural': video_neural}


    # iterate over all subjects and sessions
    subject_lst = [464724, 464725, 463189, 463190]
    session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 5, 6, 7, 8, 9], [1, 2, 3, 4, 9], [2, 3, 4, 5, 6, 10]]
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
                if length > 9:
                    print(f'subject{subject_id}, session{session}, trial{trail_index} has {length} '
                          f'frames without video and {tot_frames} total frames.')
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



    #
    # savepath = 'H:\\Shared drives\\FinkelsteinLab\\People\\ShaniElimelech\\neuronal drift\\results\\new_downsampled_n_v_data'
    # for i, subject_id in enumerate(subject_lst):
    #     sessions = session_lists[i]
    #     for session in sessions:
    #         print(f'subject:{subject_id}, session:{session} started')
    #         save_session_path = os.path.join(savepath, f'{subject_id}', f'session{session}')
    #         try:
    #             for camera_num in [0, 1]:  # align data for each camera
    #                 # generate for all trials aligned video and neural data and assemble it into a full session
    #                 df_ndata = pd.DataFrame()
    #                 all_video_session = []
    #                 nanvalues = {}
    #
    #                 for trail_index, trial_data in get_session_trials_aligned_frames(subject_id, session,
    #                                      camera_num, dj_modules, clean_ignore=False, clean_omission=False):
    #                     df_ndata = pd.concat([df_ndata,trial_data['trial_neural_frames']], ignore_index=True, axis=1)
    #                     trial_video_frames = [np.array(frames).mean(axis=0) for frames in trial_data['trial_video_frames_groups']]
    #                     all_video_session.extend(trial_video_frames)
    #
    #                 short_ndata = df_ndata.to_numpy()  # convert df to npy
    #                 short_vdata = np.array(all_video_session)  # convert df to npy
    #                 os.makedirs(save_session_path, exist_ok=True)
    #                 save_as_video(short_vdata.astype(np.uint8),savepath=f'{save_session_path}\downsampled_video_cam{camera_num}.avi')
    #                 np.save(f'{save_session_path}\short_ndata.npy', short_ndata)
    #
    #         except (ValueError, RuntimeError, Exception) as e:
    #                 print(f"Handled error: {e}\nskipping subject{subject_id} session{session}")
    #                 continue


