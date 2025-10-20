import os
import numpy as np
import datajoint as dj
from scipy import stats as sc
from VideoPipeline import Video, VideoPair
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, LeaveOneOut, cross_val_predict
from sklearn.metrics import recall_score, confusion_matrix
from trial_alignment import align_trials_and_get_lickrate
from ExtractVideoNeuralAlignment import get_trials_data_table_for_mouse_session, get_dff_table_for_mouse_session, get_trial_video_frames_groups, get_trial_neural_frames


def ignore_trials_decoder(subject_id, session, num_features, dj_modules, frame_rate, time_bin, saveroot, lick_trials):
    img = dj_modules['img']
    tracking = dj_modules['tracking']
    exp2 = dj_modules['exp2']
    key = {'subject_id': subject_id, 'session': session}
    outcome_table = ((exp2.BehaviorTrial & key) - tracking.TrackingTrialBad - tracking.VideoGroomingTrial).fetch('outcome')
    outcome_labels = (outcome_table == "ignore").astype(int)
    ignore_perc = sum(outcome_labels) / len(outcome_labels)

    start_trials, end_trials, _, _ = align_trials_and_get_lickrate(subject_id, session, frame_rate, time_bin,
                                                                   dj_modules, clean_ignore=False, clean_omission=False,
                                                                   flag_electric_video=True)
    session_string = f"session{session}"
    original_video_path = 'E:/Arseny_behavior_video'
    camera_num = 0
    trials_data = get_trials_data_table_for_mouse_session(subject_id, session, camera_num, tracking, video_neural, exp2,
                                                          clean_ignore=False, clean_omission=False)

    '''
    lick_trial_num = set(lick_trials.keys())
    all_session_video_frames = []
    full_session_labels = []
    for index, row in trials_data.iterrows():
        if str(row['trial']) not in lick_trial_num and int(outcome_labels[index]) == 0:
            continue
        elif int(outcome_labels[index]) == 0:
            print(f'{row['trial']} -> {outcome_labels[index]}')
            trial_video_frames = get_trial_video_frames_groups(row, original_video_path, subject_id, session_string, camera_num)
            trial_neural_frames_indexes_with_video = row["trial_neural_frames_indexes"][:len(trial_video_frames)]
            frames_dict = lick_trials[str(row['trial'])]
            for frame in frames_dict:
                start_time = frames_dict[frame][0]
                end_time = frames_dict[frame][1]
                frame_current = start_trials[index] + 4 + int(frame)
                neural_frame_idx_current = np.where(np.array(trial_neural_frames_indexes_with_video) == frame_current)[0][0]
                frames_lick_video = trial_video_frames[neural_frame_idx_current][start_time:end_time]
                all_session_video_frames.extend(frames_lick_video)
                full_session_labels.extend([0]* (end_time - start_time))

        elif int(outcome_labels[index]) == 1:
            print(f'trial number {row['trial']} -> {outcome_labels[index]}')
            trial_video_frames = get_trial_video_frames_groups(row, original_video_path, subject_id, session_string, camera_num)
            trial_neural_frames_indexes_with_video = row["trial_neural_frames_indexes"][:len(trial_video_frames)]
            num_frames = len(trial_neural_frames_indexes_with_video)
            for frame in range(5, num_frames):
                ignore_frame = trial_video_frames[frame][0]
                all_session_video_frames.append(ignore_frame)
                full_session_labels.append(1)

    video_data = np.stack(all_session_video_frames, axis=0)
    shape = video_data.shape
    flattened = np.reshape(video_data, (shape[0], -1), order='C')
    labels = np.asarray(full_session_labels)
    ignore_per_new = np.average(labels)
    # Remove zero-variance pixels
    std = np.std(flattened, axis=0)
    mask = std >= 1e-4
    normalized = np.zeros_like(flattened)
    normalized[:, mask] = sc.zscore(flattened[:, mask], axis=0)

    #centered_video = np.zeros_like(flattened)
    #centered_video[:, mask] = np.mean(flattened[:, mask], axis=0)

    U, S, VT = np.linalg.svd(normalized, full_matrices=False)
    video_data = U[:, :num_features]
    '''




    # Initialize videos
    video0 = Video(subject_id, session, camera_num=0, video_path=None)
    #video1 = Video(subject_id, session, camera_num=1, video_path=None)

    # Align neural data to video data and downsample video
    original_video_path = 'E:/Arseny_behavior_video'
    video0_array, neural_array = video0.align_with_neural_data(dj_modules, original_video_path, clean_ignore=False,
                                                               clean_omission=False,
                                                               save_root=saveroot, compute_neural_data=False)

    # once you computed neural array for one camera you dont need to repeat for the second
    #video1_array = video1.align_with_neural_data(dj_modules, original_video_path, clean_ignore=False, clean_omission=False, save_root=saveroot, compute_neural_data=False)
   

    # compute svd of two cameras
    pair = VideoPair(subject_id, session, video0)
    video_features = pair.compute_svd(saveroot)[:, :num_features]

    # take non lick frames from ignore trials and lick frame (frame time 1) from lick trials
    full_neural_array = np.array([item for sublist in neural_array for item in sublist])
    time_to_lick = 5  # getting frame 1
    full_session_labels = []
    video_new = []

    for i_trial in range(len(neural_array)):
        if outcome_labels[i_trial]:  # ignore trial -> take few frames - non lick
            num_frames = len(neural_array[i_trial])
            start_trial = neural_array[i_trial][2 * num_frames // 3]
            start_trial_idx = np.where(full_neural_array == start_trial)[0][0]
            end_trial = neural_array[i_trial][num_frames - 1]
            end_trial_idx = np.where(full_neural_array == end_trial)[0][0]
            block = video_features[start_trial_idx:end_trial_idx + 1, :]
            video_new.extend([block[j, :] for j in range(block.shape[0])])
            full_session_labels.extend([1] * block.shape[0])


        else:  # not ignore -> take only frame 1 - lick time
            full_session_labels.append(0)
            frame = start_trials[i_trial] + time_to_lick
            neural_frame_idx = np.where(full_neural_array == frame)[0][0]
            video_frame = video_features[neural_frame_idx, :]
            video_new.append(video_frame)

    video_data = np.stack(video_new, axis=0)
    labels = np.asarray(full_session_labels)


    print('size of new video:', video_features.shape)
    # print('Number of frames in old video:', len(video0_array))
    print('Number of frames in new labels:', len(full_session_labels))
    print('Number of frames in old labels:', len(outcome_labels))
    print(f'number of ignor_trials : {sum(full_session_labels) / len(full_session_labels)}')

    scoring = {
        'accuracy': 'accuracy',
        'recall': 'recall',
        'auc': 'roc_auc',
        'f1': 'f1'
    }
    model = make_pipeline(StandardScaler(), LogisticRegression())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        video_data,
        labels,
        cv=skf,
        scoring=scoring,
        return_train_score=False
    )

    accuracy = scores['test_accuracy']
    recall = scores['test_recall']
    auc = scores['test_auc']
    f1 = scores['test_f1']

    return accuracy, recall, auc, f1, ignore_perc





lick_trials = {'6': {'0':(107,120), '1': (0,20)}, '7': {'1': (22,55)}, '8': {'0': (55,75), '1': (70,85)}, '25': {'0': (91,100), '1': (65,77)}, '26': {'1': (11,44)}, '27': {'0': (66,84), '1': (57,64)}, '33': {'1': (2,20)}
               , '34': {'0': (91,97), '1': (34,44)}, '44': {'0': (87,105), '1': (3,19)}, '45': {'0': (0,22), '1': (60,70)}, '48': {'0': (109,120), '1': (35,50)}, '49': {'0': (78,102), '1': (9,25)}
               , '50': {'0': (44,58), '1': (13,28)}, '54': {'0': (103,120), '1': (55,66)}, '55': {'0': (58,66), '1': (33,40)}, '74': {'0': (63,79), '1': (0,9)}, '75': {'0': (100,120), '1': (25,40)}
                , '87': {'0': (66, 74), '1': (12,22)}, '88': {'1': (25,44)}, '101': {'0': (106,120), '1': (99,108)}, '102': {'0': (80,104), '1': (5,18)}, '103': {'0': (44,58), '1': (0,11)}, '107': {'1': (70,80)}
               , '108': {'0': (50,60), '1': (23,33)}, '117': {'0': (71,81), '1': (67,80)}, '118': {'0': (60,80), '1': (0,5)}, '129': {'0': (60,72), '1': (5,15)}, '130': {'0': (66,78), '1': (0,14)}
               , '132': {'0': (98,110), '1': (11,30)}, '133': {'0': (85,100), '1': (5,23)}, '134': {'0': (85,98), '1': (0,9)}, '136': {'0': (48,66), '1': (13,22)}, '137': {'1': (5,25)}
               , '138': {'0': (40,50), '1': (16,25)}, '139': {'0': (97,112), '1': (28,40)}, '143': {'0': (97,105), '1': (5,18)}, '144': {'0': (42,53), '1': (70,81)}, '146': {'0': (64,77), '1': (22,31)}
               , '147': {'0': (34,46), '1': (45,56)}, '149': {'0': (30,50), '1': (39,50)}, '150': {'1': (33,41)}, '176': {'0': (33,53), '1': (60,72)}, '177': {'1': (0,22)}, '178': {'0': (40,53), '1': (37,45)}
               , '179': {'0': (46,70), '1': (42,51)}}

subject_id = 464724
sessions = [4] #[4, 1, 2, 3, 5, 6]
num_features = 200
frame_rate = 2
time_bin = (-2,5)
saveroot = 'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/ignore_trials- 200 first pcs'
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
for session in sessions:
    accuracy, recall, auc, f1, ignore_perc = ignore_trials_decoder(subject_id, session, num_features, dj_modules, frame_rate,time_bin, saveroot, lick_trials)
    print('------------------------------------------------------------')
    print(f'subject_id: {subject_id} | session: {session} | percentage of ignore trials: {ignore_perc}')
    print(f'accuracy: {accuracy} | recall: {recall} | auc: {auc}')
    print('------------------------------------------------------------')