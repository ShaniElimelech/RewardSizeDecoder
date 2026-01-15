from RewardSizeDecoder.RewardSizeDecoder.data_preprocessing.ExtractVideoNeuralAlignment import get_all_trial_video_frames
import os
import matplotlib.pyplot as plt
import datajoint as dj
import numpy as np



 # Connect to Datajoint
dj.config['database.host'] = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
dj.config['database.user'] = 'ShaniE'
dj.config['database.password'] = 'opala'
conn = dj.conn()
img = dj.VirtualModule('IMG', 'arseny_learning_imaging')
tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')
video_neural = dj.VirtualModule('VIDEONEURAL', "lab_videoNeuralAlignment")
exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')


original_video_path='D:/Arseny_behavior_video'
subject_lst = [464724, 464725, 463189, 463190]
session_lists = [[1, 2, 3, 4, 5, 6], [1, 2, 6, 7, 8, 9], [1, 3, 4, 9], [2, 3, 5, 6, 10]]
camera_num = 0


for i, subject in enumerate(subject_lst):
    session_list = session_lists[i]
    for j, session in enumerate(session_list):
        session_string = f"session{session}"
        key = {'subject_id': subject, 'session': session, 'trial':3, 'tracking_device_id':3}
        video_file_trial_num = int((tracking.TrackingTrial & key).fetch("tracking_datafile_num")[0])
        video_file_name = f"video_cam_{camera_num}_v{video_file_trial_num:03d}.avi"
        trial_video_file_path = os.path.join(original_video_path, f'{subject}', session_string, video_file_name)
        trial_frame_list = get_all_trial_video_frames(trial_video_file_path, video_file_trial_num, camera_num)
        frame = np.array(trial_frame_list[0])
        H, W = frame.shape
        # frame = frame.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))
        # pads = ((0, 0), (10, 10))  # (top,bottom), (left,right)
        # padded = np.pad(frame, pads, mode='edge')
        y0, x0 = H // 10, W // 10
        y0 += y0 % 2
        x0 += x0 % 2
        print(f'y0={y0}, x0={x0}')
        # crop_frame = frame[11:, 6:]
        crop_frame = frame[y0:, x0:]
        H, W = crop_frame.shape
        crop_frame = crop_frame.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
        H, W = crop_frame.shape
        print(f'subject {subject} session {session} frame shape ({H},{W}')

        plt.figure(figsize=(6, 6))  # new figure each time
        plt.imshow(frame, cmap="gray", aspect="equal")
        plt.title(f"Cropped frame — subject {subject}, session {session}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
