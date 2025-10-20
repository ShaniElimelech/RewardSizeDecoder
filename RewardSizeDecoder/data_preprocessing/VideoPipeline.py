import cv2   ### please install opencv-python package with pip and not conda otherwise you will have plugins problems
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats as sc
import pickle
import datajoint as dj
import pandas as pd
from .ExtractVideoNeuralAlignment import get_session_trials_aligned_frames, get_trials_data_table_for_mouse_session, get_all_trial_video_frames
from scipy.ndimage import gaussian_filter1d
from .block_svd import svd_in_blocks


class Video:
    def __init__(self, subject_id, session, camera_num, video_path=None):
        self.subject_id = subject_id
        self.session = session
        self.camera_num = camera_num
        self.loaded = False
        self.video_array = None
        self.video_path = video_path

    def open_video(self):
        """
        This function opens the video file and returns it as a 3D numpy array  with shape (frames, height, width)
        """
        for attempt in range(3):
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                break
            else:
                print(f"Attempt {attempt + 1} failed to open video.")
                cap.release()
                time.sleep(1)
        else:
            raise RuntimeError(f'Error: Could not open video after 3 attempts at {self.video_path}')

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()

        self.video_array = np.array(frames)
        self.loaded = True
        return self.video_array

    def get_flattened_video(self, rotate=False):
        if not self.loaded:
            self.open_video()

        frames = self.video_array
        if rotate:
            frames = np.rot90(frames, k=1, axes=(1, 2))

        shape = frames.shape
        flattened = np.reshape(frames, (shape[0], -1), order='C')
        return flattened, shape

    def save_as_video(self, savepath, framerate=2):
        """
        Save a 3D NumPy array as a video file.

        Parameters:
            array (np.ndarray): The 3D NumPy array with dimensions (time, X, Y).
            output_file (str): The name of the output video file.
            fps (int, optional): Frames per second (default is 30).
            codec (str, optional): FourCC codec identifier (default is 'mp4v').
        """
        assert self.loaded, 'Video not loaded so it cannot be saved.'

        # Get the dimensions of the array
        codec = 'XVID'
        num_frames, x_dim, y_dim = self.video_array.shape

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(savepath, fourcc, framerate, (y_dim, x_dim))

        # Loop through the frames and write each frame to the video
        for frame in self.video_array.astype(np.uint8):
            # Ensure the frame is in the correct format (e.g., 8-bit BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Write the frame to the video
            out.write(frame)
        # Release the VideoWriter to save the video file
        out.release()

    def align_with_neural_data(self, dj_modules, original_video_path, clean_ignore=False, clean_omission=False,
                               save_root=None, compute_neural_data=True):
        """
        Aligns video data with neural data on a per-trial basis, averages video frames per neural frame,
        and optionally saves both aligned video and neural data arrays.
        Parameters:
            subject_id (str): Subject ID.
            session (int): Session number.
            camera_num (int): Camera number.
            dj_modules (module): DataJoint module for accessing trials and frames.
            clean_ignore (bool): Skip trials marked with ignore flag.
            clean_omission (bool): Skip omissions trials.
            save_root (str): Root path for saving outputs. Defaults to self.savepath if None.
            compute_neural_data (bool): Whether to process and save neural data. Default is True.
        Saves:
            - Downsampled video aligned to neural activity.
            - Neural data (if compute_neural_data is True).
        """

        df_ndata = pd.DataFrame()
        all_video_session = []
        neural_indexes = []

        for trial_index, trial_data in get_session_trials_aligned_frames(
                self.subject_id, self.session, self.camera_num, dj_modules, original_video_path, clean_ignore,
                clean_omission, compute_neural_data):

            if compute_neural_data:
                df_ndata = pd.concat([df_ndata, trial_data['trial_neural_frames']],
                                     ignore_index=True, axis=1)

            trial_video_frames = [
                np.rint(np.array(frames).mean(axis=0)).astype(int) for frames in trial_data['trial_video_frames_groups']]
            all_video_session.extend(trial_video_frames)
            neural_indexes.extend(trial_data['trial_neural_frames_indexes_with_video'])

        short_vdata = np.array(all_video_session)
        self.video_array = short_vdata
        self.loaded = True

        if save_root:
            save_dir = os.path.join(save_root, 'downsampled_n_v_data', f'{self.subject_id}', f'session{self.session}')
            os.makedirs(save_dir, exist_ok=True)
            video_save_path = os.path.join(save_dir, f'downsampled_video_cam{self.camera_num}.avi')
            self.save_as_video(video_save_path)
            np.save(os.path.join(save_dir, 'neural_indexes.npy'), neural_indexes)

        if compute_neural_data:
            short_ndata = df_ndata.to_numpy()
            if save_root:
                np.save(os.path.join(save_dir, 'short_ndata.npy'), short_ndata)
            return short_vdata, neural_indexes, short_ndata

        return short_vdata, neural_indexes

    @ staticmethod
    def downsample_by_block_average(video):
        """
        Downsample a grayscale video (T, H, W) spatially by factor of 2 using block averaging.
        Automatically crops frames if dimensions are odd.
        """
        T, H, W = video.shape

        # Ensure dimensions are even (auto-crop if needed)
        if H % 2 != 0:
            H -= 1  # crop last row
            video = video[:, :H, :]
        if W % 2 != 0:
            W -= 1  # crop last column
            video = video[:, :, :W]

        # Reshape and block-average
        downsampled = video.reshape(T, H // 2, 2, W // 2, 2).mean(axis=(2, 4))

        # Preserve original dtype (e.g. uint8)
        return np.rint(downsampled).astype(video.dtype)

    @staticmethod
    def crop_frames(video):
        """
        crop the top (two photon imaging) and left side (lick port) of the image
        """
        f, H, W = video.shape
        y0, x0 = int(H // 10), int(W // 10)
        crop_video = video[:, y0:, x0:]

        return crop_video

    def custom_video_downsampling(self, frame_rate, dj_modules, original_video_path, clean_ignore=False, clean_omission=False, save_root=None):
        """
        use when you want to downsample the video data to a different frame rate than neural data.
       """
        if save_root:
            save_dir = os.path.join(save_root, 'downsampled_n_v_data', f'{self.subject_id}', f'session{self.session}')
            if os.path.exists(os.path.join(save_dir, f'downsampled_video_cam{self.camera_num}.npy')):
                self.video_array = np.load(os.path.join(save_dir, f'downsampled_video_cam{self.camera_num}.npy'))
                self.loaded = True
                return

        all_video_session = []
        video_neural = dj_modules['video_neural']
        tracking = dj_modules['tracking']
        exp2 = dj_modules['exp2']

        session_string = f"session{self.session}"
        all_videos_path = original_video_path
        if self.camera_num not in [0, 1]:
            raise ValueError("Camera number must be 0 or 1!")

        # Get data from DataJoint
        trials_data = get_trials_data_table_for_mouse_session(self.subject_id, self.session, self.camera_num, tracking, video_neural,
                                                              exp2, clean_ignore, clean_omission)
        if trials_data.empty:
            raise ValueError(f'There is no neural data for subject{subject_id} session{session}')

        for index, row in trials_data.iterrows():
            video_file_trial_num = row["tracking_datafile_num"]
            video_file_name = f"video_cam_{self.camera_num}_v{video_file_trial_num:03d}.avi"
            trial_video_file_path = os.path.join(all_videos_path, f'{self.subject_id}', session_string, video_file_name)
            trial_frame_list = get_all_trial_video_frames(trial_video_file_path, video_file_trial_num, self.camera_num)
            frames = np.array(trial_frame_list)
            i = 0
            jump = int(250 // frame_rate)
            while i < len(trial_frame_list):
                if i+jump < len(trial_frame_list):
                    ave_frame = np.rint(frames[i:i + jump].mean(axis=0)).astype(int)
                else:
                    ave_frame = np.rint(frames[i:].mean(axis=0)).astype(int)
                all_video_session.append(ave_frame)
                i = i + jump

        short_vdata = np.array(all_video_session)
        if self.camera_num == 0:  # crop only the side camera
            short_vdata = self.crop_frames(short_vdata)
        short_vdata = self.downsample_by_block_average(short_vdata)
        if save_root:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f'downsampled_video_cam{self.camera_num}.npy'), short_vdata)

        self.video_array = short_vdata
        self.loaded = True

    def gaussian_smooth_and_resample(self, frame_rate, dj_modules, original_video_path, clean_ignore=False, clean_omission=False):
        """
        smooth video data using gaussian and then resample to a frame rate of your choice.
        """
        all_video_session = []
        video_neural = dj_modules['video_neural']
        tracking = dj_modules['tracking']
        exp2 = dj_modules['exp2']

        session_string = f"session{self.session}"
        all_videos_path = original_video_path
        if self.camera_num not in [0, 1]:
            raise ValueError("Camera number must be 0 or 1!")

        # Get data from DataJoint
        trials_data = get_trials_data_table_for_mouse_session(self.subject_id, self.session, self.camera_num, tracking, video_neural, exp2, clean_ignore, clean_omission)
        if trials_data.empty:
            raise ValueError(f'There is no neural data for subject{subject_id} session{session}')

        for index, row in trials_data.iterrows():
            video_file_trial_num = row["tracking_datafile_num"]
            video_file_name = f"video_cam_{self.camera_num}_v{video_file_trial_num:03d}.avi"
            trial_video_file_path = os.path.join(all_videos_path, f'{self.subject_id}', session_string, video_file_name)
            trial_frame_list = np.array(get_all_trial_video_frames(trial_video_file_path, video_file_trial_num, self.camera_num))
            original_video_rate = 250
            sigma = 0.7 * (original_video_rate / frame_rate)
            smoothed_video = gaussian_filter1d(trial_frame_list, sigma=sigma, axis=0, mode='mirror', truncate=4.0)
            step = 250// frame_rate
            smooth_video = smoothed_video[::step]
            all_video_session.extend(smooth_video)

        short_vdata = np.array(all_video_session)
        self.video_array = short_vdata
        self.loaded = True



class VideoPair:
    def __init__(self, subject_id, session, video0: Video, video1: Video = None):
        """
        Args:
            video0: video type object from camera 0
            video1: video type object from camera 1, if None then svd is computed only on video 0
        """
        self.video0 = video0
        self.video1 = video1
        self.two_cams = video1 is not None
        self.subject_id = subject_id
        self.session = session
        self.shapes = {}

    def concatenate_flattened(self):
        flat0, shape0 = self.video0.get_flattened_video()
        self.shapes['cam0'] = shape0

        if self.two_cams:
            flat1, shape1 = self.video1.get_flattened_video(rotate=True)
            self.shapes['cam1'] = shape1
            return np.concatenate((flat0, flat1), axis=1)
        return flat0

    def compute_svd(self, frame_rate, save_root=None):
        video_matrix = self.concatenate_flattened()

        '''
        # Remove zero-variance pixels
        std = np.std(video_matrix, axis=0)
        mask = std >= 1e-4

        normalized = np.zeros_like(video_matrix)
        normalized[:, mask] = sc.zscore(video_matrix[:, mask], axis=0)
        '''

        X = np.asarray(video_matrix, dtype=np.float32, order="C")

        std = X.std(axis=0)
        mask = std >= 1e-4
        cols = np.where(mask)[0]

        # Normalize in-place, columnwise, in small blocks to cap memory
        block = 2048
        eps = 1e-12
        for i in range(0, len(cols), block):
            c = cols[i:i + block]
            m = X[:, c].mean(axis=0, dtype=np.float32)
            s = X[:, c].std(axis=0, dtype=np.float32)
            s = np.maximum(s, eps)  # avoid div-by-zero
            X[:, c] -= m  # in-place center
            X[:, c] /= s  # in-place scale

        X[:, ~mask] = 0.0
        normalized = X  # already normalized

        #centered_video = np.zeros_like(video_matrix)
        #centered_video[:, mask] = np.mean(video_matrix[:, mask], axis=0)


        U, S, VT = np.linalg.svd(normalized, full_matrices=False)
        #explained_var = (S ** 2) / np.sum(S ** 2)
        #num_components = np.argmax(np.cumsum(explained_var) >= 0.9) + 1

        '''
        n_segments = int(frame_rate //2)
        # get U @ S from svd in blocks algorithm
        U, VT = svd_in_blocks(
            video_matrix,
            n_segments,
            P=video_matrix.shape[1],
            k_seg=300,
            k_global=500,
            dtype=np.float32,
        )
        '''

        if save_root:
            save_dir = os.path.join(save_root, 'video_svd', f'{self.subject_id}', f'session{self.session}')
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, f'OG_shape_{int(self.two_cams) + 1}cameras.pkl'), 'wb') as f:
                pickle.dump(self.shapes, f)

            #np.save(os.path.join(save_dir, f'num_components_0.9_{int(self.two_cams) + 1}cameras'), num_components)
            #np.save(os.path.join(save_dir, f'v_singular_values_{int(self.two_cams) + 1}cameras'), S[:500])
            np.save(os.path.join(save_dir, f'v_spatial_dynamics_{int(self.two_cams) + 1}cameras'), VT[:500])
            np.save(os.path.join(save_dir, f'v_temporal_dynamics_{int(self.two_cams) + 1}cameras'), U[:, :500])

            self.plot_principal_components(VT, save_dir)
            #self.plot_variance(explained_var, save_dir)

        return U

    def plot_principal_components(self, VT, save_dir):
        """
        Visualizes spatial weights of selected principal components (PCs)
        from the SVD decomposition for each camera.
        """
        cutoff = 0
        for cam_name, shape in self.shapes.items():
            pixels = shape[1] * shape[2]
            height, width = shape[1], shape[2]
            pc_weights = np.reshape(VT[:, cutoff:cutoff+pixels], (VT.shape[0], height, width), order='C')
            cutoff += pixels

            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            pcs = [0, 1, 2, 3, 4, 5]
            for ax, pc in zip(axes.flatten(), pcs):
                im = ax.imshow(np.abs(pc_weights[pc]), cmap='inferno')
                ax.set_title(f'Principal Component no.{pc}')
                fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{cam_name}_pcs_weights_1.png'))
            plt.close()
            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            pcs = [6, 7, 8, 9, 10, 20]
            for ax, pc in zip(axes.flatten(), pcs):
                im = ax.imshow(np.abs(pc_weights[pc]), cmap='inferno')
                ax.set_title(f'Principal Component no.{pc}')
                fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{cam_name}_pcs_weights_2.png'))
            plt.close()
            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            pcs = [100, 150, 200, 250, 300, 500]
            for ax, pc in zip(axes.flatten(), pcs):
                im = ax.imshow(np.abs(pc_weights[pc]), cmap='inferno')
                ax.set_title(f'Principal Component no.{pc}')
                fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{cam_name}_pcs_weights_3.png'))
            plt.close()


    def plot_variance(self, explained_var, save_dir):
        # plot variance explained
        plt.figure()
        plt.plot(explained_var)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir, f'{int(self.two_cams) + 1}cameras_ve.png'))
        plt.close()

        # plot cumulative variance explained
        # Compute cumulative sum
        cumulative = np.cumsum(explained_var)
        plt.plot(cumulative)
        plt.title("Cumulative Sum of variance explained")
        plt.xlabel("pc number")
        plt.ylabel("Cumulative VE")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{int(self.two_cams) + 1}cameras_cumulative_ve.png'))
        plt.close()


if __name__ == '__main__':
    """
    running example of video pipeline
    """

    video_path = "/your/video/root"  # only in case you already computed downsampled video
    saveroot = 'G:/Shared drives/FinkelsteinLab/People/ShaniElimelech/neuronal drift/results'
    subject_id = 464724
    session = 1

    # Initialize videos
    video0 = Video(subject_id, session, camera_num=0, video_path=None)
    video1 = Video(subject_id, session, camera_num=1, video_path=None)

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

    # Align neural data to video data and downsample video
    original_video_path = 'put here your path to original video folder'
    video0_array, neural_array = video0.align_with_neural_data(dj_modules, original_video_path, clean_ignore=False, clean_omission=False,
        save_root=saveroot, compute_neural_data=True)
    # once you computed neural array for one camera you dont need to repeat for the second
    video1_array = video1.align_with_neural_data(dj_modules, original_video_path, clean_ignore=False, clean_omission=False,
                                                              save_root=saveroot, compute_neural_data=False)

    # compute svd of two cameras
    pair = VideoPair(subject_id, session, video0, video1)
    U = pair.compute_svd(saveroot)







