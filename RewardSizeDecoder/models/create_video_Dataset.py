import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset


# ---------------- Define split video to sets function -----------------
def split_data(n_frames, start_trials):
    """
    n_frames: N-number of total frames (total length of a session)
    start_trials: (T, 1) size tensor, T-number of trials. contains start indexes of each trial
    out ->
    train, validation, test video tensors

    the function split the dataset into blocks of 10 trials. 8 for train 1 val and 1 test.
    all in all 80% goes to training and 20% for validation and test
    """

    start_trials = np.asarray(start_trials)
    n_trials = len(start_trials)

    # Determine trials end indexes
    ends = np.append(start_trials[1:], [n_frames])

    # Define which trial is train / val / test
    trial_ids = np.arange(n_trials)

    # Identify within each block 10 position
    block_pos = trial_ids % 10

    # boolean masks
    is_train = block_pos < 8
    is_val = block_pos == 8
    is_test = block_pos == 9

    # For all trials selected for training, collect their frame ranges
    def trials_to_frames(mask):
        selected = np.where(mask)[0]
        # build ranges from start_trials[i]..ends[i]
        return np.concatenate([np.arange(int(start_trials[i]), int(ends[i])) for i in selected])

    train_idx = trials_to_frames(is_train)
    val_idx = trials_to_frames(is_val)
    test_idx = trials_to_frames(is_test)

    return train_idx, val_idx, test_idx


# ---------------- Define custom data sets -------------------
class VideoFramesDataset(Dataset):
    def __init__(self, video_path, indices, transform=None):
        """
        frames_np: numpy array of shape [N, H, W, C] or [N, T, H, W, C]
        indices: list/np.array of indices that belong to this split (train/val/test)
        """
        self.video = np.load(video_path, mmap_mode="r")  # mmap once light weight
        self.idx = indices.astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = self.video[self.idx[i]] # numpy slice, load on demand
        # numpy -> torch
        x = torch.from_numpy(x).float()
        # dtype/scale
        x = x.float().div_(255.0)
        if self.transform:
            x = self.transform(x)
        return x