import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math, random, copy, os, time
import numpy as np
from ConvAutoEncoder import FlexibleEncoder, FlexibleDecoder
from torchvision import transforms
from create_video_Dataset import split_data, VideoFramesDataset
import matplotlib.pyplot as plt
import datajoint as dj
from ..data_preprocessing.VideoPipeline import Video


# -------------------- MODEL BUILDER -------------------------
class ModelBuilder:
    """
    build encoder and decoder models according to your desirable configuration
    output ->
    encoder and decoder objects
    """
    def __init__(self, in_chan, out_chan, latent_num, H, W):
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.H = H
        self.W = W
        self.latent_num = latent_num

    def build(self, cfg:dict):
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        enc = FlexibleEncoder(
            in_chan=self.in_chan,
            num_latent=self.latent_num,
            H_input=self.H,
            W_input=self.W,
            channels=cfg["channels"],
            kernel_size=cfg["kernel"],
            stride=cfg["stride"],
            padding=cfg["padding"],
            use_maxpool=(cfg["stride"] == 1)
        ).to(device)

        dec = FlexibleDecoder(
            out_chan=self.out_chan,
            num_latent=self.latent_num,
            channels=cfg["channels"],
            kernel_size=cfg["kernel"],
            stride=cfg["stride"],
            padding=cfg["padding"],
            use_maxpool=(cfg["stride"] == 1)
        ).to(device)

        # Build deconv shape
        dec.build_from_encoder(enc)

        return enc, dec


# ----------------- ARCHITECTURE GENERATOR -------------------
class ArchitectureGenerator:
    def __init__(
            self, in_size=128,
            kernel_lst=(3, 4, 5, 6, 7, 8),
            stride_lst=(1, 2, 3, 4),
            padding_lst=(2, 3, 4, 5, 6, 7, 8),
            channels_lst=(16, 32, 64, 128, 256, 512)
    ):
        self.in_size = in_size
        self.kernel_lst = kernel_lst
        self.stride_lst = stride_lst
        self.padding_lst = padding_lst
        self.channels_lst = channels_lst

    def generate(self):
        archs = []
        for kernel in self.kernel_lst:
            for stride in self.stride_lst:
                for pad in self.padding_lst:
                    H = W = self.in_size
                    channels = []
                    layers = 0

                    while True:
                        H_new = math.floor((H + pad - kernel)/stride) + 1
                        W_new = math.floor((W + pad - kernel)/stride) + 1

                        if stride == 1:  # maxpool simulation
                            H_new //= 2
                            W_new //= 2

                        if H_new <= 0 or W_new <= 0:
                            break
                        if min(H_new, W_new) <= 6:  # (6,6) is the minimum features final map
                            break

                        ch = random.choice(self.channels_lst)
                        channels.append(ch)
                        H, W = H_new, W_new
                        layers += 1

                    if layers == 0:
                        continue

                    archs.append(dict(
                        kernel=kernel,
                        stride=stride,
                        padding=pad,
                        channels=sorted(channels),
                        latent_dim=16
                    ))
        return archs


# --------------------- TRAINER CLASS ------------------------
class Trainer:
    def __init__(self, train_loader, val_loader=None, lr=1e-4, weight_decay=1e-5, max_epochs=1000, patience=10):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cuda = torch.cuda.is_available()
        self.use_amp = True
        self.encoder = None
        self.decoder = None
        # storage for plotting
        self.epoch_train_loss_values = []
        self.epoch_val_loss_values = []
        self.test_loss_value = None

    def train(self, encoder, decoder, early_stop=False, save_checkpoint=False):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        scaler = torch.amp.GradScaler(enabled=self.use_amp)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        best_val = float("inf")
        best_weights = None

        for epoch in range(self.max_epochs):
            encoder.train(); decoder.train()
            epoch_loss = 0
            batch_num = 0
            use_tqdm = True  # change to False in order to disable
            loader = self.train_loader
            if use_tqdm:
                loader = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}")

            for train_batch in loader:
                batch_num += 1
                train_batch = train_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.autocast(device_type=self.device, enabled=self.use_cuda):
                    latents = encoder(train_batch)
                    reconstructed = decoder(latents)
                    loss = loss_fn(reconstructed, train_batch)
                if self.use_cuda:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

            lr_scheduler.step()
            epoch_loss /= batch_num
            self.epoch_train_loss_values.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{self.max_epochs} training loss: {epoch_loss:.4f}')

            # validation
            if val_loader is not None:
                encoder.eval(); decoder.eval()
                val_loss_total = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(self.device, non_blocking=True)
                        val_latents = encoder(val_batch)
                        val_reconstructed = decoder(val_latents)
                        val_loss = loss_fn(val_reconstructed, val_batch)
                        val_loss_total += val_loss.item()

                val_epoch_loss = val_loss_total / len(val_loader)
                self.epoch_val_loss_values.append(val_epoch_loss)

                # check improvement
                if val_epoch_loss < best_val:
                    best_val = val_epoch_loss
                    best_loss_epoch = epoch + 1
                    best_weights = {
                        "encoder": copy.deepcopy(encoder.state_dict()),
                        "decoder": copy.deepcopy(decoder.state_dict()),
                    }

                    if save_checkpoint:
                        # save checkpoint of best model so far
                        torch.save({
                            "encoder": encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                            "epoch": best_loss_epoch,
                            "val_loss": best_val,
                        }, "best_model.pth")

                # ===== Early stopping =====
                if early_stop and len(self.epoch_val_loss_values) >= 2 * self.patience:
                    prev = np.mean(self.epoch_val_loss_values[-2 * self.patience:-self.patience])
                    curr = np.mean(self.epoch_val_loss_values[-self.patience:])
                    if curr > prev:  # validation deteriorated
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        # restore best weights
        if best_weights is not None:
            encoder.load_state_dict(best_weights["encoder"])
            decoder.load_state_dict(best_weights["decoder"])

        # store inside trainer
        self.encoder = encoder
        self.decoder = decoder

        return best_val, best_epoch

    def test(self, test_loader, cfg=None, plot_examples=False, num_examples=50, save_dir="recon_plots"):
        loss_fn = nn.MSELoss()
        self.encoder.eval()
        self.decoder.eval()

        tot = 0
        picked_orig = []
        picked_rec = []

        n_batches = len(test_loader)
        k = max(1, num_examples // n_batches)  # samples per batch

        with torch.no_grad():
            for xb in test_loader:
                xb = xb.to(self.device)
                z = self.encoder(xb)
                rec = self.decoder(z)
                tot += loss_fn(rec, xb).item()

                if plot_examples:
                    arch_str = ""
                    if cfg is not None:
                        arch_str = (
                            f"_k{cfg.get('kernel')}"
                            f"_s{cfg.get('stride')}"
                            f"_p{cfg.get('padding')}"
                        )
                    savepath = os.path.join(save_dir, arch_str)
                    os.makedirs(savepath, exist_ok=True)
                    # random indices within batch
                    b = xb.shape[0]
                    take = min(k, b)
                    idx = np.random.choice(b, take, replace=False)
                    # plot each pair of original+reconstructed frames
                    for i in idx:
                        self._plot_frame_pair(
                            orig=xb[i].cpu(),
                            recon=rec[i].cpu(),
                            save_path=os.path.join(savepath, f"example_{i}.png")
                        )

        self.test_loss_value = tot / len(test_loader)
        print(f"Test loss: {self.test_loss_value:.4f}")

        return self.test_loss_value

    def plot_losses(self, cfg=None, save_dir="plots", title="Loss Curves (MSE)"):
        os.makedirs(save_dir, exist_ok=True)
        arch_str = ""
        if cfg is not None:
            arch_str = (
                f"_k{cfg.get('kernel')}"
                f"_s{cfg.get('stride')}"
                f"_p{cfg.get('padding')}"
            )

        plt.figure(figsize=(8, 5))
        epochs_train = range(1, len(self.epoch_train_loss_values) + 1)
        plt.plot(epochs_train, self.epoch_train_loss_values,
                 label="Train Loss", linewidth=2, color="black")

        if len(self.epoch_val_loss_values) > 0:
            epochs_val = range(1, len(self.epoch_val_loss_values) + 1)
            plt.plot(epochs_val, self.epoch_val_loss_values,
                     label="Validation Loss", linewidth=2, color="blue")

        if self.test_loss_value is not None:
            plt.axhline(self.test_loss_value, color="red",
                        linestyle="--", label=f"Test Loss={self.test_loss_value:.4f}")

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        # unique filename with architecture spec
        safe_title = title.replace(" ", "_")
        filename = f"{safe_title}{arch_str}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def _plot_frame_pair(orig, recon, save_path):
        """
        orig, recon : tensors [2, H, W]  (2 cameras)
        Save one image with original & reconstruction for each camera
        Layout:
            Original Cam1 | Recon Cam1
            Original Cam2 | Recon Cam2
        """

        # Undo normalization: [-1,1] → [0,1] -> [0,255]
        def unnorm(x):
            x = x * 0.5 + 0.5  # inverse Normalize
            x = torch.clamp(x, 0, 1)
            x = (x * 255).byte()    # inverse scaling
            return x.numpy()

        # Unnormalize both
        o = unnorm(orig)
        r = unnorm(recon)

        # Expect shape [C, H, W], C=2
        cam1_o = o[0]
        cam1_r = r[0]
        cam2_o = o[1]
        cam2_r = r[1]

        # 2x2 grid figure
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))

        axes[0, 0].imshow(cam1_o, cmap="gray")
        axes[0, 0].set_title("Original Cam1")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cam1_r, cmap="gray")
        axes[0, 1].set_title("Recon Cam1")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(cam2_o, cmap="gray")
        axes[1, 0].set_title("Original Cam2")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cam2_r, cmap="gray")
        axes[1, 1].set_title("Recon Cam2")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()


# --------------------- SEARCHER ------------------------------
class ArchitectureSearcher:
    def __init__(self, builder, trainer, architectures, top_k=5):
        self.builder = builder
        self.trainer = trainer
        self.architectures = architectures
        self.top_k = top_k

    def run_fast_search(self):
        results = []
        for cfg in self.architectures:
            enc, dec = self.builder.build(cfg)
            val_loss = self.trainer.train(enc, dec, early_stop=False)
            results.append((val_loss, cfg))
        results.sort(key=lambda x:x[0])
        return results[:self.top_k]


# --------------------- LATENT EXTRACTOR ---------------------
class LatentExtractor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode(self, frames, batch_size=256):
        self.encoder.eval()
        loader = DataLoader(frames, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
        latents = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                z = self.encoder(xb)
                latents.append(z.cpu())
        return torch.cat(latents, dim=0).numpy()   # shape: [N_frames, D]


def fetch_video_start_trials(subject_id, session, frame_rate, dj_modules, clean_ignore=True, clean_omission=False):
    # unpacking data joint modules
    exp2 = dj_modules['exp2']
    tracking = dj_modules['tracking']
    key = {'subject_id': subject_id, 'session': session}  # specify key for dj fetching
    # Base exclusion - removing bad trials and grooming trials
    restriction_list = [tracking.TrackingTrialBad, tracking.VideoGroomingTrial]

    def get_restricted_table(restricted_table, restriction_list):
        if len(restriction_list) > 0:
            for restriction in restriction_list:
                restricted_table = restricted_table - restriction
        return restricted_table

    # Conditional exclusions - ignore and omission trials
    if clean_ignore:
        restriction_list.append(exp2.BehaviorTrial & 'outcome="ignore"')
    if clean_omission:
        restriction_list.append(exp2.TrialRewardSize & 'reward_size_type="omission"')

    TrackingTrial = get_restricted_table((tracking.TrackingTrial & key & {'tracking_device_id': 3}), restriction_list)
    trial_video_data = TrackingTrial.fetch()
    trial_duration = trial_video_data["tracking_duration"].astype(float)
    TrialsStartFrame = np.zeros(trial_video_data.shape[0])
    for i in range(1, len(TrialsStartFrame)):
        TrialsStartFrame[i] = TrialsStartFrame[i - 1] + int(np.ceil(trial_duration[i - 1] * frame_rate))

    return TrialsStartFrame



# take only 16 trials for a test


subject_id = 464724
session = 1
save_dir = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/subject {subject_id}/session{session}'
frame_rate = 150
host = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
user = 'ShaniE'
password = 'opala'
dj_info = {'host_path': host, 'user_name': user, 'password': password}
dj.config['database.host'] = dj_info['host_path']
dj.config['database.user'] = dj_info['user_name']
dj.config['database.password'] = dj_info['password']
conn = dj.conn()
tracking = dj.VirtualModule('TRACKING', 'arseny_learning_tracking')
exp2 = dj.VirtualModule('EXP2', 'arseny_s1alm_experiment2')
video_neural = dj.VirtualModule('VIDEONEURAL', "lab_videoNeuralAlignment")
dj_modules = {'tracking': tracking, 'exp2': exp2, 'video_neural': video_neural}

# retrieve videos
video0 = Video(subject_id, session, camera_num=0, video_path=None)
video1 = Video(subject_id, session, camera_num=1, video_path=None)
original_video_path = 'D:/Arseny_behavior_video'
video0.create_full_video_array(dj_modules, original_video_path, clean_ignore=False, clean_omission=False)
video1.create_full_video_array(dj_modules, original_video_path, clean_ignore=False, clean_omission=False)

# temporal downsample
video0.custom_temporal_downsampling(frame_rate, save_root=None)
video1.custom_temporal_downsampling(frame_rate, save_root=None)

# resize and crop video 0 to get shape (N,128,128)
video0.downsample_by_block_average(factor=2)
video0.crop_frames(new_H=128, new_W=128)

# resize and pas video 1 to get shape (N,128,128)
video1.downsample_by_block_average(factor=2)
video1.pad_frames(new_H=128, new_W=128)

# verify shapes match
assert video0.video_array.shape == video1.video_array.shape,  \
    f"mismatch: {video0.video_array.shape} vs {video1.video_array.shape}"

# concat two cameras to get video with shape (N ,C=2, W=128, H=128)
full_video = np.stack([video0.video_array, video1.video_array], axis=1)

start_trials = fetch_video_start_trials(subject_id, session, frame_rate, dj_modules)
# split video to data loaders
train_idx, val_idx, test_idx = split_data(full_video, start_trials[:16])
transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])  # scale [0,1] → [-1,1]
train_ds = VideoFramesDataset(full_video, train_idx, transform)
val_ds = VideoFramesDataset(full_video, val_idx, transform)
test_ds = VideoFramesDataset(full_video, test_idx, transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)


# Generate architectures
gen = ArchitectureGenerator(in_size=128)
architectures = gen.generate()

#Create model builder
builder = ModelBuilder(in_chan=2, out_chan=2, latent_num=16, H=128, W=128)

# Run fast search (20 epochs each)
trainer = Trainer(train_loader, val_loader, max_epochs=20)
searcher = ArchitectureSearcher(builder, trainer, architectures)
top5 = searcher.run_fast_search()

# Train top-5 fully with early stopping
best_models = []
for loss, cfg in top5:
    enc, dec = builder.build(cfg)
    best_val, best_epoch = Trainer(train_loader, val_loader, max_epochs=1000, patience=10).train(
        enc, dec, early_stop=True
    )

    # Evaluate test performance
    # plot reconstructed test frames vs original
    save_recon_frames = os.path.join(save_dir, 'reconstruction_frames')
    test_loss = trainer.test(test_loader, cfg, plot_examples=True, num_examples=50, save_dir=save_recon_frames)
    best_models.append((test_loss, best_epoch, cfg, enc, dec))
    # plot MSE loss for train, val and test during epochs
    save_loss_plot = os.path.join(save_dir, 'training_loss_plots')
    trainer.plot_losses(cfg, save_dir=save_loss_plot)


# rank the final models
best_models.sort(key=lambda x: x[0])
best = best_models[:1]
_, best_epoch, cfg, enc, dec = best

# train again on train+val data on best model
train_val_idx = train_idx + val_idx
train_val_ds = VideoFramesDataset(full_video, train_val_idx, transform)
train_val_loader = DataLoader(train_val_ds, batch_size=256, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
best_val, best_epoch = Trainer(train_val_loader, None, max_epochs=best_epoch).train(
    enc, dec, early_stop=False)

# Encode latents for all video frames
full_idx = train_idx + val_idx + test_idx
full_v_ds = VideoFramesDataset(full_video, full_idx, transform)
extractor = LatentExtractor(enc)
all_video_latents = extractor.encode(full_v_ds)

