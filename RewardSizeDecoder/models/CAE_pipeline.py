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
from data_preprocessing.VideoPipeline import Video
import multiprocessing


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
        enc = FlexibleEncoder(cfg).to(device)
        enc.build_model()

        dec = FlexibleDecoder(cfg).to(device)
        dec.build_model()

        return enc, dec


# ----------------- ARCHITECTURE GENERATOR -------------------
class ArchitectureGenerator:
    def __init__(
            self,
            input_size=(2, 128, 128),
            num_latents=16,
            kernel_lst=(3, 4, 5, 6, 7, 8),
            stride_lst=(1, 2, 3, 4),
            channels_lst=(64, 256, 32, 128, 512, 16, 1024)   #  (16, 32, 64, 128, 256, 512, 1024)
    ):
        self.input_size = input_size
        self.num_latents = num_latents
        self.kernel_lst = kernel_lst
        self.stride_lst = stride_lst
        self.channels_lst = channels_lst

    def generate(self):
        """
        Generates multiple autoencoder architectures with a fixed number of latents.
        """

        archs = []
        for kernel in self.kernel_lst:
            for stride in self.stride_lst:
                input_dim = self.input_size[1]
                out_dim_lst = [input_dim]
                channels = []
                layers = 0
                ch_curr = 0
                pad_lst = []
                while input_dim > 6 and ch_curr < len(self.channels_lst):
                    output_dim = (input_dim + stride - 1) // stride
                    total_padding_needed = max(0, (output_dim - 1) * stride + kernel - input_dim)
                    left_pad = total_padding_needed // 2
                    right_pad = total_padding_needed - left_pad

                    if stride == 1:  # maxpool
                        output_dim //= 2

                    out_dim_lst.append(output_dim)
                    pad_lst.append((left_pad, right_pad))
                    channels.append(self.channels_lst[ch_curr])
                    input_dim = output_dim
                    ch_curr +=1
                    layers += 1

                if layers == 0:     # not a valid architecture
                    continue
                # todo - maybe replace it with a generator that yields each time only one architecture to save memory
                archs.append(dict(
                    input_dim=self.input_size,  # original frame size (channels, H, W)
                    num_latent=self.num_latents,
                    kernel=kernel,              # fixed kernel size for all layers - might be variational in future version
                    stride=stride,              # fixed stride size for all layers - might be variational in future version
                    padding=pad_lst,            # list of tuples - padding for every layer (supports asymmetric padding)
                    channels=sorted(channels),  # output channels for every layer
                    output_dim=out_dim_lst,     # frame size after each conv layer
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
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_cuda)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        best_val_loss = float("inf")
        best_epoch = 0
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
                    if train_batch.size() != reconstructed.size():
                        raise ValueError(f'target latents size does not match reconstructed latents size\nparams: {encoder.hparams}')
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
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_epoch = epoch + 1
                    best_weights = {
                        "encoder": copy.deepcopy(encoder.state_dict()),
                        "decoder": copy.deepcopy(decoder.state_dict()),
                    }

                    # todo: add save path off the model to torch.save()

                    if save_checkpoint:
                        # save checkpoint of best model so far
                        torch.save({
                            "encoder": encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                            "epoch": best_epoch,
                            "val_loss": best_val_loss,
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

        return best_val_loss, best_epoch

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
            val_loss, _ = self.trainer.train(enc, dec, early_stop=False)
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




if __name__ == "__main__":
    subject_id = 464724
    session = 1
    latent_num = 16
    save_dir = f'C:/Users/admin/RewardSizeDecoder pipeline/RewardSizeDecoder/results/AE/latent_num={latent_num}/subject {subject_id}/session{session}'
    frame_rate = 125
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
    video0.create_full_video_array(dj_modules, original_video_path, clean_ignore=True, clean_omission=False)
    video1.create_full_video_array(dj_modules, original_video_path, clean_ignore=True, clean_omission=False)

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
    train_idx, val_idx, test_idx = split_data(full_video, start_trials)
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])  # scale [0,1] -> [-1,1]
    train_ds = VideoFramesDataset(full_video, train_idx, transform)
    val_ds = VideoFramesDataset(full_video, val_idx, transform)
    test_ds = VideoFramesDataset(full_video, test_idx, transform)
    import logging
    multiprocessing.log_to_stderr().setLevel(logging.CRITICAL)


    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)


    # Generate architectures
    gen = ArchitectureGenerator(input_size=full_video.shape[1:], num_latents=latent_num)
    architectures = gen.generate()

    #Create model builder
    builder = ModelBuilder(in_chan=2, out_chan=2, latent_num=latent_num, H=128, W=128)

    # Run fast search (20 epochs each)
    trainer = Trainer(train_loader, val_loader, max_epochs=20)
    searcher = ArchitectureSearcher(builder, trainer, architectures)
    top5 = searcher.run_fast_search()

    # Train top-5 fully with early stopping
    best_models = []
    for loss, cfg in top5:
        enc, dec = builder.build(cfg)
        trainer = Trainer(train_loader, val_loader, max_epochs=500, patience=10)
        best_val, best_epoch = trainer.train(enc, dec, early_stop=True)

        # Evaluate test performance
        # plot reconstructed test frames vs original
        save_recon_frames = os.path.join(save_dir, 'reconstruction_frames')
        test_loss = trainer.test(test_loader, cfg, plot_examples=True, num_examples=50, save_dir=save_recon_frames)
        best_models.append((test_loss, best_epoch, cfg, enc, dec))
        # plot MSE loss for train, val and test during epochs
        save_loss_plot = os.path.join(save_dir, 'training_loss_plots')
        trainer.plot_losses(cfg, save_dir=save_loss_plot)


    # rank the final models and proceed with the best performed model
    best_models.sort(key=lambda x: x[0])
    best = best_models[0]
    _, best_epoch, cfg, enc, dec = best

    # train again on train+val data on best model
    train_val_idx = np.concatenate([train_idx, val_idx])
    train_val_ds = VideoFramesDataset(full_video, train_val_idx, transform)
    train_val_loader = DataLoader(train_val_ds, batch_size=256, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
    best_val, best_epoch = Trainer(train_val_loader, None, max_epochs=best_epoch).train(
        enc, dec, early_stop=False)

    # Encode latents for all video frames
    full_idx = np.concatenate([train_idx, val_idx, test_idx])
    full_v_ds = VideoFramesDataset(full_video, full_idx, transform)
    extractor = LatentExtractor(enc)
    all_video_latents = extractor.encode(full_v_ds)     # shape: [N_frames, D]
    # save latents
    np.save(os.path.join(save_dir, f'all_video_latents_dim={latent_num}.npy'), all_video_latents)

# todo - add check memory of each model architecture, estimate_model_footprint
# test