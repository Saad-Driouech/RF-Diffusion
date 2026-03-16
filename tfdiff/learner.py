import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import _nested_map


class tfdiffLoss(nn.Module):
    """Combined time + frequency domain MSE loss.

    Inputs are real-valued tensors of shape [B, N, C, 2] where the last
    dimension holds [real, imag]. The frequency term penalises spectral
    errors that plain time-domain MSE tends to ignore (e.g. wrong
    bandwidth or spectral shape).

    freq_weight: weight on the frequency term. Start at 0.1; increase
    toward 1.0 if PSD mismatch persists after many iterations.
    """

    def __init__(self, freq_weight: float = 0.1):
        super().__init__()
        self.freq_weight = freq_weight

    def forward(self, target, est):
        # ── time-domain MSE ──────────────────────────────────────────
        t_loss = torch.mean((target - est) ** 2)

        # ── frequency-domain MSE ─────────────────────────────────────
        # Convert [B, N, C, 2] → complex [B, N, C], then FFT over time
        t_c = torch.view_as_complex(target.contiguous())   # [B, N, C]
        p_c = torch.view_as_complex(est.contiguous())      # [B, N, C]
        t_fft = torch.fft.fft(t_c, dim=1)
        p_fft = torch.fft.fft(p_c, dim=1)
        f_loss = torch.mean(torch.abs(t_fft - p_fft) ** 2)

        return t_loss + self.freq_weight * f_loss
        

class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.task_id = params.task_id
        self.log_dir = log_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
        # self.prof = torch.profiler.profile(
        #     schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=2, active=1, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
        #     with_modules=True, with_flops=True
        # )
        # eeg
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, 5, gamma=0.5)
        # mimo
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=0.5)
        self.params = params
        self.iter = 0
        self.is_master = True
        self.loss_fn = tfdiffLoss() if self.task_id == 4 else nn.MSELoss()
        self.summary_writer = None
        # Load a fixed held-out validation batch for GNSS visualizations
        if self.task_id == 4:
            self.val_batch = self._load_gnss_val_batch(n_samples=4)

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'lr_scheduler' in state_dict:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.lexists(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_iter=None):
        device = next(self.model.parameters()).device
        # self.prof.start()
        while True:  # epoch
            for features in tqdm(self.dataset, desc=f'Epoch {self.iter // len(self.dataset)}') if self.is_master else self.dataset:
                if max_iter is not None and self.iter >= max_iter:
                    # self.prof.stop()
                    return
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_iter(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at iteration {self.iter}.')
                if self.is_master:
                    if self.iter % 50 == 0:
                        self._write_summary(self.iter, features, loss)
                    if self.iter % (len(self.dataset)) == 0:
                        self.save_to_checkpoint()
                # self.prof.step()
                self.iter += 1
            self.lr_scheduler.step()

    def train_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']  # orignial data, x_0, [B, N, S*A, 2]
        cond = features['cond']  # cond, c, [B, C]
        B = data.shape[0]
        # random diffusion step, [B]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64)
        degrade_data = self.diffusion.degrade_fn(
            data, t ,self.task_id)  # degrade data, x_t, [B, N, S*A, 2]
        predicted = self.model(degrade_data, t, cond)
        if self.task_id==3:
            data = data.reshape(-1,512,1,2)
        loss = self.loss_fn(data, predicted)
        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.optimizer.step()
        return loss

    def _load_gnss_val_batch(self, n_samples=4):
        """Load a fixed held-out batch from the GNSS test split."""
        from tfdiff.dataset import GNSSDataset, Collator
        val_ds = GNSSDataset(self.params.data_dir, mode='test')
        collator = Collator(self.params)
        samples = [val_ds[i] for i in range(min(n_samples, len(val_ds)))]
        batch = collator.collate(samples)
        return {k: v.to(self.device) for k, v in batch.items()}

    def _write_gnss_summary(self, writer, iter):
        """Log GNSS-specific panels to TensorBoard."""
        self.model.eval()
        with torch.no_grad():
            data = self.val_batch['data']   # [B, 1024, 4, 2]
            cond = self.val_batch['cond']   # [B, 7, 2]
            B = data.shape[0]
            t_max = (self.diffusion.max_step - 1) * torch.ones(B, dtype=torch.int64, device=self.device)

            # Degrade to full noise then reconstruct
            x_T = self.diffusion.degrade_fn(data, t_max.cpu(), task_id=4).to(self.device)
            x_hat = self.model(x_T, t_max, cond)           # [B, 1024, 4, 2]

            x0_c = torch.view_as_complex(data.contiguous())    # [B, 1024, 4]
            xh_c = torch.view_as_complex(x_hat.contiguous())   # [B, 1024, 4]

            # ── Scalars ────────────────────────────────────────────────────────
            # Spectral MSE
            fft_real = torch.fft.fft(x0_c, dim=1)
            fft_pred = torch.fft.fft(xh_c, dim=1)
            freq_mse = torch.mean(torch.abs(fft_real - fft_pred) ** 2).item()
            writer.add_scalar('val/freq_mse', freq_mse, iter)

            # Reconstruction SNR (dB)
            sig_pwr  = torch.mean(torch.abs(x0_c) ** 2).item()
            noise_pwr = torch.mean(torch.abs(x0_c - xh_c) ** 2).item() + 1e-12
            writer.add_scalar('val/snr_db', 10 * np.log10(sig_pwr / noise_pwr), iter)

            # Per-step reconstruction loss at t=0,25,50,75,99
            for step in [0, 25, 50, 75, 99]:
                t_s = step * torch.ones(B, dtype=torch.int64)
                x_s = self.diffusion.degrade_fn(data.cpu(), t_s, task_id=4).to(self.device)
                x_s_hat = self.model(x_s, t_s.to(self.device), cond)
                step_loss = self.loss_fn(data, x_s_hat).item()
                writer.add_scalar(f'val/recon_loss_t{step}', step_loss, iter)

            # Use first sample for all figure panels
            x0 = x0_c[0].cpu().numpy()     # [1024, 4] complex
            xh = xh_c[0].cpu().numpy()     # [1024, 4] complex

            # ── PSD overlay — one subplot per antenna ──────────────────────────
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            freqs = np.fft.fftshift(np.fft.fftfreq(1024))
            for i, ax in enumerate(axes.flat):
                psd_real = np.abs(np.fft.fftshift(np.fft.fft(x0[:, i]))) ** 2
                psd_pred = np.abs(np.fft.fftshift(np.fft.fft(xh[:, i]))) ** 2
                ax.semilogy(freqs, psd_real, label='Real',      alpha=0.85, lw=1.2)
                ax.semilogy(freqs, psd_pred, label='Generated', alpha=0.85, lw=1.2, linestyle='--')
                ax.set_title(f'Antenna {i+1} PSD')
                ax.set_xlabel('Normalised Frequency')
                ax.set_ylabel('Power')
                ax.legend(fontsize=7)
                ax.grid(True, which='both', linestyle='--', linewidth=0.4)
            plt.suptitle(f'Power Spectral Density — iter {iter}')
            plt.tight_layout()
            writer.add_figure('val/psd_per_antenna', fig, iter)
            plt.close(fig)

            # ── STFT Spectrogram comparison — antenna 0 ────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            n_fft, hop = 64, 16
            for ax, sig, title in zip(axes, [x0[:, 0], xh[:, 0]], ['Real', 'Generated']):
                spec = np.lib.stride_tricks.sliding_window_view(sig.real, n_fft)[::hop] * np.hanning(n_fft)
                spec_db = 20 * np.log10(np.abs(np.fft.fft(spec, axis=-1)[:, :n_fft//2]).T + 1e-9)
                im = ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f'{title} Spectrogram (Ant 1, real part)')
                ax.set_xlabel('Time Frame')
                ax.set_ylabel('Frequency Bin')
                plt.colorbar(im, ax=ax, format='%+2.0f dB')
            plt.suptitle(f'STFT Spectrogram — iter {iter}')
            plt.tight_layout()
            writer.add_figure('val/spectrogram', fig, iter)
            plt.close(fig)

            # ── Time-domain amplitude — first 256 samples per antenna ──────────
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
            t_ax = np.arange(256)
            for i, ax in enumerate(axes.flat):
                ax.plot(t_ax, np.abs(x0[:256, i]), label='Real',      alpha=0.85, lw=1.2)
                ax.plot(t_ax, np.abs(xh[:256, i]), label='Generated', alpha=0.85, lw=1.2, linestyle='--')
                ax.set_title(f'Antenna {i+1}  |IQ|')
                ax.set_xlabel('Sample')
                ax.legend(fontsize=7)
                ax.grid(True, linestyle='--', linewidth=0.4)
            plt.suptitle(f'Time-domain Amplitude (first 256 samples) — iter {iter}')
            plt.tight_layout()
            writer.add_figure('val/time_amplitude', fig, iter)
            plt.close(fig)

            # ── IQ Constellation — antennas 0 & 1 ─────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for i, ax in enumerate(axes):
                ax.scatter(x0[:, i].real, x0[:, i].imag, s=1, alpha=0.25, label='Real')
                ax.scatter(xh[:, i].real, xh[:, i].imag, s=1, alpha=0.25, label='Generated')
                ax.set_title(f'IQ Constellation — Antenna {i+1}')
                ax.set_xlabel('I')
                ax.set_ylabel('Q')
                ax.legend(fontsize=7, markerscale=6)
                ax.set_aspect('equal')
                ax.grid(True, linestyle='--', linewidth=0.4)
            plt.suptitle(f'IQ Constellation — iter {iter}')
            plt.tight_layout()
            writer.add_figure('val/iq_constellation', fig, iter)
            plt.close(fig)

            # ── Forward degradation visualisation ─────────────────────────────
            fig, axes = plt.subplots(1, 5, figsize=(16, 3))
            for ax, step in zip(axes, [0, 25, 50, 75, 99]):
                t_s = step * torch.ones(1, dtype=torch.int64)
                x_deg = self.diffusion.degrade_fn(data[:1].cpu(), t_s, task_id=4)
                sig_deg = torch.view_as_complex(x_deg.contiguous())[0, :, 0].numpy()
                spec = np.lib.stride_tricks.sliding_window_view(sig_deg.real, n_fft)[::hop] * np.hanning(n_fft)
                spec_db = 20 * np.log10(np.abs(np.fft.fft(spec, axis=-1)[:, :n_fft//2]).T + 1e-9)
                ax.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f't = {step}')
                ax.axis('off')
            plt.suptitle('Forward Degradation (Ant 1, real part)')
            plt.tight_layout()
            writer.add_figure('val/degradation_steps', fig, iter)
            plt.close(fig)

        self.model.train()

    def _write_summary(self, iter, features, loss):
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        if self.task_id == 4 and iter % 500 == 0:
            self._write_gnss_summary(writer, iter)
        writer.flush()
        self.summary_writer = writer
