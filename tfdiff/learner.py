import numpy as np
import scipy.signal
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

_GNSS_FS = 40.5e6  # GNSS sampling frequency (Hz)


def _gnss_spectrogram_db(x, fs=_GNSS_FS, noverlap=64):
    """Return (f, t, Sxx_dB) using scipy's spectrogram (Blackman window, two-sided PSD)."""
    f, t, Sxx = scipy.signal.spectrogram(
        x, fs=fs, nperseg=128, noverlap=noverlap,
        window='blackman', return_onesided=False, detrend=False, mode='psd')
    Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-20)
    return np.fft.fftshift(f), t, Sxx_db


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
        t_fft = torch.fft.fft(t_c, dim=1, norm='ortho')
        p_fft = torch.fft.fft(p_c, dim=1, norm='ortho')
        f_loss = torch.mean(torch.abs(t_fft - p_fft) ** 2)

        # Store components for logging
        self.last_t_loss = t_loss.item()
        self.last_f_loss = f_loss.item()

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
        # LR warmup for GNSS: ramp from 1% of target LR to full LR over lr_warmup_steps iters
        if self.task_id == 4:
            warmup_steps = getattr(self.params, 'lr_warmup_steps', 5000)
            self.lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        else:
            self.lr_warmup_scheduler = None
        self.summary_writer = None
        # Load fixed batches for GNSS visualizations (train to check overfitting, val to check generalisation)
        if self.task_id == 4:
            self.val_batch   = self._load_gnss_batch(mode='test',  n_samples=4)
            self.train_batch = self._load_gnss_batch(mode='train', n_samples=4)

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
                if self.lr_warmup_scheduler is not None:
                    self.lr_warmup_scheduler.step()
            if self.task_id != 4:
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

    def _load_gnss_batch(self, mode='test', n_samples=4):
        """Load a fixed batch from the GNSS dataset (mode='train' or 'test')."""
        from tfdiff.dataset import GNSSDataset, Collator
        ds = GNSSDataset(self.params.data_dir, mode=mode)
        collator = Collator(self.params)
        samples = [ds[i] for i in range(min(n_samples, len(ds)))]
        batch = collator.collate(samples)
        return {k: v.to(self.device) for k, v in batch.items()}

    def _write_gnss_summary(self, writer, iter, batch, prefix):
        """Log GNSS-specific panels to TensorBoard for a given batch.

        prefix: 'train' or 'val' — used as the TensorBoard tag namespace.
        Calling this for both splits lets you compare plots side-by-side
        in TensorBoard to detect overfitting.
        """
        self.model.eval()
        with torch.no_grad():
            data = batch['data']   # [B, 1024, 4, 2]
            cond = batch['cond']   # [B, 7, 2]
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
            writer.add_scalar(f'{prefix}/freq_mse', freq_mse, iter)

            # Reconstruction SNR (dB)
            sig_pwr  = torch.mean(torch.abs(x0_c) ** 2).item()
            noise_pwr = torch.mean(torch.abs(x0_c - xh_c) ** 2).item() + 1e-12
            writer.add_scalar(f'{prefix}/snr_db', 10 * np.log10(sig_pwr / noise_pwr), iter)

            # Per-step reconstruction loss at t=0,25,50,75,99
            for step in [0, 25, 50, 75, 99]:
                t_s = step * torch.ones(B, dtype=torch.int64)
                x_s = self.diffusion.degrade_fn(data.cpu(), t_s, task_id=4).to(self.device)
                x_s_hat = self.model(x_s, t_s.to(self.device), cond)
                step_loss = self.loss_fn(data, x_s_hat).item()
                writer.add_scalar(f'{prefix}/recon_loss_t{step}', step_loss, iter)
                # Noisy input amplitude — verify noise schedule magnitude is sane
                x_s_c = torch.view_as_complex(x_s.contiguous())
                writer.add_scalar(f'{prefix}/noisy_amp_t{step}', torch.abs(x_s_c).mean().item(), iter)

            # Output amplitude vs target amplitude — early zero-collapse detector
            output_amp = torch.abs(xh_c).mean().item()
            target_amp = torch.abs(x0_c).mean().item()
            writer.add_scalar(f'{prefix}/output_amp', output_amp, iter)
            writer.add_scalar(f'{prefix}/target_amp', target_amp, iter)
            writer.add_scalar(f'{prefix}/amp_ratio',  output_amp / (target_amp + 1e-12), iter)

            # Condition vector norm — verify conditioning is active (should be non-zero)
            writer.add_scalar(f'{prefix}/cond_norm', cond.norm(dim=-1).mean().item(), iter)

            # Weight histograms for key layers every 5000 iters
            if iter % 5000 == 0:
                for name, param in self.model.named_parameters():
                    if any(k in name for k in ('final_layer', 'adaLN_modulation', 'c_embed')):
                        writer.add_histogram(f'weights/{name}', param.data, iter)

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
            plt.suptitle(f'[{prefix}] Power Spectral Density — iter {iter}')
            plt.tight_layout()
            writer.add_figure(f'{prefix}/psd_per_antenna', fig, iter)
            plt.close(fig)

            # ── Spectrogram comparison — all 4 antennas (2×2 grid, Real | Generated) ──
            fig, axes = plt.subplots(4, 2, figsize=(12, 16))
            for i in range(4):
                f_ax, t_ax_s, Sxx_real = _gnss_spectrogram_db(x0[:, i])
                _,    _,      Sxx_gen  = _gnss_spectrogram_db(xh[:, i])
                vmin = min(Sxx_real.min(), Sxx_gen.min())
                vmax = max(Sxx_real.max(), Sxx_gen.max())
                extent = [t_ax_s[0] * 1e3, t_ax_s[-1] * 1e3, f_ax[0], f_ax[-1]]
                for ax, Sxx, title in zip(axes[i], [Sxx_real, Sxx_gen], ['Real', 'Generated']):
                    im = ax.imshow(Sxx, aspect='auto', origin='lower', cmap='turbo',
                                   vmin=vmin, vmax=vmax, extent=extent,
                                   interpolation='nearest')
                    ax.set_title(f'Antenna {i+1} — {title}')
                    ax.set_xlabel('t [ms]')
                    ax.set_ylabel('f [Hz]')
                    fig.colorbar(im, ax=ax, format='%+.0f dB-Hz')
            plt.suptitle(f'[{prefix}] Spectrogram — iter {iter}')
            plt.tight_layout()
            writer.add_figure(f'{prefix}/spectrogram', fig, iter)
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
            plt.suptitle(f'[{prefix}] Time-domain Amplitude (first 256 samples) — iter {iter}')
            plt.tight_layout()
            writer.add_figure(f'{prefix}/time_amplitude', fig, iter)
            plt.close(fig)

            # ── IQ time series — I(t) and Q(t) for all 4 antennas ────────────
            # 4 rows (antennas) × 2 cols (I | Q), first 256 samples
            fig, axes = plt.subplots(4, 2, figsize=(14, 12))
            t_iq = np.arange(256)
            for i in range(4):
                ax_i, ax_q = axes[i, 0], axes[i, 1]
                # I component
                ax_i.plot(t_iq, x0[:256, i].real, label='Real',      alpha=0.85, lw=1.0)
                ax_i.plot(t_iq, xh[:256, i].real, label='Generated', alpha=0.85, lw=1.0, linestyle='--')
                ax_i.set_title(f'Antenna {i+1} — I(t)')
                ax_i.set_xlabel('Sample')
                ax_i.set_ylabel('I')
                ax_i.legend(fontsize=7)
                ax_i.grid(True, linestyle='--', linewidth=0.4)
                # Q component
                ax_q.plot(t_iq, x0[:256, i].imag, label='Real',      alpha=0.85, lw=1.0)
                ax_q.plot(t_iq, xh[:256, i].imag, label='Generated', alpha=0.85, lw=1.0, linestyle='--')
                ax_q.set_title(f'Antenna {i+1} — Q(t)')
                ax_q.set_xlabel('Sample')
                ax_q.set_ylabel('Q')
                ax_q.legend(fontsize=7)
                ax_q.grid(True, linestyle='--', linewidth=0.4)
            plt.suptitle(f'[{prefix}] IQ Time Series (first 256 samples) — iter {iter}')
            plt.tight_layout()
            writer.add_figure(f'{prefix}/iq_time_series', fig, iter)
            plt.close(fig)

            # ── IQ Constellation — all 4 antennas ─────────────────────────────
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.scatter(x0[:, i].real, x0[:, i].imag, s=1, alpha=0.25, label='Real')
                ax.scatter(xh[:, i].real, xh[:, i].imag, s=1, alpha=0.25, label='Generated')
                ax.set_title(f'IQ Constellation — Antenna {i+1}')
                ax.set_xlabel('I')
                ax.set_ylabel('Q')
                ax.legend(fontsize=7, markerscale=6)
                ax.set_aspect('equal')
                ax.grid(True, linestyle='--', linewidth=0.4)
            plt.suptitle(f'[{prefix}] IQ Constellation — iter {iter}')
            plt.tight_layout()
            writer.add_figure(f'{prefix}/iq_constellation', fig, iter)
            plt.close(fig)

            # ── Forward degradation — Antenna 1, 5 timesteps ──────────────────
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for ax, step in zip(axes, [0, 25, 50, 75, 99]):
                t_s = step * torch.ones(1, dtype=torch.int64)
                x_deg = self.diffusion.degrade_fn(data[:1].cpu(), t_s, task_id=4)
                sig_deg = torch.view_as_complex(x_deg.contiguous())[0, :, 0].numpy()
                _, t_ax_s, Sxx_db = _gnss_spectrogram_db(sig_deg)
                ax.imshow(Sxx_db, aspect='auto', origin='lower', cmap='turbo',
                          interpolation='nearest')
                ax.set_title(f't = {step}')
                ax.set_xlabel('t [ms]')
                ax.set_ylabel('f [Hz]' if step == 0 else '')
            plt.suptitle(f'[{prefix}] Forward Degradation — Antenna 1 — iter {iter}')
            plt.tight_layout()
            writer.add_figure(f'{prefix}/degradation_steps', fig, iter)
            plt.close(fig)

        self.model.train()

    def _write_summary(self, iter, features, loss):
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iter)
        if self.task_id == 4 and hasattr(self.loss_fn, 'last_t_loss'):
            writer.add_scalar('train/t_loss', self.loss_fn.last_t_loss, iter)
            writer.add_scalar('train/f_loss', self.loss_fn.last_f_loss, iter)
        if self.task_id == 4 and iter % 500 == 0:
            self._write_gnss_summary(writer, iter, self.val_batch,   prefix='val')
            self._write_gnss_summary(writer, iter, self.train_batch, prefix='train')
        writer.flush()
        self.summary_writer = writer
