"""
cnn_features.py  —  Improved Frontend for Audio Deepfake Detection
====================================================================
Drop-in replacement for dfadetect/cnn_features.py in the
hanan-haider/attackdeepfake repository.

WHAT CHANGED vs. ORIGINAL  (numbered so you can audit each line)
──────────────────────────────────────────────────────────────────

FIX-1  CMVN (Cepstral Mean & Variance Normalisation)
       The original returns raw un-normalised cepstral coefficients.
       The AAD paper and every ASVspoof LCNN baseline apply per-utterance
       CMVN before the CNN.  Without it the model must absorb
       speaker/channel magnitude drift through its own weights, which is
       the primary driver of cross-fold EER variance.
       → apply_cmvn() subtracts utterance mean and divides by std along
         the TIME axis (per coefficient).  eps=1e-9 handles silent clips.

FIX-2  Pre-emphasis before STFT
       Raw waveform fed to torch.stft() without pre-emphasis misses the
       high-frequency naturalness cues that distinguish synthesised from
       real speech.  y[n] = x[n] − 0.97·x[n-1] is the standard AAD value.
       Applied ONLY to the spectrogram path; MFCC/LFCC handle it
       internally already.

FIX-3  Hann window on torch.stft()
       The original call had NO window argument → rectangular window
       → severe spectral leakage (spurious HF energy).  Every ASVspoof
       LCNN baseline uses a Hann window.
       → window=torch.hann_window(win_length, device=audio.device)

FIX-4  Power-law compression on mel magnitude
       Raw mel magnitude spans 5-6 orders of magnitude.  Feeding this
       directly to a CNN creates enormous gradient scale differences between
       channels.  power_compress(x, 0.3) maps to a perceptually uniform
       scale.  Phase (angle) is left unchanged — already in [−π, π].

FIX-5  Delta & Delta-Delta coefficients  (optional, default OFF)
       Temporal dynamics (rate of change) of cepstral features are strong
       liveness cues.  Controlled by use_deltas in the config.
       When enabled with a single-channel cepstral frontend the output
       becomes [B, 3, 80, T] (static + Δ + ΔΔ).
       Leave False for the paper-comparison baseline run.

FIX-6  Time-dimension alignment before torch.stack()
       MFCC and STFT can produce slightly different T lengths due to
       hop_length rounding.  The original torch.stack() would crash with a
       cryptic shape error.  _align_time() crops all features to T_min.

FIX-7  count_input_channels() helper
       Returns the correct input_channels value for lcnn.yaml given a
       CNNFeaturesSetting, so you never get a model shape mismatch.

CONFIG REFERENCE
──────────────────────────────────────────────────────────────────
  frontend_algorithm: ["lfcc"]           → input_channels: 1
  frontend_algorithm: ["mfcc"]           → input_channels: 1
  frontend_algorithm: ["mfcc", "lfcc"]   → input_channels: 2
  frontend_algorithm: ["lfcc"],
    use_spectrogram: True                → input_channels: 3
  frontend_algorithm: ["lfcc"],
    use_deltas: True                     → input_channels: 3
  frontend_algorithm: ["mfcc", "lfcc"],
    use_spectrogram: True                → input_channels: 4
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torchaudio
import torchaudio.functional as F_audio


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants  —  identical to original (FakeAVCeleb / ASVspoof values)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLING_RATE = 16_000
WIN_LENGTH    = 400          # 25 ms at 16 kHz
HOP_LENGTH    = 160          # 10 ms at 16 kHz
N_FFT         = 512
N_COEFF       = 80           # number of cepstral / mel bins
PRE_EMPHASIS  = 0.97         # standard speech pre-emphasis coefficient

device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
#  Config dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CNNFeaturesSetting:
    """
    Controls which acoustic features are extracted.

    Fields
    ──────
    frontend_algorithm : list of str
        Any combination of "mfcc" and/or "lfcc".
        Each adds one channel to the output tensor.

    use_spectrogram : bool  (default False)
        If True, adds 2 channels: power-compressed mel magnitude + phase.
        Set to False for the paper-comparison run.

    use_deltas : bool  (default False)  [FIX-5]
        If True AND exactly one cepstral algorithm is selected AND
        use_spectrogram is False, the single-channel cepstral feature is
        expanded to 3 channels via stacking of static + Δ + ΔΔ.
        Leave False for the paper-comparison baseline.
    """
    frontend_algorithm: List[str] = field(default_factory=lambda: ["lfcc"])
    use_spectrogram:    bool = False
    use_deltas:         bool = False   # FIX-5: temporal dynamics (off for baseline)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stateless transforms  —  initialised once and cached on device
# ═══════════════════════════════════════════════════════════════════════════════

MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=N_COEFF,
    melkwargs={
        "n_fft":      N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mels":     N_COEFF,
        "window_fn":  torch.hann_window,   # FIX-3: Hann window (was: rectangular)
    },
).to(device)

LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=N_COEFF,
    speckwargs={
        "n_fft":      N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
    },
).to(device)

MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=N_COEFF,
    n_stft=N_FFT // 2 + 1,   # = 257 for N_FFT = 512
    sample_rate=SAMPLING_RATE,
).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def pre_emphasize(
    audio: torch.Tensor,
    coeff: float = PRE_EMPHASIS,
) -> torch.Tensor:
    """
    FIX-2  Pre-emphasis filter:  y[n] = x[n] − coeff * x[n-1]

    Boosts high-frequency naturalness cues before STFT.
    Applied ONLY to the spectrogram path; MFCC/LFCC already include
    their own internal windowing.

    Args:
        audio  : [..., samples]  float32
        coeff  : pre-emphasis coefficient (default 0.97)

    Returns:
        [..., samples]  same shape and device as input
    """
    return torch.cat(
        [audio[..., :1],
         audio[..., 1:] - coeff * audio[..., :-1]],
        dim=-1,
    )


def apply_cmvn(feat: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    FIX-1  Per-utterance Cepstral Mean & Variance Normalisation (CMVN)

    Normalises each cepstral coefficient independently across the TIME axis
    so that the CNN sees zero-mean, unit-variance features regardless of the
    speaker or recording channel.  This is the most important fix for
    cross-fold EER stability.

    Args:
        feat : [..., n_coeff, time]
        eps  : stability epsilon for division

    Returns:
        [..., n_coeff, time]  — normalised
    """
    mean = feat.mean(dim=-1, keepdim=True)
    std  = feat.std(dim=-1, keepdim=True).clamp(min=eps)
    return (feat - mean) / std


def power_compress(x: torch.Tensor, power: float = 0.3) -> torch.Tensor:
    """
    FIX-4  Power-law (root) compression of magnitude

    Maps the 5-6 order-of-magnitude dynamic range of mel magnitude to a
    perceptually uniform scale:  out = sign(x) * |x|^power

    Applied to magnitude channels only.  Phase (angle) is left unchanged.

    Args:
        x     : tensor of any shape
        power : compression exponent (default 0.3)

    Returns:
        same shape and device as x
    """
    return x.sign() * x.abs().clamp(min=1e-9).pow(power)


def compute_deltas(feat: torch.Tensor) -> torch.Tensor:
    """
    FIX-5  Stack static + first derivative (Δ) + second derivative (ΔΔ)

    Args:
        feat : [batch, 1, n_coeff, time]

    Returns:
        [batch, 3, n_coeff, time]   (static | Δ | ΔΔ)
    """
    static = feat.squeeze(1)
    delta  = F_audio.compute_deltas(static)
    ddelta = F_audio.compute_deltas(delta)
    return torch.stack([static, delta, ddelta], dim=1)


def _align_time(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """
    FIX-6  Crop all feature tensors to the minimum time length.

    Args:
        *tensors : variable number of tensors, all [..., n_coeff, time]

    Returns:
        list of tensors all cropped to [..., n_coeff, T_min]
    """
    t_min = min(t.size(-1) for t in tensors)
    return [t[..., :t_min] for t in tensors]


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_feature_vector(
    audio: torch.Tensor,
    cnn_features_setting: CNNFeaturesSetting,
    win_length: int = WIN_LENGTH,
    hop_length:  int = HOP_LENGTH,
) -> torch.Tensor:
    """
    Extract the acoustic feature tensor from a raw waveform.

    Interface is IDENTICAL to the original — only internals changed.

    Args:
        audio                : [batch, samples] float32 at 16 kHz
        cnn_features_setting : CNNFeaturesSetting from config.yaml
        win_length, hop_length : STFT parameters

    Returns:
        [batch, C, 80, T]
    """

    channels: List[torch.Tensor] = []

    # ── Step 1: Cepstral features ─────────────────────────────────────────────
    if "mfcc" in cnn_features_setting.frontend_algorithm:
        mfcc = MFCC_FN(audio)
        mfcc = apply_cmvn(mfcc)     # FIX-1
        channels.append(mfcc)

    if "lfcc" in cnn_features_setting.frontend_algorithm:
        lfcc = LFCC_FN(audio)
        lfcc = apply_cmvn(lfcc)     # FIX-1
        channels.append(lfcc)

    # ── Step 2: Mel-spectrogram features ──────────────────────────────────────
    if cnn_features_setting.use_spectrogram:
        audio_pe = pre_emphasize(audio)                    # FIX-2
        abs_mel, angle_mel = prepare_stft_features(
            audio_pe, win_length, hop_length
        )
        channels.append(abs_mel)
        channels.append(angle_mel)

    # ── Step 3: Sanity check ──────────────────────────────────────────────────
    assert len(channels) >= 1, (
        "Feature vector is empty!  "
        "Check frontend_algorithm and use_spectrogram in your config."
    )

    # ── Step 4: Align time dimensions (FIX-6) ─────────────────────────────────
    channels = _align_time(*channels)

    # ── Step 5: Stack → [B, C, 80, T] ────────────────────────────────────────
    feature_vector = torch.stack(channels, dim=1)

    # ── Step 6: Optional Δ / ΔΔ stacking (FIX-5) ─────────────────────────────
    if (
        cnn_features_setting.use_deltas
        and feature_vector.size(1) == 1
        and not cnn_features_setting.use_spectrogram
    ):
        feature_vector = compute_deltas(feature_vector)

    return feature_vector


# ═══════════════════════════════════════════════════════════════════════════════
#  STFT helper
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_stft_features(
    audio: torch.Tensor,
    win_length: int = WIN_LENGTH,
    hop_length:  int = HOP_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mel-scaled magnitude and phase from STFT.

    FIX-3 : Hann window (was: no window = rectangular)
    FIX-4 : power_compress on magnitude (was: raw magnitude)
    FIX-2 : caller applies pre-emphasis before this function

    Returns:
        (stft_abs_mel, stft_abs_angle)  each [batch, 80, T]
    """
    window = torch.hann_window(win_length, device=audio.device)  # FIX-3

    stft_out = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        pad_mode="reflect",
    )

    stft_real_mel = MEL_SCALE_FN(stft_out.real)
    stft_imag_mel = MEL_SCALE_FN(stft_out.imag)

    complex_mel    = torch.complex(stft_real_mel, stft_imag_mel)
    stft_abs_mel   = power_compress(complex_mel.abs())   # FIX-4
    stft_abs_angle = complex_mel.angle()

    return stft_abs_mel, stft_abs_angle


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility  —  count_input_channels()
# ═══════════════════════════════════════════════════════════════════════════════

def count_input_channels(cfg: CNNFeaturesSetting) -> int:
    """
    FIX-7  Returns the input_channels value required in lcnn.yaml.

    Example:
        cfg = CNNFeaturesSetting(frontend_algorithm=["lfcc"])
        print(count_input_channels(cfg))  # → 1
    """
    n_cepstral = len(cfg.frontend_algorithm)
    if cfg.use_deltas and n_cepstral == 1 and not cfg.use_spectrogram:
        return 3
    n_spec = 2 if cfg.use_spectrogram else 0
    return n_cepstral + n_spec


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test  (run: python -m dfadetect.cnn_features)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("Running cnn_features self-test …")
    torch.manual_seed(0)
    dummy = torch.randn(2, 3 * SAMPLING_RATE).to(device)

    tests = [
        CNNFeaturesSetting(["lfcc"],        False, False),
        CNNFeaturesSetting(["mfcc"],        False, False),
        CNNFeaturesSetting(["mfcc","lfcc"], False, False),
        CNNFeaturesSetting(["lfcc"],        True,  False),
        CNNFeaturesSetting(["mfcc"],        False, True),
        CNNFeaturesSetting(["mfcc","lfcc"], True,  False),
    ]

    all_ok = True
    print(f"{'Config':<52} {'Exp':>4} {'Got':>4} {'OK?':>5}")
    print("─" * 68)
    for cfg in tests:
        exp = count_input_channels(cfg)
        try:
            got = prepare_feature_vector(dummy, cfg).size(1)
            ok  = "✓" if got == exp else "✗ MISMATCH"
            if got != exp: all_ok = False
        except Exception as e:
            got, ok = "ERR", str(e); all_ok = False
        label = f"algo={cfg.frontend_algorithm} spec={cfg.use_spectrogram} d={cfg.use_deltas}"
        print(f"{label:<52} {exp:>4} {str(got):>4} {ok:>5}")

    print("─" * 68)
    print("ALL PASSED ✓" if all_ok else "SOME FAILED ✗")
    sys.exit(0 if all_ok else 1)