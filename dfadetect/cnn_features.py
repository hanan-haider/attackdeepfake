from dataclasses import dataclass, field
from typing import List

import torch
import torchaudio

# ─── Settings ────────────────────────────────────────────────────────────────

@dataclass
class CNNFeaturesSetting:
    frontend_algorithm: List[str] = field(default_factory=lambda: ["mfcc"])
    use_spectrogram: bool = True


def count_input_channels(cfg: "CNNFeaturesSetting") -> int:
    """Returns number of input channels the CNN will receive."""
    count = 0
    if "mfcc" in cfg.frontend_algorithm:
        count += 1
    if "lfcc" in cfg.frontend_algorithm:
        count += 1
    if cfg.use_spectrogram:
        count += 2  # abs_mel + angle_mel from STFT
    return count


# ─── Audio constants 

SAMPLING_RATE = 16_000
N_MELS       = 80
N_MFCC       = 80
N_LFCC       = 80
N_FFT        = 1024          # ✅ increased from 512 → more freq bins (513) → no zero filterbanks
WIN_LENGTH   = 400           # 25ms @ 16kHz
HOP_LENGTH   = 160           # 10ms @ 16kHz
F_MIN        = 0.0
F_MAX        = 8000.0        # ✅ explicit f_max = Nyquist for 16kHz — eliminates the warning

N_STFT       = N_FFT // 2 + 1   # 513 freq bins

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Feature extractors

MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft":      N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mels":     N_MELS,    # ✅ explicitly set n_mels to match everywhere
        "f_min":      F_MIN,
        "f_max":      F_MAX,     # ✅ fix: explicit f_max silences the zero-filterbank warning
    },
).to(device)


LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=N_LFCC,
    speckwargs={
        "n_fft":      N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
    },
).to(device)


MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    n_stft=N_STFT,             # ✅ must match N_FFT // 2 + 1
    sample_rate=SAMPLING_RATE,
    f_min=F_MIN,
    f_max=F_MAX,               # ✅ explicit f_max — fixes zero filterbank warning
    norm="slaney",             # ✅ area-normalised filters → more stable gradients
    mel_scale="htk",
).to(device)


# ─── Feature preparation 

def prepare_feature_vector(
    audio: torch.Tensor,
    cnn_features_setting: CNNFeaturesSetting,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """
    Returns: [batch_size, num_channels, N_MELS, frames]
    num_channels depends on CNNFeaturesSetting:
      - mfcc only:           1
      - lfcc only:           1
      - mfcc + lfcc:         2
      - + use_spectrogram:   +2 (abs_mel, angle_mel)
    """
    feature_vector = []

    if "mfcc" in cnn_features_setting.frontend_algorithm:
        mfcc_feature = MFCC_FN(audio)          # [B, N_MFCC, T]
        feature_vector.append(mfcc_feature)

    if "lfcc" in cnn_features_setting.frontend_algorithm:
        lfcc_feature = LFCC_FN(audio)          # [B, N_LFCC, T]
        feature_vector.append(lfcc_feature)

    if cnn_features_setting.use_spectrogram:
        stft_features = prepare_stft_features(audio, win_length, hop_length)
        feature_vector += stft_features        # adds abs_mel and angle_mel

    assert len(feature_vector) >= 1, "Feature vector must contain at least one feature!"

    # Stack along channel dim → [B, num_channels, N_MELS, T]
    feature_vector = torch.stack(feature_vector, dim=1)
    return feature_vector


def prepare_stft_features(
    audio: torch.Tensor,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> tuple:
    """
    Computes mel-scaled magnitude and phase from STFT.
    Returns: (stft_abs_mel, stft_abs_angle) each of shape [B, N_MELS, T]
    """
    # Flatten batch if needed: torch.stft expects [B, T] or [T]
    batch_size = audio.shape[0]
    audio_flat = audio.squeeze(1)  # [B, T]

    stft_out = torch.stft(
        audio_flat,
        n_fft=N_FFT,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(audio.device),  # ✅ explicit Hann window
        return_complex=True,
        pad_mode="reflect",        # ✅ explicit padding for cleaner edge frames
    )
    # stft_out: [B, N_STFT, T]

    # Separate real/imag and apply mel filterbank
    stft_real_mel = MEL_SCALE_FN(stft_out.real)   # [B, N_MELS, T]
    stft_imag_mel = MEL_SCALE_FN(stft_out.imag)   # [B, N_MELS, T]

    complex_tensor = torch.complex(stft_real_mel, stft_imag_mel)
    stft_abs_mel   = complex_tensor.abs()          # magnitude → log-compressed below
    stft_abs_angle = complex_tensor.angle()        # phase

    # ✅ Log-compress magnitude for better dynamic range (like log-mel spectrogram)
    stft_abs_mel = torch.log1p(stft_abs_mel)

    return stft_abs_mel, stft_abs_angle