from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch
import torchaudio
from torchaudio.functional import compute_deltas


# ======================
# Settings
# ======================

@dataclass
class CNNFeaturesSetting:
    # which cepstral frontends to use: any subset of {"mfcc", "lfcc"}
    frontend_algorithm: List[str] = field(default_factory=lambda: ["mfcc", "lfcc"])
    # whether to use STFT-based spectrogram magnitude/phase (mel-projected)
    use_stft_spectrogram: bool = True
    # whether to use a log-mel spectrogram (magnitude only)
    use_logmel: bool = True
    # whether to append Δ and Δ² for cepstral features
    use_deltas: bool = True
    # whether to include an F0 (pitch) feature map
    use_f0: bool = False


# values from FakeAVCeleb paper
SAMPLING_RATE = 16_000
WIN_LENGTH = 400  # int((25 / 1_000) * SAMPLING_RATE)
HOP_LENGTH = 160  # int((10 / 1_000) * SAMPLING_RATE)

N_FFT = 512
N_MELS = 80
N_CEPS = 80  # for both MFCC and LFCC

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# Transforms
# ======================

MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=N_CEPS,
    melkwargs={
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
    },
).to(device)

LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=N_CEPS,
    speckwargs={
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
    },
).to(device)

# log-mel spectrogram (magnitude only)
MELSPEC_FN = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLING_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    center=True,
    power=2.0,  # magnitude^2
).to(device)

AMPSPEC_TO_DB = torchaudio.transforms.AmplitudeToDB().to(device)

# mel projection for STFT real/imag
MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    n_stft=N_FFT // 2 + 1,
    sample_rate=SAMPLING_RATE,
).to(device)

# optional F0 extractor (PyTorch 2.1+ / torchaudio pitch API)
# We will call torchaudio.functional.detect_pitch_frequency for each sample


# ======================
# Core feature preparation
# ======================

def _add_deltas(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: [B, C, T] -> returns [B, 3*C, T] with Δ and Δ² concatenated on channel dim.
    """
    delta = compute_deltas(feat)
    delta2 = compute_deltas(delta)
    return torch.cat([feat, delta, delta2], dim=1)


def prepare_feature_vector(
    audio: torch.Tensor,
    cnn_features_setting: Optional[CNNFeaturesSetting] = None,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """
    audio: [B, 1, L] or [B, L]
    returns: [B, feature_num, F, T] where feature_num ∈ {1,2,...}
    """

    if cnn_features_setting is None:
        cnn_features_setting = CNNFeaturesSetting()

    # ensure shape [B, L]
    if audio.dim() == 3:
        # [B, 1, L] -> [B, L]
        audio = audio.squeeze(1)

    feature_maps: List[torch.Tensor] = []

    # ---------- MFCC ----------
    if "mfcc" in cnn_features_setting.frontend_algorithm:
        mfcc = MFCC_FN(audio)  # [B, C, T]
        if cnn_features_setting.use_deltas:
            mfcc = _add_deltas(mfcc)  # [B, 3*C, T]
        # reshape to [B, 1, F, T] by stacking channels into frequency axis
        B, C, T = mfcc.shape
        mfcc = mfcc.view(B, 1, C, T)
        feature_maps.append(mfcc)

    # ---------- LFCC ----------
    if "lfcc" in cnn_features_setting.frontend_algorithm:
        lfcc = LFCC_FN(audio)  # [B, C, T]
        if cnn_features_setting.use_deltas:
            lfcc = _add_deltas(lfcc)  # [B, 3*C, T]
        B, C, T = lfcc.shape
        lfcc = lfcc.view(B, 1, C, T)
        feature_maps.append(lfcc)

    # ---------- STFT-based magnitude/phase in mel space ----------
    if cnn_features_setting.use_stft_spectrogram:
        stft_abs_mel, stft_angle_mel = prepare_stft_features(
            audio, win_length, hop_length
        )  # each [B, F, T]
        # treat each as one "feature map": [B, 1, F, T]
        feature_maps.append(stft_abs_mel.unsqueeze(1))
        feature_maps.append(stft_angle_mel.unsqueeze(1))

    # ---------- Log-mel spectrogram (power -> dB) ----------
    if cnn_features_setting.use_logmel:
        mel_spec = MELSPEC_FN(audio)  # [B, F, T], power
        mel_db = AMPSPEC_TO_DB(mel_spec)  # [B, F, T]
        feature_maps.append(mel_db.unsqueeze(1))

    # ---------- F0 feature map ----------
    if cnn_features_setting.use_f0:
        f0_map = prepare_f0_features(audio, hop_length=hop_length)  # [B, 1, 1, T]
        feature_maps.append(f0_map)

    assert len(feature_maps) >= 1, "Feature vector must contain at least one feature!"

    # stack on feature_num dimension
    # Each element is [B, 1, F, T]; after stack -> [B, feature_num, F, T]
    feature_vector = torch.cat(feature_maps, dim=1)

    # final shape: [batch_size, feature_num, F, frames]
    return feature_vector


# ======================
# STFT-based mel magnitude/phase
# ======================

def prepare_stft_features(
    audio: torch.Tensor,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    audio: [B, L]
    returns:
        stft_abs_mel:   [B, N_MELS, T]
        stft_angle_mel: [B, N_MELS, T]
    """
    # STFT -> [B, F, T]
    stft_out = torch.stft(
        audio,
        n_fft=N_FFT,
        return_complex=True,
        hop_length=hop_length,
        win_length=win_length,
    )

    # project real/imag separately via mel filterbanks
    stft_real_mel = MEL_SCALE_FN(stft_out.real)  # [B, N_MELS, T]
    stft_imag_mel = MEL_SCALE_FN(stft_out.imag)  # [B, N_MELS, T]

    complex_tensor = torch.complex(stft_real_mel, stft_imag_mel)
    stft_abs_mel = complex_tensor.abs()
    stft_angle_mel = complex_tensor.angle()

    return stft_abs_mel, stft_angle_mel


# ======================
# F0 / Pitch features
# ======================

def prepare_f0_features(
    audio: torch.Tensor,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """
    audio: [B, L]
    returns: [B, 1, 1, T_frames]
        - F0 per frame (Hz), normalized to [0,1] across batch for stability.
    """
    B, L = audio.shape
    f0_list = []

    for b in range(B):
        # torchaudio pitch detection: [T]
        # frame_length ~ win_length; we re-use hop_length
        f0 = torchaudio.functional.detect_pitch_frequency(
            audio[b].unsqueeze(0),
            sample_rate=SAMPLING_RATE,
            frame_time=win_length_to_ms(WIN_LENGTH),
            win_length=WIN_LENGTH,
            hop_length=hop_length,
        )
        f0_list.append(f0)

    # pad to same length and stack
    f0_padded = torch.nn.utils.rnn.pad_sequence(
        f0_list, batch_first=True
    )  # [B, T_max]
    # simple normalization over batch+time
    f0_min = f0_padded[f0_padded > 0].min() if (f0_padded > 0).any() else 0.0
    f0_max = f0_padded.max() if (f0_padded > 0).any() else 1.0
    if f0_max > f0_min:
        f0_norm = (f0_padded - f0_min) / (f0_max - f0_min)
    else:
        f0_norm = f0_padded * 0.0

    # shape [B, 1, 1, T]
    f0_norm = f0_norm.unsqueeze(1).unsqueeze(1)
    return f0_norm


def win_length_to_ms(win_length: int) -> float:
    """
    Helper: convert win_length (samples) to milliseconds.
    """
    return (win_length / SAMPLING_RATE) * 1000.0