from dataclasses import dataclass, field
from typing import List

import torch
import torchaudio


@dataclass
class CNNFeaturesSetting:
    frontend_algorithm: List[str]= field(default_factory=lambda: ["mfcc"])
    use_spectrogram: bool = True
