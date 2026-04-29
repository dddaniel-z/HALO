import os
import random
import re
from typing import Optional, Sequence

import pandas as pd
import soundfile as sf
import torch
from torch.utils import data


class MyDataset(data.Dataset):
    """Dataset that pairs <noisy, clean> wavs by filename and optionally
    subsamples the full list according to a CSV file containing indices.
    """

    def __init__(
        self,
        noisy_folder: str,
        clean_folder: str,
        csv_path: Optional[str] = None,
        shuffle: bool = True,
        train_sample_epoch: bool = True,
        num_data_per_epoch: int = 10000,
        num_tot: int = 0,
        wav_len: int = 0,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ) -> None:
        super().__init__()
        self.noisy_folder = noisy_folder
        self.clean_folder = clean_folder
        self.wav_len = wav_len
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.train_sample_epoch = train_sample_epoch
        self.num_data_per_epoch = num_data_per_epoch


        cut_suffix = lambda fn: re.sub(r"_(?:noisy|clean)\.wav$", "", fn)
        noisy_map = {
            cut_suffix(f): f for f in os.listdir(noisy_folder) if f.endswith(".wav")
        }
        clean_map = {
            cut_suffix(f): f for f in os.listdir(clean_folder) if f.endswith(".wav")
        }

        common_keys: Sequence[str] = sorted(set(noisy_map) & set(clean_map))
        self.pairs: list[tuple[str, str]] = [
            (noisy_map[k], clean_map[k]) for k in common_keys
        ]
        

        # --------------------
        if csv_path is not None and os.path.isfile(csv_path):
            idx_series = pd.read_csv(csv_path, header=None).iloc[:, 0]
            # cast to Python ints & remove out‑of‑range values
            valid_idx = [int(i) for i in idx_series if 0 <= int(i) < len(self.pairs)]
            # keep original order as in CSV (no additional shuffling)
            self.pairs = [self.pairs[i] for i in valid_idx]


        # --------------------
        if shuffle:
            random.seed(7)
            random.shuffle(self.pairs)
        if num_tot:
            self.pairs = self.pairs[:num_tot]
        self._all_pairs = self.pairs.copy()

    # ------------------------------------------------------------
    def _wav_to_complex_spec(self, wav: torch.Tensor) -> torch.Tensor:
        win = torch.hann_window(self.win_length).sqrt()
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=win,
            return_complex=False,
        )

    def sample_data_per_epoch(self):
        if self.train_sample_epoch:
            self.pairs = random.sample(self._all_pairs, self.num_data_per_epoch)
        else:
            self.pairs = self._all_pairs.copy()


    def __getitem__(self, idx: int):
        noisy_fn, clean_fn = self.pairs[idx]
        noisy_wav, fs = sf.read(os.path.join(self.noisy_folder, noisy_fn), dtype="float32")
        clean_wav, _ = sf.read(os.path.join(self.clean_folder, clean_fn), dtype="float32")

        noisy = torch.tensor(noisy_wav)
        clean = torch.tensor(clean_wav)

        return self._wav_to_complex_spec(noisy), self._wav_to_complex_spec(clean)

    def __len__(self) -> int:
        return len(self.pairs)


if __name__ == "__main__":
    import toml
    from tqdm import tqdm

    cfg = toml.load("config.toml")
    ds = MyDataset(**cfg["train_dataset"], **cfg["FFT"])
    dl = data.DataLoader(ds, **cfg["train_dataloader"])
    for noisy, clean in tqdm(dl):
        print(noisy.shape, clean.shape)
        break
