import os
from pathlib import Path

import librosa
import torch
import torchaudio


class mix_data(torch.utils.data.Dataset):
    def __init__(self, partition, nfft, hop_length, root_path):
        super().__init__()
        self.partition = partition
        self.nfft = nfft
        self.hop_length = hop_length
        self.root_path = Path(root_path).expanduser().absolute()
        self.clean_path = self.root_path / self.partition / 'clean'
        self.noisy_path = self.root_path / self.partition / 'noisy'

        self.noisy_files_list = librosa.util.find_files(self.noisy_path.as_posix())
        self.length = len(self.noisy_files_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_file_path = self.noisy_files_list[item]
        noisy_filename, _ = os.path.splitext(os.path.basename(noisy_file_path))
        clean_file_path = noisy_file_path.replace(str(self.noisy_path), str(self.clean_path))

        noisy_wav, _ = torchaudio.load(os.path.abspath(os.path.expanduser(noisy_file_path)))
        clean_wav, _ = torchaudio.load(os.path.abspath(os.path.expanduser(clean_file_path)))
        noisy_stft = torch.stft(input = noisy_wav,
                                  n_fft = self.nfft,
                                  hop_length = self.hop_length,
                                  window = torch.hamming_window(self.nfft),
                                  return_complex = True)
        clean_stft = torch.stft(input = clean_wav,
                                n_fft = self.nfft,
                                hop_length = self.hop_length,
                                window = torch.hamming_window(self.nfft),
                                return_complex = True)
        feat = torch.abs(noisy_stft)
        if self.partition == 'train':
            return feat, noisy_stft, clean_stft, noisy_wav, clean_wav
        else:
            return feat, noisy_stft, noisy_wav, clean_wav, clean_stft, noisy_filename


    
def load_mixdata(max_batch_size, world_size, rank, nfft, hop_length, data_path):
    train_data = mix_data('train', nfft, hop_length, data_path)
    val_data = mix_data('test', nfft, hop_length, data_path)
    test_data = mix_data('test', nfft, hop_length, data_path)

    train_batch_size = min(len(train_data) // world_size, max_batch_size)
    val_batch_size = min(len(val_data) // world_size, max_batch_size)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas = world_size,
        rank = rank,
        shuffle = True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data,
        num_replicas = world_size,
        rank = rank,
        shuffle = False)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = train_batch_size,
                                               shuffle = False,
                                               num_workers = 8,
                                               pin_memory = True,
                                               sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size = val_batch_size,
                                             shuffle = False,
                                             num_workers = 8,
                                             pin_memory = True,
                                             sampler = val_sampler)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size = 1,
                                              shuffle = False,
                                              num_workers = 2,
                                              pin_memory = True)
 
    return train_loader, val_loader, test_loader, train_batch_size
