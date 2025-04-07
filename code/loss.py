import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve


def sisdr_loss(mix, clean, est, eps=1e-8):
    scale = torch.sum(clean * est, dim=1, keepdim=True) / (torch.sum(clean ** 2, dim=1, keepdim=True) + eps)
    scale_clean = scale * clean
    noise = est - scale_clean
    sisdr = 10 * torch.log10(torch.sum(scale_clean ** 2, dim=1) / (torch.sum(noise ** 2,dim=1) + eps))
    return -sisdr.mean()

def sa_loss(yhat, mix_stft, cln_stft):

    """
    The signal approximation loss for speech enhancement.
    
    Args:
        yhat: The output of the convolutional neural network before the non-linear activation function.
        mix_stft: The noisy speech in the short-time Fourier transform domain.
        cln_stft: The clean speech in the short-time Fourier transform domain.
    
    Returns:
        The signal approximation loss.
    """

    # return torch.mean((F.sigmoid(yhat) * torch.abs(mix_stft) - torch.abs(cln_stft)) ** 2)
    return torch.mean((yhat * torch.abs(mix_stft) - torch.abs(cln_stft)) ** 2)


# def l1_loss(yhat, mix_stft, cln_stft):

#     """
#     The signal approximation loss for speech enhancement.
    
#     Args:
#         yhat: The output of the convolutional neural network before the non-linear activation function.
#         mix_stft: The noisy speech in the short-time Fourier transform domain.
#         cln_stft: The clean speech in the short-time Fourier transform domain.
    
#     Returns:
#         The signal approximation loss.
#     """

#     # return torch.mean((F.sigmoid(yhat) * torch.abs(mix_stft) - torch.abs(cln_stft)) ** 2)
#     return torch.mean(torch.abs(yhat * torch.abs(mix_stft) - torch.abs(cln_stft)))

def l1_loss(real_cln, imag_cln, out_real, out_imag):
    return torch.mean(torch.abs(real_cln - out_real) + torch.abs(imag_cln - out_imag))

def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
        bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
        def mSDRLoss(orig, est):
            # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
            # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
            #  > Maximize Correlation while producing minimum energy output.
            correlation = bsum(orig * est)
            energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
            return -(correlation / (energies + eps))

        noise = mixed - clean
        noise_est = mixed - clean_est

        a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
        wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
        return torch.mean(wSDR)