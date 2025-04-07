import torch


@torch.no_grad()
def si_sdr(true, est):

    para = (torch.sum(true * est, 1, keepdim = True) / torch.sum(true ** 2, 1, keepdim = True)) * true
    num = torch.sum(para ** 2, 1)
    dnm = torch.sum((est - para) ** 2, 1)

    return 10 * torch.log10(num / dnm)


@torch.no_grad()
def calc_si_sdri(true, est, mix):
    # print("est:",si_sdr(true, est),"   ori:",si_sdr(true, mix))
    return si_sdr(true, est) - si_sdr(true, mix)

@torch.no_grad()
def snr(true, est):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    """
    signal_power = torch.sum(true ** 2, 1)
    noise_power = torch.sum((true - est) ** 2, 1)
    return 10 * torch.log10(signal_power / noise_power)

@torch.no_grad()
def calc_snri(true, est, mix):
    """
    Calculate the improvement in SNR.
    """
    return snr(true, est) - snr(true, mix)