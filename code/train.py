import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from dataset import load_mixdata
from istft import ISTFT
from loss import sisdr_loss, wSDRLoss
from metric import calc_si_sdri, calc_snri
from model import DASA

if os.path.isfile('distributed.py'):
    from distributed import setup_cluster
    

def train(model, optimizer, train_loader, val_loader, epochs, gpu, world_size, rank, 
          nfft, hop_length, istft, savepath, loss_fun, metric):

    """
    Train a speech enhancement model. The model checkpoint with the maximum validation SI-SNRi is stored 
    in 'model_*.pth'.
    
    Args:
        model: The speech enhancement model to be trained.
        optimizer: The optimiser used.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        epochs: The number of epochs.
        gpu: The gpu used.
        world_size: The world size (world_size == 1 for single-GPU training).
        rank: The rank of the device (rank == 0 for single-GPU training).
        nfft: The FFT size for the short-time Fourier transform.
        hop_length: The hop length for the short-time Fourier transform.
        istft: The inverse STFT function.
        savepath: Path to save model checkpoints.
        loss_fun: The loss function to use.
        metric: The metric to optimize for (si-sdri or snri).
        
    Returns:
        max_metric: The maximum validation metric value.
    """
    history = {}
    history['train_loss'], history['val_si_sdri'], history['val_loss'], history['val_snri'] = [], [], [], []
    metric_key = f"val_{metric}"
    max_metric = torch.tensor(float('-inf')).to(gpu)

    for epoch in range(epochs):
        data_loader = train_loader
        model.train()

        ave_loss = torch.Tensor([0.]).to(gpu)
        data_size = torch.Tensor([0.]).to(gpu)
        # mix_stft[B,1,F,T], cln_stft, noisy_wav[B,1,T], clean_wav
        for x, mix_stft, cln_stft, noisy_wav, clean_wav in data_loader:
            x = x.to(gpu, non_blocking = True)
            mix_stft = mix_stft.to(gpu, non_blocking = True)
            cln_stft = cln_stft.to(gpu, non_blocking = True)
            noisy_wav = noisy_wav.to(gpu, non_blocking = True).squeeze(1)
            clean_wav = clean_wav.to(gpu, non_blocking = True).squeeze(1)
            noisy_mag, noisy_phase, noisy_real, noisy_imag = torch.abs(mix_stft), torch.angle(mix_stft), mix_stft.real, mix_stft.imag
            cln_mag, cln_phase, cln_real, cln_imag = torch.abs(cln_stft), torch.angle(cln_stft), cln_stft.real, cln_stft.imag
            with torch.set_grad_enabled(True):
                cRM = model(noisy_mag)
                cRM = cRM.permute(0, 2, 3, 1)
                enhanced_real = cRM[..., 0] * noisy_real.squeeze(1) - cRM[..., 1] * noisy_imag.squeeze(1)
                enhanced_imag = cRM[..., 1] * noisy_real.squeeze(1) + cRM[..., 0] * noisy_imag.squeeze(1)
                out_audio = istft(enhanced_real, enhanced_imag, 80000)   
                if loss_fun == wSDRLoss or loss_fun == sisdr_loss:
                    # wav->(B, T)
                    loss = loss_fun(noisy_wav, clean_wav, out_audio.squeeze(1))                                        
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()

                ave_loss += loss * x.size(dim = 0)
                data_size += x.size(dim = 0)

        if world_size > 1:
            dist.all_reduce(ave_loss, op = dist.ReduceOp.SUM)
            dist.all_reduce(data_size, op = dist.ReduceOp.SUM)
        ave_loss /= data_size
        history['train_loss'].append(ave_loss)
        print(ave_loss)

        # validation phase
        data_loader = val_loader
        model.eval()

        data_size = torch.Tensor([0.]).to(gpu)
        ave_si_sdri = torch.Tensor([0.]).to(gpu)
        ave_snri = torch.Tensor([0.]).to(gpu)
        ave_loss_val = torch.Tensor([0.]).to(gpu)
        current_sisdr_list = []
        for x, mix_stft, mix_wav, cln_wav, cln_stft, filename in data_loader:
            x = x.to(gpu, non_blocking = True)
            mix_stft = mix_stft.to(gpu, non_blocking = True)
            cln_stft = cln_stft.to(gpu, non_blocking = True)
            mix_wav = mix_wav.to(gpu, non_blocking = True).squeeze(1)
            cln_wav = cln_wav.to(gpu, non_blocking = True).squeeze(1)
            noisy_mag, noisy_phase, noisy_real, noisy_imag = torch.abs(mix_stft), torch.angle(mix_stft), mix_stft.real, mix_stft.imag
            cln_mag, cln_phase, cln_real, cln_imag = torch.abs(cln_stft), torch.angle(cln_stft), cln_stft.real, cln_stft.imag
            with torch.set_grad_enabled(False):
                cRM = model(noisy_mag)
                cRM = cRM.permute(0, 2, 3, 1)
                enhanced_real = cRM[..., 0] * noisy_real.squeeze(1) - cRM[..., 1] * noisy_imag.squeeze(1)
                enhanced_imag = cRM[..., 1] * noisy_real.squeeze(1) + cRM[..., 0] * noisy_imag.squeeze(1)
                out_audio = istft(enhanced_real, enhanced_imag, 80000)
                if loss_fun == wSDRLoss or loss_fun == sisdr_loss:
                    loss_val = loss_fun(mix_wav, cln_wav, out_audio.squeeze(1))
                ave_loss_val += loss_val * x.size(dim = 0)
                si_sdri = calc_si_sdri(cln_wav, out_audio.squeeze(1), mix_wav)
                snri = calc_snri(cln_wav, out_audio.squeeze(1), mix_wav)
                current_sisdr_list.extend(zip(filename, si_sdri.tolist()))
                data_size += x.size(dim = 0)    
                ave_si_sdri += torch.sum(si_sdri)
                ave_snri += torch.sum(snri)


        if world_size > 1:
            dist.all_reduce(ave_si_sdri, op = dist.ReduceOp.SUM)
            dist.all_reduce(data_size, op = dist.ReduceOp.SUM)
        ave_si_sdri /= data_size
        ave_snri /= data_size
        ave_loss_val /= data_size
        ave_metric = ave_si_sdri if metric == 'si-sdri' else ave_snri
        if ave_metric > max_metric:
            max_metric = ave_metric
            best_model_path = savepath + f"/pt/best_model_{metric}_{ave_metric.item():.4f}.pth"
            best_pth = savepath + '/pt/best.pth'
            if rank == 0:
                torch.save(model.state_dict(), best_model_path)
                torch.save(model.state_dict(), best_pth)
                # Save metric results
                with open(f'{savepath}/{epoch}_results_{metric}_{max_metric.item():.4f}.txt', 'a') as f:
                    for fname, metric_value in current_sisdr_list:  # Adjust for snri if necessary
                        f.write(f"{fname} {metric_value}\n")
        history['val_si_sdri'].append(ave_si_sdri)
        history['val_loss'].append(ave_loss_val)
        history['val_snri'].append(ave_snri)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {history['train_loss'][-1].item():.5f} - val_si_sdri: {history['val_si_sdri'][-1].item():.5f} - val_snri: {history['val_snri'][-1].item():.5f} - val_loss: {history['val_loss'][-1].item():.5f}")
            logging.info(f"Epoch {epoch + 1}/{epochs} - train_loss: {history['train_loss'][-1].item():.5f} - val_si_sdri: {history['val_si_sdri'][-1].item():.5f} - val_snri: {history['val_snri'][-1].item():.5f} - val_loss: {history['val_loss'][-1].item():.5f}")
    return max_metric

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help='The number of epochs to train the model.')
    parser.add_argument("--batch_size", type=int, default=8, help='The batch size per GPU.')
    parser.add_argument("--nfft", type=int, default=1024, help='The FFT size for the short-time Fourier transform.')
    parser.add_argument("--hop_length", type=int, default=256, help='The hop length for the short-time Fourier transform.')
    parser.add_argument('--dist', action='store_true', help='A flag indicating that distributed training is activated.')
    parser.add_argument('--prefix', type=str, help='The prefix used to identify the checkpoint files in distributed training.')
    parser.add_argument('--fix_seed', action='store_true', help='A flag to fix the seed for random number generators for reproducibility.')
    parser.add_argument("--lr_per_GPU", type=float, default=0.002, help='The learning rate for each GPU.')
    parser.add_argument("--loss", type=str, default='wSDRLoss', help='The loss function to be used in training.')
    parser.add_argument("--model", type=str, default='DASA', help='The model architecture to be used for training.')
    parser.add_argument("--data_path", type=str, default='voicebank', help='The directory path for the clean speech dataset.')
    parser.add_argument("--metric", type=str, default='snri', help='The evaluation metric to be used.')
    parser.add_argument("--save_path", type=str, default='', help='The directory path where the model and outputs will be saved.')
    parser.add_argument('--gpu', type=int, default=0, help='The index of the GPU to use for training.')
    return parser.parse_args()


def main():
    loss_functions = {
        'wSDRLoss' : wSDRLoss,
        'sisdrLoss' : sisdr_loss,
        'l1loss' : torch.nn.L1Loss(),
        'mseloss' : torch.nn.MSELoss(),
    }
    model_functions = {
        'DASA' : DASA,
    }
    args = get_args()
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    datafile = args.data_path.split('/')[-1]
    savepath = args.save_path  + datafile + '_' + time + '_' + args.loss + '_' + args.model + '_' +str(args.lr_per_GPU) + '_' + str(args.batch_size) + '_' + str(args.nfft) + '_' + str(args.hop_length) + '_' + str(args.metric)
    loss_fun= loss_functions[args.loss]
    model_args = {"nfft": args.nfft, "hop_length": args.hop_length}
    
    # Fix seeds (for debugging only)
    os.mkdir(savepath)
    os.mkdir(savepath+'/pt')
    os.mkdir(savepath+'/audio')
    log_filename = savepath + '/log.txt'
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(log_filename, 'a', 'utf-8'),
                                logging.StreamHandler()])
    # shutil.copy('train_dcunet.py', savepath + '\\train_dcunet.py')
    if args.fix_seed:
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    # Set up a distributed training environment
    if args.dist:
        rank, local_rank, world_size, gpu = setup_cluster(args.prefix)
    else:
        rank, local_rank, world_size, gpu = 0, 0, 1, torch.device(f'cuda:{args.gpu}')

    if args.dist:
        if rank == 0:
            logging.info(args)
    else:
        logging.info(args)
    
    # Load_data
    if rank == 0:
        logging.info('loading data ...')
    train_loader, val_loader, test_loader, _\
    = load_mixdata(args.batch_size, world_size, rank, args.nfft, args.hop_length, args.data_path)
    
    model = model_functions[args.model](**model_args).to(gpu)
    print(model)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank])

    # Define the optimizer
    lr = args.lr_per_GPU * world_size
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    istft = ISTFT(args.nfft, args.hop_length, window='hamming').to(gpu)
    metric = args.metric
    # Training
    if rank == 0:
        logging.info(f"training ...")

    val_si_sdri = train(model, optimizer, train_loader, val_loader, 
                        args.epochs, gpu, world_size, rank, args.nfft, args.hop_length, istft, savepath, loss_fun, metric)
    
    
    # Display results
    if rank == 0:
        logging.info('========================================')
        logging.info(args)
        logging.info('validation SI-SNRi: ' + str(val_si_sdri.item()) + 'dB')
        logging.info('========================================')

    # Kill distributed training environment (for the multi-GPU case only)
    if world_size > 1:
        time.sleep(10); dist.destroy_process_group()


if __name__ == '__main__':
    main()
