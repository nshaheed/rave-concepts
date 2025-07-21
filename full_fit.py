import os
import re

import torch, torchaudio
import rave
import tqdm
import gin

from absl import app, flags, logging
from torch.autograd import Variable
import cached_conv as cc
import numpy as np

# Take already processed latents and RMS values and perform a linear regression
# TODO output to a file?
# TODO decaying lr?

FLAGS = flags.FLAGS
flags.DEFINE_integer('gpu', default=-1, help='GPU to use')
flags.DEFINE_string('out_path', 'generations', help="output path")

# size of latent in samples
LATENT_SIZE = 2048

def get_audio_files(path):
    audio_files = []

    # these are hard coded for now, need to find a way to get this
    # without torchaudio.get_audio_backend() as it's been deprecated
    # and returns None
    # valid_exts = rave.core.get_valid_extensions()
    valid_exts = ['.wav', '.flac', '.ogg', '.aiff', '.aif', '.aifc']
    for root, _, files in os.walk(path):
        valid_files = list(filter(lambda x: os.path.splitext(x)[1] in valid_exts, files))
        audio_files.extend([(path, os.path.join(root, f)) for f in valid_files])
    return audio_files


def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(3)

    # device
    if FLAGS.gpu >= 0:
        device = torch.device('cuda:%d'%FLAGS.gpu)
    else:
        device = torch.device('cpu')

    cc.MAX_BATCH_SIZE = 8

    # define linear regression model
    learning_rate = 0.0001

    linear_regression = torch.nn.Linear(128, 1, bias=False).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(linear_regression.parameters(), lr=learning_rate)

    epochs = 30000

    # TODO make these flags
    latents = torch.Tensor(np.load('processed/latents_large.npy'))
    rmss = torch.Tensor(np.load('processed/rms_large.npy'))

    epoch_progress_bar = tqdm.tqdm(range(epochs), leave=True)
    for epoch in epoch_progress_bar:
        # linear regression
        inputs = latents.squeeze(0).transpose(0,1)
        labels = rmss.unsqueeze(0).transpose(0,1)

        # breakpoint()
        outputs = linear_regression(inputs)
        loss = criterion(outputs, labels)

        # clear gradients between epochs
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_progress_bar.set_description(f'loss: {loss.item():.5f}')

    print(f'{linear_regression.weight=}')
    print(f'{linear_regression.bias=}')

if __name__ == "__main__":
    app.run(main)
