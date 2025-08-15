import os
import re

import torch, torchaudio
import rave
import tqdm
import gin

from absl import app, flags, logging
from torch.autograd import Variable
import cached_conv as cc

FLAGS = flags.FLAGS
flags.DEFINE_string("model", required=True, default=None, help="model path")
flags.DEFINE_multi_string(
    "input", required=True, default=None, help="model inputs (file or folder)"
)
flags.DEFINE_integer("gpu", default=-1, help="GPU to use")
flags.DEFINE_string("out_path", "generations", help="output path")

# size of latent in samples
LATENT_SIZE = 2048


def get_audio_files(path):
    audio_files = []

    # these are hard coded for now, need to find a way to get this
    # without torchaudio.get_audio_backend() as it's been deprecated
    # and returns None
    # valid_exts = rave.core.get_valid_extensions()
    valid_exts = [".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc"]
    for root, _, files in os.walk(path):
        valid_files = list(
            filter(lambda x: os.path.splitext(x)[1] in valid_exts, files)
        )
        audio_files.extend([(path, os.path.join(root, f)) for f in valid_files])
    return audio_files


def main(argv):
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(3)

    model_path = FLAGS.model
    paths = FLAGS.input

    # load model
    logging.info("building RAVE model")

    is_scripted = False
    if not os.path.exists(model_path):
        logging.error("path %s does not seem to exist." % model_path)
        exit()
    if os.path.splitext(model_path)[1] == ".ts":
        model = torch.jit.load(model_path)
        is_scripted = True
    else:
        config_path = rave.core.search_for_config(model_path)
        if config_path is None:
            logging.error("config not found in folder %s" % model_path)
        gin.parse_config_file(config_path)
        model = rave.RAVE()
        run = rave.core.search_for_run(model_path)
        if run is None:
            logging.error("run not found in folder %s" % model_path)
        model = model.load_from_checkpoint(run)

    # logging.info(f'{model=}')

    # device
    if FLAGS.gpu >= 0:
        device = torch.device("cuda:%d" % FLAGS.gpu)
        model = model.to(device)
    else:
        device = torch.device("cpu")

    # parse inputs
    audio_files = sum([get_audio_files(f) for f in paths], [])
    receptive_field = rave.core.get_minimum_size(model)

    # logging.info(f'{audio_files=}')

    cc.MAX_BATCH_SIZE = 8

    # define linear regression model
    learning_rate = 0.001

    linear_regression = torch.nn.Linear(128, 1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 1000

    latents = None
    out = None

    epoch_progress_bar = tqdm.tqdm(range(epochs), leave=True)
    for epoch in epoch_progress_bar:
        total_loss = 0
        progress_bar = tqdm.tqdm(audio_files, leave=False)

        for i, (d, f) in enumerate(progress_bar):
            linear_regression.train()

            try:
                x, sr = torchaudio.load(f)
            except:
                logging.warning("could not open file %s." % f)
                continue

            # load file
            if sr != model.sr:
                x = torchaudio.functional.resample(x, sr, model.sr)
            if model.n_channels != x.shape[0]:
                if model.n_channels < x.shape[0]:
                    x = x[: model.n_channels]
                else:
                    print(
                        "[Warning] file %s has %d channels, butt model has %d channels ; skipping"
                        % (f, model.n_channels)
                    )
                    x = x.to(device)

            if True:
                latents = model.encode(x[None], return_mb=False).detach()
                latents = model.encoder.reparametrize(latents)[0]
                out = model.decode(latents)

            # calculate RMS
            chunks = out.reshape(
                latents.shape[-1], -1
            )  # resize in terms of how many latents there are

            # rms = (chunks.square().sum(-2) / LATENT_SIZE).sqrt()
            rms = (chunks.square().sum(-1) / LATENT_SIZE).sqrt()

            # linear regression
            inputs = Variable(latents).squeeze(0).transpose(0, 1)
            labels = Variable(rms.unsqueeze(0).transpose(0, 1))

            outputs = linear_regression(inputs)
            loss = criterion(outputs, labels)

            # clear gradients between epochs
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            progress_bar.set_description(f"loss: {loss:.5f}")

            total_loss += loss.item()

        epoch_progress_bar.set_description(f"total loss: {total_loss:.5f}")


if __name__ == "__main__":
    app.run(main)
