import os
import re

import torch, torchaudio
import rave
import tqdm
import gin

from absl import app, flags, logging
import cached_conv as cc

FLAGS = flags.FLAGS
flags.DEFINE_string('model', required=True, default=None, help="model path")
flags.DEFINE_multi_string('input', required=True, default=None, help="model inputs (file or folder)")
flags.DEFINE_integer('gpu', default=-1, help='GPU to use')
flags.DEFINE_string('out_path', 'generations', help="output path")

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
    
    model_path = FLAGS.model
    paths = FLAGS.input

    # load model
    logging.info("building RAVE model")

    is_scripted = False
    if not os.path.exists(model_path):
        logging.error('path %s does not seem to exist.'%model_path)
        exit()
    if os.path.splitext(model_path)[1] == ".ts":
        model = torch.jit.load(model_path)
        is_scripted = True
    else:
        config_path = rave.core.search_for_config(model_path)
        if config_path is None:
            logging.error('config not found in folder %s'%model_path)
        gin.parse_config_file(config_path)
        model = rave.RAVE()
        run = rave.core.search_for_run(model_path)
        if run is None:
            logging.error("run not found in folder %s"%model_path)
        model = model.load_from_checkpoint(run)

    # logging.info(f'{model=}')

    # device
    if FLAGS.gpu >= 0:
        device = torch.device('cuda:%d'%FLAGS.gpu)
        model = model.to(device)
    else:
        device = torch.device('cpu')


    # parse inputs
    audio_files = sum([get_audio_files(f) for f in paths], [])
    receptive_field = rave.core.get_minimum_size(model)

    # logging.info(f'{audio_files=}')


    progress_bar = tqdm.tqdm(audio_files)
    cc.MAX_BATCH_SIZE = 8

    for i, (d, f) in enumerate(progress_bar):
        #TODO reset cache
            
        try:
            x, sr = torchaudio.load(f)
        except: 
            logging.warning('could not open file %s.'%f)
            continue
        progress_bar.set_description(f)

        # load file
        if sr != model.sr:
            x = torchaudio.functional.resample(x, sr, model.sr)
        if model.n_channels != x.shape[0]:
            if model.n_channels < x.shape[0]:
                x = x[:model.n_channels]
            else:
                print('[Warning] file %s has %d channels, butt model has %d channels ; skipping'%(f, model.n_channels))
        x = x.to(device)

        # forward out model (TODO break into encode and decode)
        breakpoint()
        out = model.forward(x[None]).detach()

        latents = model.encode(x[None], return_mb=False).detach()
        latents = model.encoder.reparametrize(latents)[0]
        audio = model.decode(latents)

        # the difference in length of the input vs the forwarded audio
        delta = out.shape[-1] - x.shape[-1]
        # logging.info(f'{delta=}')

        # breakpoint()
        out = out[:,:,:delta]

        # breakpoint()

        # save file
        out_path = re.sub(d, "", f)
        out_path = os.path.join(FLAGS.out_path, f)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torchaudio.save(out_path, out[0].cpu(), sample_rate=model.sr)


    



if __name__ == "__main__":
    app.run(main)
