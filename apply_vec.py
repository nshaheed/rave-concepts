import os
import gin
import torch, torchaudio
import rave
import tqdm
from absl import app, flags, logging
from torch.linalg import vector_norm

FLAGS = flags.FLAGS
flags.DEFINE_string('model', required=True, default=None, help="model path")
flags.DEFINE_string('input', required=True, default=None, help="audio file path")
flags.DEFINE_integer('gpu', default=-1, help='GPU to use')
flags.DEFINE_string('out_path', 'generations', help="output path")

# size of latent in samples
LATENT_SIZE = 2048

# all 128 dims
rcv = torch.Tensor(
    [-5.5358e-05,  2.1329e-05,  2.2820e-05,  3.1298e-06,  6.6115e-05,
      8.6198e-06, -5.6105e-05, -2.6582e-05,  7.5841e-04, -6.6450e-07,
      5.2164e-06, -3.9819e-05, -5.1730e-05,  3.5145e-05,  9.2621e-06,
      -1.5673e-03,  3.4478e-05, -3.9946e-05,  3.9025e-05, -1.0954e-02,
      1.2616e-05, -3.4744e-05,  3.0606e-05, -2.5453e-05,  1.7030e-03,
      4.7518e-05, -1.2996e-05,  3.1672e-05,  4.6332e-05,  8.9784e-06,
      -2.9503e-05, -8.8635e-06,  1.6687e-05,  3.7655e-05,  1.4768e-05,
      -1.4187e-05, -1.4075e-05, -3.1623e-05, -3.6107e-05, -3.4688e-05,
      -2.6609e-04,  9.8918e-03,  8.8907e-06,  2.4866e-06,  2.6090e-05,
      -3.9745e-05, -4.8370e-05, -5.1097e-05, -8.6250e-06, -1.4050e-05,
      4.0254e-05, -1.3478e-05, -2.9538e-05,  3.6081e-05, -2.5273e-05,
      -9.2807e-05, -3.8926e-05, -1.6963e-05, -5.4906e-05,  1.9656e-05,
      -6.6501e-06,  8.6115e-06, -5.5972e-06, -6.5807e-05, -2.7937e-05,
      1.9654e-05, -4.7336e-05,  3.5231e-05, -4.2058e-05, -4.0343e-05,
      -8.6926e-04,  1.5279e-05, -2.2864e-05, -9.8116e-05, -3.6277e-05,
      -2.0459e-06, -2.5387e-05,  2.2265e-05, -4.1548e-06,  2.1717e-03,
      2.7144e-05, -1.1626e-05, -4.7093e-05, -2.4317e-05,  1.5199e-05,
      -1.9475e-05, -1.5643e-05,  5.0602e-06, -3.6860e-05, -4.3569e-05,
      2.7850e-05,  2.9772e-05, -1.9171e-05,  1.6531e-05,  2.1227e-05,
      -9.6470e-06,  1.1843e-06,  5.1443e-05, -4.2397e-05,  2.4255e-05,
      -3.4355e-05, -4.6045e-05, -2.3152e-05,  1.0931e-05, -2.7845e-05,
      2.3636e-05,  2.6532e-05,  3.8675e-05,  9.2530e-06,  1.7953e-05,
      -1.5784e-03,  2.4700e-05,  1.8246e-05,  3.9034e-06, -1.5184e-05,
      -2.5133e-05,  3.1023e-05,  1.4141e-05,  1.1337e-06, -3.4368e-05,
      1.9716e-06, -1.3541e-05,  1.1061e-05,  3.7955e-05, -1.9805e-05,
      2.8372e-05, -3.6336e-05, -2.8022e-06])

rcv = (1 / vector_norm(rcv)) * rcv

def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(3)

    model_path = FLAGS.model
    path = FLAGS.input

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

    # device
    if FLAGS.gpu >= 0:
        device = torch.device('cuda:%d'%FLAGS.gpu)
        model = model.to(device)
    else:
        device = torch.device('cpu')


    # parse inputs
    audio_file = FLAGS.input
    receptive_field = rave.core.get_minimum_size(model)

    logging.info('processing audio...')
    with torch.no_grad():
        try:
            x, sr = torchaudio.load(audio_file)
        except:
            logging.warning('could not open file %s.'%f)
            return

        # load file
        if sr != model.sr:
            x = torchaudio.functional.resample(x, sr, model.sr)
        if model.n_channels != x.shape[0]:
            if model.n_channels < x.shape[0]:
                x = x[:model.n_channels]
            else:
                print('[Warning] file %s has %d channels, butt model has %d channels ; skipping'%(f, model.n_channels))
                x = x.to(device)

        latent = model.encode(x[None], return_mb=False).detach()
        latent = model.encoder.reparametrize(latent)[0]
        env = torch.linspace(0, 280, latent.shape[-1])
        env = torch.logspace(start=0.01, end=1.0, steps=latent.shape[-1], base=280) - 1
        # go towards silence
        # env = (torch.logspace(start=0.01, end=1.0, steps=latent.shape[-1], base=280) - 1) * -0.01
        # env = torch.logspace(start=0.01, end=1.0, steps=latent.shape[-1], base=7) - 1

        # breakpoint()

        # apply concept envelope
        # TODO make this more efficient if I need to
        for i in range(latent.shape[-1]):
            latent[0,:,i] = latent[0,:,i] + (env[i] * rcv)


        # breakpoint()
        out = model.decode(latent)

        logging.info('saving audio...')
        torchaudio.save('./output.wav', out[0].cpu(), sample_rate=model.sr)

if __name__ == "__main__":
    app.run(main)
