import torch, torchaudio
from torch.utils.data import IterableDataset
import random

from music2latent import EncoderDecoder

from utils import get_audio_files


class RandomGain(IterableDataset):
    """
    Reads in audio files, applies a random gain, encodes both versions and returns them
    """

    def __init__(
            self,
            data_dir,
            device='cpu',
            sample_rate=44100,
    ):
        self.files = get_audio_files(data_dir)
        self.encdec = EncoderDecoder()
        self.song_batch_size = 4
        self.sr = sample_rate
        self.device = device

    def __iter__(self):
        while True:
            latents_orig = torch.zeros(65,0).to(self.device)
            latents_gain = torch.zeros(64,0).to(self.device)
            gain = []
            
            for _ in range(self.song_batch_size):
                # encode the batch of songs and the randomly applied gain

                # get random audio file
                audio_file = self.files[random.randint(0, len(self.files)-1)]

                # load and resmaple
                audio, sr = torchaudio.load(audio_file)

                # only take first channel
                if audio.shape[0] > 1:
                    audio = audio[0]

                resampler = torchaudio.transforms.Resample(sr, self.sr)
                audio = resampler(audio)

                # get random gain
                gain = random.random()

                if random.random() < 0.01:
                    gain = 0
                elif random.random() < 0.2:
                    gain = gain * 5

                # apply gain
                audio_gain = gain * audio

                # encode both
                audio_latent = self.encdec.encode(audio, max_waveform_length=44100*1).squeeze().to(self.device)
                audio_gain_latent = self.encdec.encode(audio_gain, max_waveform_length=44100*1).squeeze().to(self.device)
                gain_tensor = torch.full((1,audio_latent.shape[-1]), gain).to(self.device)

                audio_latent = torch.cat([audio_latent,gain_tensor], 0) # add gain to the end

                latents_orig = torch.cat([latents_orig, audio_latent], -1)
                latents_gain = torch.cat([latents_gain, audio_gain_latent], -1)


            for i in torch.randperm(latents_orig.shape[-1]):
                yield latents_orig[:,i], latents_gain[:,i]

            # # shuffle our results...
            # indices = torch.randperm(latents_orgin.shape[-1])


            # latents_orig = latents_orig[indices]
            # latents_gain = latents_gain[indices]

            # yield 


            
            
