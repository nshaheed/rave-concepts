import os


def get_audio_files(path):
    audio_files = []

    # these are hard coded for now, need to find a way to get this
    # without torchaudio.get_audio_backend() as it's been deprecated
    # and returns None
    # valid_exts = rave.core.get_valid_extensions()
    valid_exts = [".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".mp3"]

    for root, _, files in os.walk(path):
        valid_files = list(
            filter(lambda x: os.path.splitext(x)[1] in valid_exts, files)
        )
        audio_files.extend([os.path.join(root, f) for f in valid_files])
    return audio_files
