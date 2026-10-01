import hashlib
import logging
import os
import tempfile

import torch
import torchaudio

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".wav", ".mp3", ".flac"}

_RESAMPLE_KW = dict(
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="sinc_interp_kaiser",
    beta=14.769656459379492,
)


def resample(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, sr_in, sr_out, **_RESAMPLE_KW)


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _has_ffmpeg() -> bool:
    try:
        import ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


def decode_to_wav_cache(src: str, cache_dir: str) -> str:
    """WAV inputs are returned as-is. Other formats are decoded once to a native-rate
    float32 WAV in cache_dir, keyed by the source md5. No resampling happens here."""
    if os.path.splitext(src)[1].lower() == ".wav":
        return src
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, f"{file_md5(src)}.wav")
    if os.path.isfile(dst):
        return dst

    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=cache_dir)
    os.close(fd)
    try:
        if _has_ffmpeg():
            import ffmpeg
            try:
                (ffmpeg.input(src)
                 .output(tmp, format="wav", acodec="pcm_f32le")
                 .overwrite_output()
                 .run(capture_stdout=True, capture_stderr=True))
            except ffmpeg.Error as e:
                msg = e.stderr.decode(errors="replace") if e.stderr else ""
                raise RuntimeError(f"ffmpeg failed to decode {src}:\n{msg}") from e
        else:
            log.warning("ffmpeg-python not installed; decoding %s with torchaudio", src)
            wav, sr = torchaudio.load(src)
            torchaudio.save(tmp, wav.float(), sr, encoding="PCM_F", bits_per_sample=32)
        os.replace(tmp, dst)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    log.info("decoded %s -> %s", os.path.basename(src), os.path.basename(dst))
    return dst


def load_audio(path: str, sr: int, cache_dir: str, trim_samples: int = 0) -> torch.Tensor:
    """Decode, trim trim_samples at the file's own rate, resample to sr, return (C<=2, N) float32."""
    wav_path = decode_to_wav_cache(path, cache_dir)
    wav, file_sr = torchaudio.load(wav_path, frame_offset=max(0, trim_samples))
    wav = wav.float()
    if wav.shape[0] > 2:
        wav = wav[:2]
    return resample(wav, file_sr, sr)


def save_wav_f32(tensor: torch.Tensor, path: str, sr: int) -> None:
    torchaudio.save(path, tensor.float().cpu(), sr, encoding="PCM_F", bits_per_sample=32)
