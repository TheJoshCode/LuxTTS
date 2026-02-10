import soundfile as sf

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional
import gc
import psutil

import numpy as np
import torch
import librosa
import torchaudio
import inspect
from transformers import pipeline
from huggingface_hub import snapshot_download
from lhotse.utils import fix_random_seed

from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm

from dataclasses import dataclass, field
from typing import Optional, List

from linacodec.vocoder.vocos import Vocos
from zipvoice.onnx_modeling import OnnxModel
from torch.nn.utils import parametrize


class LuxTTS:
    def __init__(self, model_path='YatharthS/LuxTTS', device='cuda', threads=4):
        if device == 'cpu':
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu(model_path, threads)
        else:
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_gpu(model_path, device=device)

        self.model = model
        self.feature_extractor = feature_extractor
        # Move the internal MelSpectrogram to the correct device
        self.feature_extractor.fbank = self.feature_extractor.fbank.to(device)
        self.vocos = vocos
        self.tokenizer = tokenizer
        self.transcriber = transcriber
        self.device = device
        self.vocos.freq_range = 12000
        
        # Keep models in float32 to avoid dtype issues
        # Rely on aggressive cache clearing for VRAM management
        if device == 'cuda':
            torch.cuda.empty_cache()

    def encode_prompt(self, prompt_text, prompt_audio, duration=4, rms=0.1):
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(
            prompt_audio, prompt_text, self.tokenizer, self.feature_extractor, self.device, target_rms=rms, duration=duration
        )
        return {
            "prompt_tokens": prompt_tokens, 
            'prompt_features_lens': prompt_features_lens, 
            'prompt_features': prompt_features, 
            'prompt_rms': prompt_rms
        }

    def _read_audio(self, audio_path):
        """Read audio file and return audio data and sample rate"""
        return sf.read(audio_path)

    def generate_speech(self, text, encode_dict, num_steps=4, guidance_scale=3.0, t_shift=0.5, speed=1.0):
        # Unpack dict explicitly to ensure compatibility
        return generate(
            **encode_dict,
            text=text,
            model=self.model,
            vocoder=self.vocos,
            tokenizer=self.tokenizer,
            num_step=num_steps,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            speed=speed
        )

@dataclass
class LuxTTSConfig:
    model_dir: Optional[str] = None
    checkpoint_name: str = "model.pt"
    vocoder_path: Optional[str] = None
    trt_engine_path: Optional[str] = None
    tokenizer: str = "emilia"
    lang: str = "en-us"


@torch.inference_mode()
def process_audio(audio, prompt_text, tokenizer, feature_extractor, device, target_rms=0.1, duration=4, feat_scale=0.1):
    prompt_wav, sr = librosa.load(audio, sr=24000, duration=duration)
    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0).to(device)
    
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=24000).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    
    # Clear cache after processing
    is_cuda = isinstance(device, str) and device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')
    if is_cuda:
        torch.cuda.empty_cache()
    
    return prompt_tokens, prompt_features_lens, prompt_features, prompt_rms

@torch.inference_mode()
def generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1):
    device = prompt_features.device
    tokens = tokenizer.texts_to_token_ids([text])
    
    (pred_features, _, _, _) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed * 1.3,
        t_shift=t_shift,
        duration='predict',
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    pred_features = pred_features.permute(0, 2, 1) / 0.1
    
    # Decode with vocoder
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)
    
    # Clear intermediate tensors
    del pred_features
    is_cuda = isinstance(device, str) and device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')
    if is_cuda:
        torch.cuda.empty_cache()

    return wav

def load_models_gpu(model_path=None, device="cuda"):
    if model_path is None or not os.path.exists(str(model_path)):
        model_path = snapshot_download("YatharthS/LuxTTS")

    token_file = os.path.join(model_path, "tokens.txt")
    model_ckpt = os.path.join(model_path, "model.pt")
    model_config_path = os.path.join(model_path, "config.json")

    tokenizer = EmiliaTokenizer(token_file=token_file)

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    # Some releases wrap actual model kwargs under a top-level 'model' key.
    if isinstance(model_config, dict) and "model" in model_config and isinstance(model_config["model"], dict):
        # prefer nested model dict (it contains the expected fm_decoder_* keys)
        model_config = model_config["model"]

    # model_config is a dict loaded from config.json; pass only supported kwargs
    # Some configs include extra top-level keys (e.g., 'model') that are not
    # ZipVoiceDistill forwards kwargs to ZipVoice, so inspect ZipVoice.__init__
    sig = inspect.signature(ZipVoice.__init__)
    valid_keys = {p for p in sig.parameters.keys() if p not in ("self",)}
    filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}
    dropped = set(model_config.keys()) - set(filtered_config.keys())
    if dropped:
        logging.warning(f"Dropped unexpected model config keys when constructing ZipVoiceDistill: {sorted(dropped)}")

    model = ZipVoiceDistill(
        **filtered_config,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )
    load_checkpoint(filename=model_ckpt, model=model, strict=True)
    model = model.to(device).eval()

    feature_extractor = VocosFbank()

    # Try several likely locations for the vocoder config and weights
    vocoder_cfg_candidates = [
        os.path.join(model_path, "vocoder_config.yaml"),
        os.path.join(model_path, "vocoder", "config.yaml"),
        os.path.join(model_path, "vocoder", "vocoder_config.yaml"),
        os.path.join(model_path, "vocoder", "vocos_config.yaml"),
    ]
    vocoder_cfg_path = next((p for p in vocoder_cfg_candidates if os.path.exists(p)), None)
    if vocoder_cfg_path is None:
        raise FileNotFoundError(
            f"Vocoder config not found. Tried: {vocoder_cfg_candidates}"
        )

    vocos = Vocos.from_hparams(vocoder_cfg_path)

    # IMPORTANT: remove parametrizations before loading the plain-weight state dict
    try:
        if hasattr(vocos.upsampler.upsample_layers[0], "parametrizations"):
            parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
        if hasattr(vocos.upsampler.upsample_layers[1], "parametrizations"):
            parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    except Exception:
        # If structure differs, ignore and continue; load_state_dict will surface issues
        pass

    vocoder_bin_candidates = [
        os.path.join(model_path, "vocos.bin"),
        os.path.join(model_path, "vocoder", "vocos.bin"),
        os.path.join(model_path, "vocoder", "vocos.pt"),
    ]
    vocoder_bin_path = next((p for p in vocoder_bin_candidates if os.path.exists(p)), None)
    if vocoder_bin_path is None:
        raise FileNotFoundError(
            f"Vocoder weights not found. Tried: {vocoder_bin_candidates}"
        )

    state_dict = torch.load(vocoder_bin_path, map_location=device)
    vocos.load_state_dict(state_dict)

    vocos = vocos.to(device).eval()

    return model, feature_extractor, vocos, tokenizer, None


def load_models_cpu(model_path=None, num_thread=2):
    if model_path is None or not os.path.exists(str(model_path)):
        model_path = snapshot_download('YatharthS/LuxTTS')

    tokenizer = EmiliaTokenizer(token_file=os.path.join(model_path, "tokens.txt"))
    model = OnnxModel(os.path.join(model_path, "text_encoder.onnx"), 
                      os.path.join(model_path, "fm_decoder.onnx"), num_thread=num_thread)

    vocos = Vocos.from_hparams(os.path.join(model_path, 'vocoder/config.yaml')).eval()
    vocos.load_state_dict(torch.load(os.path.join(model_path, 'vocoder/vocos.bin'), map_location='cpu'))
    
    # Parametrizations usually not needed for ONNX/CPU path but added for safety
    if hasattr(vocos.upsampler.upsample_layers[0], 'parametrizations'):
        parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
        parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")

    return model, VocosFbank(), vocos, tokenizer, ""

@torch.inference_mode()
def generate_cpu(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocos, tokenizer, num_step=4, guidance_scale=3.0, t_shift=0.5, speed=1.0, target_rms=0.1):
    tokens = tokenizer.texts_to_token_ids([text])
    pred_features = model.sample(tokens=tokens, prompt_tokens=prompt_tokens, prompt_features=prompt_features, 
                                 prompt_features_lens=prompt_features_lens, speed=speed * 1.3, 
                                 t_shift=t_shift, num_step=num_step, guidance_scale=guidance_scale)
    pred_features = torch.from_numpy(pred_features).permute(0, 2, 1) / 0.1
    wav = vocos.decode(pred_features).squeeze(1).clamp(-1, 1)
    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)
    return wav.cpu()

def get_memory_info():
    proc = psutil.Process()
    mem_info = proc.memory_info()
    virtual_mem = psutil.virtual_memory()
    return {
        'process_rss_mb': mem_info.rss / (1024*1024),
        'system_available_mb': virtual_mem.available / (1024*1024),
        'system_percent': virtual_mem.percent
    }

def clear_memory(verbose=None):
    if verbose:
        verbose("🧹 Running memory cleanup...")
    collected = gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except:
        pass
    if verbose:
        mem = get_memory_info()
        verbose(f"   Collected {collected} objects | RAM: {mem['process_rss_mb']:.1f}MB | Available: {mem['system_available_mb']:.1f}MB")

def unload_model():
    """Placeholder for unload model functionality"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()