import torch
from zipvoice.modeling_utils import process_audio, generate, load_models_gpu, load_models_cpu
from zipvoice.onnx_modeling import generate_cpu

class LuxTTS:
    """
    LuxTTS class for encoding prompt and generating speech on cpu/cuda/mps.
    Supports manual transcription text - no automatic transcription is performed.
    """

    def __init__(self, model_path='YatharthS/LuxTTS', device='cuda', threads=4):
        if model_path == 'YatharthS/LuxTTS':
            model_path = None

        # Auto-detect better device if cuda is requested but not available
        if device == 'cuda' and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("CUDA not available, switching to MPS")
                device = 'mps'
            else:
                print("CUDA not available, switching to CPU")
                device = 'cpu'

        if device == 'cpu':
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu(model_path, threads)
            print("Loading model on CPU")
        else:
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_gpu(model_path, device=device)
            print("Loading model on GPU")
        print("luxvoice decided to use " + device)
        self.model = model
        self.feature_extractor = feature_extractor
        self.vocos = vocos
        self.tokenizer = tokenizer
        self.transcriber = transcriber  # Will be None/empty - manual transcription only
        self.device = device
        self.vocos.freq_range = 12000

    def encode_prompt(self, prompt_audio, prompt_text, duration=5, rms=0.001):
        """
        Encodes audio prompt according to duration and rms (volume control).
        
        Args:
            prompt_audio: Path to audio file or audio array
            prompt_text: Manually provided transcription text (REQUIRED)
            duration: Duration in seconds to extract from audio
            rms: Target RMS for volume normalization
            
        Returns:
            encode_dict: Dictionary containing encoded prompt features
        """
        if not prompt_text or not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a non-empty string. Manual transcription is required.")
            
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(
            prompt_audio, prompt_text, self.tokenizer, self.feature_extractor, 
            self.device, target_rms=rms, duration=duration
        )
        
        encode_dict = {
            "prompt_tokens": prompt_tokens, 
            "prompt_features_lens": prompt_features_lens, 
            "prompt_features": prompt_features, 
            "prompt_rms": prompt_rms,
            "prompt_text": prompt_text  # Store for reference
        }

        return encode_dict

    def generate_speech(self, text, encode_dict, num_steps=4, guidance_scale=3.0, t_shift=0.5, speed=1.0, return_smooth=False):
        """
        Encodes text and generates speech using flow matching model.
        
        Args:
            text: Text to synthesize
            encode_dict: Dictionary from encode_prompt containing prompt features
            num_steps: Number of diffusion steps (higher = better quality, slower)
            guidance_scale: Guidance scale for classifier-free guidance
            t_shift: Temperature shift parameter
            speed: Speed multiplier for generation
            return_smooth: If True, return 24kHz audio instead of 48kHz
        """
        prompt_tokens = encode_dict["prompt_tokens"]
        prompt_features_lens = encode_dict["prompt_features_lens"]
        prompt_features = encode_dict["prompt_features"]
        prompt_rms = encode_dict["prompt_rms"]

        # Set vocoder output sample rate
        self.vocos.return_48k = not return_smooth

        if self.device == 'cpu':
            final_wav = generate_cpu(
                prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, 
                text, self.model, self.vocos, self.tokenizer, 
                num_step=num_steps, guidance_scale=guidance_scale, t_shift=t_shift, speed=speed
            )
        else:
            final_wav = generate(
                prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, 
                text, self.model, self.vocos, self.tokenizer, 
                num_step=num_steps, guidance_scale=guidance_scale, t_shift=t_shift, speed=speed
            )

        return final_wav.cpu()