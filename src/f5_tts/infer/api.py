from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import soundfile as sf
import base64
import io
from importlib.resources import files
from cached_path import cached_path

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT

app = FastAPI(title="F5 TTS API")

class TTSRequest(BaseModel):
    text: str

# Initialize model and vocoder globally
def init_model():
    # Model configuration
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    
    # Load model checkpoint
    repo_name = "F5-TTS"
    exp_name = "F5TTS_Base"
    ckpt_step = 1200000
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    
    # Load vocoder
    vocoder_name = "vocos"
    vocoder = load_vocoder(vocoder_name=vocoder_name)
    
    # Load model
    model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name)
    
    # Get reference audio path
    ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    
    return model, vocoder, ref_audio

model, vocoder, ref_audio = init_model()
ref_text = "Some call me nature, others call me mother nature."

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Process the audio
        audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=request.text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="vocos"
        )
        
        # Convert audio to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, sample_rate, format='WAV')
        audio_bytes.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "sample_rate": sample_rate
        }
    
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 