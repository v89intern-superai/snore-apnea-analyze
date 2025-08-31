import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from .config import CKPT_DIR

def load_model_and_processor():
    processor = Wav2Vec2Processor.from_pretrained(CKPT_DIR)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(CKPT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ลด RAM/VRAM: half + inference_mode ถ้า GPU รองรับ
    if device.type == "cuda":
        model = model.to(device).half().eval()
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        model = model.to(device).eval()

    id2label = model.config.id2label or {0:"Snore", 1:"OSA"}
    return processor, model, device, id2label
