import os

class JanusModelLoader:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["Janus-Pro-1B","Janus-Pro-7B","Janus-4o-7B"], {"default": "Janus-4o-7B"}),
                "quantization": (["none", "4-bit", "4-bit-double"], {"default": "none"}),
            },
        }
    
    RETURN_TYPES = ("JANUS_MODEL", "JANUS_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Janus-Pro"

    def load_model(self, model_name, quantization):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

                # Configure quantization (4-bit only)
        quantization_config = None
        if quantization != "none":
            if quantization == "4-bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif quantization == "4-bit-double":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        
        # Load model with quantization config
        if quantization_config:
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        # 获取ComfyUI根目录
        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 构建模型路径
        model_dir = os.path.join(comfy_path, 
                               "models", 
                               "Janus-Pro",
                               os.path.basename(model_name))
        if not os.path.exists(model_dir):
            raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")
            
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
        
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        vl_gpt = vl_gpt.to(dtype).to(device).eval()
        
        return (vl_gpt, vl_chat_processor) 
