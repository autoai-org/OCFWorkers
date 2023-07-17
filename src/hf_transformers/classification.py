import torch
import argparse
from loguru import logger
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSequenceClassification

from _base import InferenceWorker

model_class_mapping = {
    'AutoModel': AutoModel,
    'AutoModelForCausalLM': AutoModelForCausalLM,
    'LlamaForCausalLM': LlamaForCausalLM,
    'AutoModelForSequenceClassification': AutoModelForSequenceClassification,
}

dtype_mapping = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

class HFWorker(InferenceWorker):
    def __init__(self, model_name, fast_tokenizer, model_class, trust_remote_code, dtype) -> None:
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=fast_tokenizer)
        if model_class == 'AutoModelForSequenceClassification':
            self.model = model_class_mapping[model_class].from_pretrained(
                model_name, torch_dtype=dtype_mapping[dtype], trust_remote_code=trust_remote_code
            )
            self.model.cuda()
        else:
            self.model = model_class_mapping[model_class].from_pretrained(
                model_name, torch_dtype=dtype_mapping[dtype], device_map='auto', trust_remote_code=trust_remote_code
            )
    
    async def handle_requests(self, msg):
        prompt = msg.get('prompt', '')
        encoded_inputs = self.tokenizer.encode(prompt, padding=True)
        prediction = self.model(torch.tensor([encoded_inputs]).cuda())['logits']
        predicted_label = torch.argmax(prediction, dim=1)
        result= predicted_label.tolist()[0]
        output = {
            'output': {
                'text': result,
            },
            'params': {
                'prompt': prompt,
            }
        }
        return output

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="openlm-research/open_llama_7b")
    parser.add_argument("--fast-tokenizer", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument('--model-class', type=str, default='AutoModelForCausalLM')
    parser.add_argument('--dtype', type=str, default='float16')
    args = parser.parse_args()
    logger.info(f"args: {args}")
    worker = HFWorker(args.model_name, args.fast_tokenizer, args.model_class, args.trust_remote_code, args.dtype)
    worker.start()