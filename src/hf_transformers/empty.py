import torch
import argparse
from loguru import logger
from _base import InferenceWorker

class HFWorker(InferenceWorker):
    def __init__(self, model_name, fast_tokenizer, model_class, trust_remote_code, dtype) -> None:
        super().__init__(model_name)
    
    async def handle_requests(self, msg):
        output = {
            'output': {
                'text': 'test',
            },
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