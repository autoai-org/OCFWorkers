import os
import torch
import argparse
from loguru import logger
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM

from _base import InferenceWorker

def translate_chatml_to_openchat(prompt):
    prompt = prompt.replace('<|im_start|>system\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>user\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>assistant\n', '<bot>: ')
    prompt = prompt.replace('<|im_start|>user', '<human>:')
    prompt = prompt.replace('<|im_start|>assistant', '<bot>:')
    prompt = prompt.replace('\n<|im_end|>', '')
    prompt = prompt.replace('<|im_end|>', '')
    prompt = prompt.rstrip()
    return prompt

model_class_mapping = {
    'AutoModel': AutoModel,
    'AutoModelForCausalLM': AutoModelForCausalLM,
    'LlamaForCausalLM': LlamaForCausalLM
}

class HFWorker(InferenceWorker):
    def __init__(self, model_name, fast_tokenizer, model_class) -> None:
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=fast_tokenizer)
        self.model = model_class_mapping[model_class].from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='auto',
        )

    async def handle_requests(self, msg):
        prompt = msg.get('prompt', '')
        max_new_tokens = msg.get('max_new_tokens', 128)
        temperature = msg.get('temperature', 0.9)
        top_k = msg.get('top_k', 50)
        top_p = msg.get('top_p', 0.9)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generation_output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        decoded_output = self.tokenizer.decode(generation_output[0])
        output = {
            'output': {
                'text': decoded_output,
            },
            'params': {
                'prompt': prompt,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
            }
        }
        return output

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="openlm-research/open_llama_7b")
    parser.add_argument("--fast-tokenizer", action="store_true", default=False)
    parser.add_argument('--model-class', type=str, default='AutoModelForCausalLM')

    args = parser.parse_args()
    logger.info(f"args: {args}")
    worker = HFWorker(args.model_name, args.fast_tokenizer, args.model_class)
    worker.start()