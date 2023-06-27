CUDA_VISIBLE_DEVICES=0 python hf_worker.py --model-name togethercomputer/RedPajama-INCITE-7B-Chat --fast-tokenizer & \
CUDA_VISIBLE_DEVICES=1 python hf_worker.py --model-name togethercomputer/RedPajama-INCITE-Chat-3B-v1 --fast-tokenizer & \
CUDA_VISIBLE_DEVICES=2 python hf_worker.py --model-name openlm-research/open_llama_7b --model-class LlamaForCausalLM & \
CUDA_VISIBLE_DEVICES=3 python hf_worker.py --model-name mosaicml/mpt-7b-chat --trust-remote-code --fast-tokenizer  --dtype bfloat16
