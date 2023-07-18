docker run \
--network host \
--gpus all \
--mount type=bind,source="$(pwd)"/.cache,target=/cache \
25b9d183699f49370f1d2e94 \
python3 /app/hf_transformers/hf_worker.py --model-name openlm-research/open_llama_7b --model-class LlamaForCausalLM