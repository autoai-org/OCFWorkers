singularity exec --nv images/worker.sif python3 /app/hf_transformers/hf_worker.py --model-name openlm-research/open_llama_7b --model-class LlamaForCausalLM