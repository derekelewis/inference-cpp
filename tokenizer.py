from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", use_fast=True)
print(tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str("I'm here.\n"))
