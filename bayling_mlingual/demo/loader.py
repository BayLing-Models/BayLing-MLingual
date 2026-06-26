from transformers import AutoTokenizer

from bayling_mlingual import BayLingMLingualConfig, BayLingMLingualForCausalLM


def load_tokenizers(mt_tokenizer_path, llm_tokenizer_path):
    tokenizer_mt = AutoTokenizer.from_pretrained(mt_tokenizer_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_tokenizer_path)
    if "llama3" in llm_tokenizer_path or "llama-3" in llm_tokenizer_path:
        tokenizer_llm.pad_token_id = 128002
    else:
        tokenizer_llm.pad_token_id = 0
    tokenizer_llm.padding_side = "left"
    return tokenizer_mt, tokenizer_llm


def load_model(model_path, tokenizer_llm, max_gen_len):
    config = BayLingMLingualConfig.from_pretrained(model_path)
    config.max_gen_len = max_gen_len
    model = BayLingMLingualForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        len_tokenizer_llm=len(tokenizer_llm),
    )
    model.model_mt.lm_head.weight = model.model_mt.model.shared.weight
    # model.model_mt.lm_head._hf_hook.execution_device=model.model_mt.model.shared.weight.device.index

    model.eval()
    return model
