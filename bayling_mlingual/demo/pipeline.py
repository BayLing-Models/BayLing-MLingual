from .constants import PROMPT_TEMPLATE
from .tokenization import llm_input_features, mt_input_features, pad_and_mask


def interact_with_model(model, tokenizer_mt, tokenizer_llm, langs_map, input_text, src_lang, tgt_lang):
    mt_input = input_text

    input_ids_mt = mt_input_features(
        tokenizer_mt=tokenizer_mt,
        input_text_m2m=[mt_input],
        source_language=src_lang,
        langs_map=langs_map,
    )

    prompt = PROMPT_TEMPLATE.format(input_text=mt_input)

    input_ids_prompt = llm_input_features(
        tokenizer_llm=tokenizer_llm,
        input_texts_llm=[prompt],
        add_special_tokens=False,
    )

    input_ids, attention_mask, augmentation = pad_and_mask(input_ids_mt, input_ids_prompt, tokenizer_llm.pad_token_id)

    llm_generate_ids, mt_dec_generate_ids = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        augmentation=augmentation,
        forced_decoder_start_token_id=[tokenizer_mt.convert_tokens_to_ids(langs_map[tgt_lang])],
    )
    llm_outputs = tokenizer_llm.batch_decode(llm_generate_ids, skip_special_tokens=True)
    mt_dec_outputs = tokenizer_mt.batch_decode(mt_dec_generate_ids, skip_special_tokens=True)

    return llm_outputs[0], mt_dec_outputs[0]
