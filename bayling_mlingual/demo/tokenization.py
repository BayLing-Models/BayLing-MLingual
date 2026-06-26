import torch


def mt_input_features(tokenizer_mt, input_text_m2m, source_language, langs_map, max_length=-1):
    tokenizer_mt.src_lang = langs_map[source_language]
    encoding_m2m = tokenizer_mt(
        input_text_m2m,
        padding=False,
        truncation=False if max_length == -1 else True,
        max_length=None if max_length == -1 else max_length,
        return_tensors=None,
        add_special_tokens=True,
    )
    return encoding_m2m["input_ids"]


def llm_input_features(tokenizer_llm, input_texts_llm, max_length=-1, add_special_tokens=True):
    encoding_llm = tokenizer_llm(
        input_texts_llm,
        padding=False,
        truncation=False if max_length == -1 else True,
        max_length=None if max_length == -1 else max_length,
        return_tensors=None,
        add_special_tokens=add_special_tokens,
    )
    return encoding_llm["input_ids"]


def pad_and_mask(input_ids_mt, input_ids_prompt, pad_token_id):
    input_ids = [mt + prompt for mt, prompt in zip(input_ids_mt, input_ids_prompt)]

    max_len = max(len(seq) for seq in input_ids)
    augmentation = [[0] * (max_len - len(input_ids[i])) + [1] * len(input_ids_mt[i]) + [2] * len(input_ids_prompt[i]) for i in range(len(input_ids_mt))]
    attention_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in input_ids]
    input_ids = [[pad_token_id] * (max_len - len(seq)) + seq for seq in input_ids]

    return torch.tensor(input_ids).cuda(), torch.tensor(attention_mask).cuda(), torch.tensor(augmentation).cuda()
