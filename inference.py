import json
from pathlib import Path

import fire

from bayling_mlingual.demo.lang_maps import langs_map_nllb
from bayling_mlingual.demo.loader import load_model, load_tokenizers
from bayling_mlingual.demo.pipeline import interact_with_model


DEFAULT_EXAMPLES = [
    {
        "src_lang": "Chinese",
        "tgt_lang": "Czech",
        "input_text": "请介绍什么是大语言模型",
    },
    {
        "src_lang": "Ukrainian",
        "tgt_lang": "Urdu",
        "input_text": "Ви знаєте Шекспіра? Які його найвідоміші твори?",
    },
    {
        "src_lang": "Dutch",
        "tgt_lang": "Vietnamese",
        "input_text": "Waarom is de aarde blauw? ",
    },
]


def _load_records(input_file):
    path = Path(input_file)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("examples", data.get("data", [data]))
        return data
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix in {".tsv", ".txt"}:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 3:
                    src_lang, tgt_lang, input_text = parts
                elif len(parts) == 2:
                    src_lang, input_text = parts
                    tgt_lang = "English"
                else:
                    raise ValueError(
                        "TSV/TXT input expects src_lang<TAB>tgt_lang<TAB>input_text "
                        "or src_lang<TAB>input_text."
                    )
                records.append({
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "input_text": input_text,
                })
        return records
    raise ValueError("input_file must be .json, .jsonl, .tsv, or .txt")


def _normalize_record(record):
    if not isinstance(record, dict):
        raise ValueError("Each input record must be a JSON object or TSV row.")

    src_lang = record.get("src_lang") or record.get("source_language") or record.get("source")
    tgt_lang = record.get("tgt_lang") or record.get("target_language") or record.get("target")
    input_text = record.get("input_text") or record.get("text") or record.get("input")

    if not src_lang or not tgt_lang or input_text is None:
        raise ValueError(
            "Each record needs src_lang, tgt_lang, and input_text "
            "(aliases: source/source_language, target/target_language, text/input)."
        )
    if src_lang not in langs_map_nllb:
        raise ValueError(f"Unsupported src_lang: {src_lang}")
    if tgt_lang not in langs_map_nllb:
        raise ValueError(f"Unsupported tgt_lang: {tgt_lang}")

    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "input_text": input_text,
    }


def _write_outputs(outputs, output_file):
    if output_file is None:
        for item in outputs:
            print(json.dumps(item, ensure_ascii=False))
        return

    path = Path(output_file)
    with path.open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(
    model_path: str,
    mt_tokenizer_path: str,
    llm_tokenizer_path: str,
    input_file: str = None,
    output_file: str = None,
    max_gen_len: int = 256,
):
    records = _load_records(input_file) if input_file else DEFAULT_EXAMPLES
    records = [_normalize_record(record) for record in records]

    tokenizer_mt, tokenizer_llm = load_tokenizers(mt_tokenizer_path, llm_tokenizer_path)
    model = load_model(model_path, tokenizer_llm, max_gen_len)

    outputs = []
    for record in records:
        output_en, output_target = interact_with_model(
            model=model,
            tokenizer_mt=tokenizer_mt,
            tokenizer_llm=tokenizer_llm,
            langs_map=langs_map_nllb,
            input_text=record["input_text"],
            src_lang=record["src_lang"],
            tgt_lang=record["tgt_lang"],
        )
        outputs.append({
            **record,
            "output_en": output_en,
            "output_target": output_target,
        })

    _write_outputs(outputs, output_file)


if __name__ == "__main__":
    fire.Fire(main)
