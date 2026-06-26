from .lang_maps import langs_map_nllb


LANGUAGES = list(langs_map_nllb.keys())
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{input_text}\n\n"
    "### Response:\n"
)
GRADIO_TITLE = "Custom Model Interface"
GRADIO_HEADER = "## 🌐 BayLing-MLingual Multilingual Instruction Interface"
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 6654
SHARE = True
