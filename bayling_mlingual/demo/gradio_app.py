import gradio as gr

from .constants import GRADIO_HEADER, GRADIO_TITLE, LANGUAGES, SERVER_NAME, SERVER_PORT, SHARE
from .lang_maps import langs_map_nllb
from .loader import load_model, load_tokenizers
from .pipeline import interact_with_model


def build_demo(model, tokenizer_mt, tokenizer_llm, langs_map=langs_map_nllb):
    def run(input_text, src_lang, tgt_lang):
        return interact_with_model(
            model=model,
            tokenizer_mt=tokenizer_mt,
            tokenizer_llm=tokenizer_llm,
            langs_map=langs_map,
            input_text=input_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

    with gr.Blocks(title=GRADIO_TITLE) as demo:
        gr.Markdown(GRADIO_HEADER)

        with gr.Row():
            src_lang = gr.Dropdown(
                choices=LANGUAGES, label="Source Language"
            )
            tgt_lang = gr.Dropdown(
                choices=LANGUAGES, label="Target Language"
            )

        input_text = gr.Textbox(label="Input Text", lines=4, placeholder="Enter the input text here")

        with gr.Row():
            output_lr = gr.Textbox(label="Output (Low-resource Language)", lines=4)
            output_en = gr.Textbox(label="Output (English Reference)", lines=4)

        submit_btn = gr.Button("Run Model")

        submit_btn.click(
            fn=run,
            inputs=[input_text, src_lang, tgt_lang],
            outputs=[output_en, output_lr],
        )

    return demo


def launch_demo(
    model_path: str,
    mt_tokenizer_path: str,
    llm_tokenizer_path: str,
    max_gen_len: int = 256,
):
    langs_map = langs_map_nllb
    tokenizer_mt, tokenizer_llm = load_tokenizers(mt_tokenizer_path, llm_tokenizer_path)
    model = load_model(model_path, tokenizer_llm, max_gen_len)
    demo = build_demo(model, tokenizer_mt, tokenizer_llm, langs_map)
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=SHARE)
