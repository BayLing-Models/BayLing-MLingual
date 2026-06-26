import fire

from bayling_mlingual.demo.gradio_app import launch_demo


def main(
    model_path: str,
    mt_tokenizer_path: str,
    llm_tokenizer_path: str,
    max_gen_len: int = 256,
):
    launch_demo(
        model_path=model_path,
        mt_tokenizer_path=mt_tokenizer_path,
        llm_tokenizer_path=llm_tokenizer_path,
        max_gen_len=max_gen_len,
    )


if __name__ == "__main__":
    fire.Fire(main)