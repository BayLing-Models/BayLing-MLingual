from transformers import LlamaConfig


class BayLingMLingualConfig(LlamaConfig):
    def __init__(
        self,
        mt_path=None,
        llm_path=None,
        dec_lambda=0.2,
        ot_lambda=1.0,
        max_gen_len=128,
        mt_pad_token_id=0,
        mt_eos_token_id=2,
        llm_bos_token_id=1,
        llm_eos_token_id=2,
        llm_pad_token_id=0,
        freeze_enc=True,
        freeze_llm=True,
        freeze_dec=True,
        freeze_mapping_enc2llm=False,
        freeze_mapping_llm2dec=False,
        mt_vocab_size=32000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mt_path = mt_path
        self.llm_path = llm_path
        self.dec_lambda = dec_lambda
        self.ot_lambda = ot_lambda
        self.max_gen_len = max_gen_len
        self.mt_pad_token_id = mt_pad_token_id
        self.mt_eos_token_id = mt_eos_token_id
        self.llm_bos_token_id = llm_bos_token_id
        self.llm_eos_token_id = llm_eos_token_id
        self.llm_pad_token_id = llm_pad_token_id
        self.freeze_enc = freeze_enc
        self.freeze_llm = freeze_llm
        self.freeze_dec = freeze_dec
        self.freeze_mapping_enc2llm = freeze_mapping_enc2llm
        self.freeze_mapping_llm2dec = freeze_mapping_llm2dec
        self.mt_vocab_size = mt_vocab_size
