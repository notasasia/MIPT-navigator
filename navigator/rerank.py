"""Score question/passage pairs locally. No model or document is downloaded at inference."""

from functools import lru_cache
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from .corpus import ROOT


class Reranker:
    def __init__(self, directory):
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("<pad>") or 0, pad_token="<pad>"
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self.model = ort.InferenceSession(
            str(directory / "reranker.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.inputs = {t.name for t in self.model.get_inputs()}

    def score(self, q, texts):
        rows = []
        for offset in range(0, len(texts), 8):
            batch = self.tokenizer.encode_batch(
                [(q, t) for t in texts[offset : offset + 8]]
            )
            arrays = {
                "input_ids": [x.ids for x in batch],
                "attention_mask": [x.attention_mask for x in batch],
                "token_type_ids": [x.type_ids for x in batch],
            }
            values = self.model.run(
                None, {k: np.asarray(arrays[k], dtype=np.int64) for k in self.inputs}
            )[0]
            rows.extend(float(s) for s in values.reshape(-1))
        return rows


@lru_cache(maxsize=1)
def local_reranker():
    folder = Path(os.environ.get("NAV_RERANK_DIR", ROOT / "models/reranker"))
    if (folder / "reranker.onnx").is_file() and (folder / "tokenizer.json").is_file():
        return Reranker(folder)
    return None
