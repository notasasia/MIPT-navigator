from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import threading
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import onnxruntime as rt
from snowballstemmer import stemmer
from tokenizers import Tokenizer
from .corpus import ROOT

STOP = set(
    "и на к а о у как что чем в во с для это я мне мы ли есть можно из до по от которые если вас не при кто про чтобы какой нужно хочу мне делать вы мфти свой им или сообщите".split()
)
stem = stemmer("russian")
lock = threading.Lock()


@lru_cache(maxsize=24000)
def normal(w):
    if len(w) < 4 or w.isdigit():
        return w
    with lock:
        return stem.stemWord(w)


def tokens(s):
    return [
        normal(w)
        for w in re.findall(r"\d{1,6}-\d+|[a-zа-яё]+|\d+", s.lower().replace("ё", "е"))
        if w not in STOP and len(w) > 1
    ]


class Encoder:
    def __init__(self):
        local = ROOT / "data/model"
        self.tok = Tokenizer.from_file(str(local / "tokenizer.json"))
        self.tok.enable_truncation(max_length=160)
        self.tok.enable_padding(
            pad_id=self.tok.token_to_id("[PAD]") or 0, pad_token="[PAD]"
        )
        settings = rt.SessionOptions()
        settings.intra_op_num_threads = 2
        settings.inter_op_num_threads = 1
        self.session = rt.InferenceSession(
            str(local / "encoder.onnx"), settings, providers=["CPUExecutionProvider"]
        )
        self.signature = hashlib.sha256(
            (local / "encoder.onnx").read_bytes()
            + (local / "tokenizer.json").read_bytes()
        ).hexdigest()

    def encode(self, texts):
        out = np.empty((len(texts), 312), dtype="float32")
        order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
        for i in range(0, len(texts), 24):
            seq = order[i : i + 24]
            e = self.tok.encode_batch([texts[j] for j in seq])
            inputs = {
                "input_ids": np.asarray([v.ids for v in e], dtype="int64"),
                "attention_mask": np.asarray(
                    [v.attention_mask for v in e], dtype="int64"
                ),
                "token_type_ids": np.asarray([v.type_ids for v in e], dtype="int64"),
            }
            cls = self.session.run(None, inputs)[0][:, 0]
            out[seq] = cls / np.maximum(
                np.linalg.norm(cls, axis=1, keepdims=True), 1e-8
            )
        return out


@lru_cache(maxsize=1)
def encoder():
    return Encoder()


_QUANTITY = re.compile(
    r"(?<![\w.])([0-9]+(?:[.,][0-9]+)?)(?=\s*(?:рабоч|календарн|дн|день|час|недел|месяц|рубл|%))",
    re.I,
)


def comparable_rule(part):
    raw = re.sub(
        r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", part.text.casefold().replace("ё", "е")
    )
    raw = " ".join(raw.split())
    numbers = tuple(_QUANTITY.findall(raw))
    if not numbers or len(tokens(raw)) < 5 or len(raw) > 850:
        return None
    return _QUANTITY.sub("#", raw), numbers


class Search:
    def __init__(self, store):
        self.store = store
        self.docs = store.docs()
        self.rows = store.parts()
        self.ids = {f.id: f for f in self.rows}
        self.keys = defaultdict(list)
        self.freq = defaultdict(list)
        self.lens = np.zeros(len(self.rows))
        strings = []
        for j, f in enumerate(self.rows):
            d = self.docs[f.doc]
            self.keys[d.key].append(f)
            terms = tokens(d.title + " " + d.number + " " + f.parent + " " + f.text)
            for t, n in Counter(terms).items():
                self.freq[t].append((j, n))
            self.lens[j] = len(terms)
            strings.append(d.title + ". " + f.parent[:100] + ". " + f.text[:1800])
        self.variants = defaultdict(list)
        if any(not d.key for d in self.docs.values()):
            for fragment in self.rows:
                rule = comparable_rule(fragment)
                if rule:
                    self.variants[rule[0]].append((fragment, rule[1]))
        self.avg = max(float(self.lens.mean()) if len(self.lens) else 0, 1)
        self.vec = self.matrix(strings) if strings else np.empty((0, 312), np.float32)

    def conflicting(self, parts, allowed):
        if not self.variants:
            return []
        for part in parts:
            rule = comparable_rule(part)
            if not rule:
                continue
            for other, values in self.variants.get(rule[0], ()):
                if other.doc != part.doc and other.doc in allowed and values != rule[1]:
                    return [part, other]
        return []

    def matrix(self, strings):
        e = encoder()
        h = hashlib.sha256(e.signature.encode())
        for s in strings:
            h.update(s.encode() + b"\0")
        p = self.store.root / (h.hexdigest() + ".npy")
        for path in (p, ROOT / "data/vectors" / p.name):
            if path.exists():
                a = np.load(path, allow_pickle=False)
                if a.shape == (len(strings), 312) and np.isfinite(a).all():
                    return a
        a = e.encode(strings)
        with tempfile.NamedTemporaryFile(dir=self.store.root, delete=False) as tmp:
            np.save(tmp, a, allow_pickle=False)
            name = tmp.name
        os.replace(name, p)
        return a

    def available(self, selected, day, history=False):
        allowed = set(self.docs if selected is None else selected)
        return {
            d.id
            for d in self.docs.values()
            if d.id in allowed
            and not any(v and v > day for v in (d.effective, d.signed, d.registered))
            and not (d.key == "oldfls" and day >= "2021-12-01" and not history)
        }

    def find(self, q, allowed, limit=18, mode="hybrid"):
        ix = np.array(
            [i for i, f in enumerate(self.rows) if f.doc in allowed], dtype=int
        )
        if not len(ix):
            return []
        score = np.zeros(len(self.rows))
        query = q
        if re.search(r"ноутбук|принтер|компьютер", q, re.I) and re.search(
            r"ремонт|неисправ|сломал|не работа", q, re.I
        ):
            query += " техническая поддержка оборудование"
        for t in set(tokens(query)):
            postings = self.freq.get(t, [])
            if not postings:
                continue
            i, f = np.asarray(postings).T
            score[i] += (
                math.log(1 + (len(self.rows) - len(i) + 0.5) / (len(i) + 0.5))
                * f
                * 2.3
                / (f + 1.3 * (0.30 + 0.70 * self.lens[i] / self.avg))
            )
        sim = (
            self.vec @ encoder().encode([q])[0]
            if mode != "bm25"
            else np.zeros(len(score))
        )
        fused = {}
        for v, minimum in [(score, 0)] + ([(sim, 0.1)] if mode != "bm25" else []):
            ranks = ix[np.argsort(-v[ix], kind="stable")]
            for rank, i in enumerate(ranks[: max(80, limit)], 1):
                if v[i] <= minimum:
                    continue
                fused[i] = fused.get(i, 0) + 1 / (25 + rank)
        return [
            (self.rows[i], float(score[i]), float(sim[i]))
            for i in sorted(fused, key=lambda i: -fused[i])[:limit]
        ]
