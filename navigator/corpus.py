"""Structural indexing and local originals. No macros, formulas or links are executed."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pymupdf
from docx import Document as Word
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
FORMATS = {".pdf", ".docx", ".xlsx"}
SIZE = 35 * 1024**2
LIMIT = 190 * 1024**2
CLAUSE = re.compile(
    r"^\s*(\d{1,3}(?:\.\d{1,3}){0,3})(?:[.)](?=\s|[А-Я])|(?<=\d)(?=\s))\s*(.+)"
)
TITLE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:общие положения|порядок (?:поиска|подачи|открытия|обработки)|режим работы|каналы подачи|правила поселения|процедура оформления документов|заключительные положения|предоставляемые сервисы|приложение\s*(?:№\s*)?\d*)$",
    re.I,
)


def clean(s):
    return " ".join(str(s).replace("\u00ad", "").replace("\uf02d", "–").split())


def sha(b):
    return hashlib.sha256(b).hexdigest()


# Each digest identifies an exact original observed in the competition data.
OFFICIAL = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
SAMPLES = json.loads((ROOT / "data/samples.json").read_text(encoding="utf-8"))
SOURCE_KEYS = {
    "0788efe4": "loan",
    "0ea0b6ac": "housing",
    "2a8080d4": "hall",
    "9efd4969": "dc",
    "d6dae7e9": "eb",
    "e6fd27dd": "oldfls",
}
META = [
    (
        "Регламент о командировании",
        "travel",
        "Командирование работников",
        "",
        "",
        "",
        "",
        "Дата и номер утверждающего приказа не заполнены.",
    ),
    (
        "Приказ и Регламент работы первой линии",
        "support",
        "Первая линия технической поддержки",
        "2993-1",
        "2026-06-19",
        "",
        "",
        "",
    ),
    (
        "Регламент поиска и подбора",
        "hire",
        "Поиск и подбор персонала",
        "",
        "",
        "",
        "",
        "Номер и дата в грифе приложения не заполнены.",
    ),
    (
        "Регламент бронирования",
        "hall",
        "Бронирование концертного зала",
        "3897-1",
        "",
        "2025-09-01",
        "",
        "По отметке регистрации в ЭДО.",
    ),
    (
        "Регламент поселения",
        "housing",
        "Поселение сотрудников",
        "",
        "",
        "",
        "",
        "В грифе указан 2024 год без дня и месяца.",
    ),
    (
        "Регламент эксплуатации",
        "dc",
        "Эксплуатация ЦОД",
        "1292-1",
        "",
        "2025-03-28",
        "",
        "По отметке регистрации в ЭДО.",
    ),
    (
        "Приказ и регламент по ЭБ",
        "eb",
        "ГИИС «Электронный бюджет»",
        "1027-1",
        "2019-06-25",
        "",
        "",
        "Регламент действует с даты регистрации, которая отдельно не подтверждена.",
    ),
    (
        "460-1",
        "oldfls",
        "Изменения документов об открытии ФЛС",
        "460-1",
        "2017-04-28",
        "",
        "2017-05-01",
        "Дата 01.05.2017 относится к приложенному регламенту.",
    ),
    (
        "Приказ № 5990-1",
        "loan",
        "Внутреннее заимствование средств",
        "5990-1",
        "2026-11-03",
        "",
        "2026-11-03",
        "",
    ),
    (
        "Приказ N 3065-1",
        "fls",
        "Открытие финансового лицевого счёта",
        "3065-1",
        "2021-11-25",
        "",
        "2021-12-01",
        "Дата действия новой редакции регламента.",
    ),
    (
        "Распоряжение N 700",
        "number",
        "Присвоение номера ФЛС",
        "700",
        "2024-12-28",
        "",
        "",
        "",
    ),
    (
        "Приказ N 794-1",
        "estimate",
        "Согласование и утверждение смет",
        "794-1",
        "2023-03-28",
        "",
        "",
        "",
    ),
    (
        "Приказ N 889-1",
        "system",
        "Подготовка и согласование смет в 1С",
        "889-1",
        "2023-04-04",
        "",
        "2023-04-04",
        "Дата ввода функционала. Приложенная инструкция в исходном DOCX не обнаружена.",
    ),
    (
        "Приказ N 3614-1",
        "structure",
        "Финансовая структура МФТИ",
        "3614-1",
        "2025-08-11",
        "",
        "",
        "",
    ),
    (
        "Приказ N 3699-1",
        "budget",
        "Формирование бюджета на 2026–2028 годы",
        "3699-1",
        "2025-08-18",
        "",
        "",
        "",
    ),
    (
        "Копия Приложение 1б",
        "rcni",
        "Годовая смета РЦНИ с физическими лицами",
        "",
        "",
        "",
        "",
        "Форма, ссылка на неё содержится в п. 2.2 приказа № 794-1.",
    ),
    (
        "Приложение 1_новая",
        "oldform",
        "Шаблон сметы 2019 года",
        "",
        "",
        "",
        "",
        "Принадлежность версии формы действующему приказу требует сверки.",
    ),
    ("Приложение 1а", "form", "Бланк сметы 1а", "", "", "", "", ""),
    (
        "Приложение к приказу об утверждении",
        "struct_table",
        "Справочник финансовой структуры 2026",
        "",
        "",
        "",
        "",
        "",
    ),
    (
        "Приложения к приказу_",
        "budget_forms",
        "Приложения для бюджетирования на 2026 год",
        "",
        "",
        "",
        "",
        "",
    ),
]


@dataclass
class Doc:
    id: str
    name: str
    kind: str
    title: str
    key: str = ""
    number: str = ""
    signed: str = ""
    registered: str = ""
    effective: str = ""
    note: str = ""
    pages: int = 0
    sheets: list = field(default_factory=list)
    ocr: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    tables: int = 0
    formulas: int = 0


@dataclass
class Part:
    id: str
    doc: str
    text: str
    loc: str
    clause: str = ""
    parent: str = ""
    page: int = 0
    end: int = 0
    para: int = 0
    sheet: str = ""
    cells: str = ""
    method: str = "текст"


def numbers(s):
    m = CLAUSE.match(s)
    if not m:
        return ""
    n, rest = m.groups()
    if "." not in n and not re.match(r"\d+[.)]", s) and not TITLE.match(s):
        return ""
    if re.match(r"ставк|рубл|процент|разряд", rest, re.I):
        return ""
    return n


def safe_zip(data, case=False):
    if len(data) > (LIMIT if case else SIZE):
        raise ValueError("Превышен лимит размера файла.")
    z = ZipFile(BytesIO(data))
    entries = z.infolist()
    if (
        len(entries) > (700 if case else 6000)
        or sum(e.file_size for e in entries) > LIMIT
        or any(e.flag_bits & 1 for e in entries)
    ):
        z.close()
        raise ValueError("Архив защищён паролем или превышает лимит распаковки.")
    return z


def unzip_case(data):
    with safe_zip(data, True) as z:
        files = []
        for e in z.infolist():
            n = e.filename.replace("\\", "/").split("/")[-1]
            if (
                not e.is_dir()
                and Path(n).suffix.lower() in FORMATS
                and not n.startswith(("~$", "."))
            ):
                if e.file_size > SIZE:
                    raise ValueError(n + " превышает 35 МБ.")
                files.append((n, z.read(e)))
        if not files:
            raise ValueError("В архиве нет PDF, DOCX или XLSX.")
        return files


def ocr(page):
    if not shutil.which("tesseract"):
        raise ValueError("Нет системной программы Tesseract.")
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    opts = ["tesseract", "stdin", "stdout", "-l", "rus"]
    d = ROOT / "data/ocr"
    if (d / "rus.traineddata").is_file():
        opts += ["--tessdata-dir", str(d)]
    maxside = max(page.rect.width, page.rect.height)
    small = page.get_pixmap(
        matrix=pymupdf.Matrix(1450 / maxside, 1450 / maxside), colorspace=pymupdf.csGRAY
    )
    orient = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "0", "--dpi", "150"],
        input=small.tobytes("png"),
        capture_output=True,
        timeout=30,
        env=env,
    )
    report = orient.stdout.decode(errors="replace")
    degrees = re.search(r"Rotate: (\d+)", report)
    conf = re.search(r"confidence: ([\d.]+)", report)
    rotation = int(degrees[1]) if degrees and conf and float(conf[1]) >= 8 else 0
    zoom = min(3.25, 3200 / maxside)
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom).prerotate(rotation), colorspace=pymupdf.csGRAY
    )
    out = subprocess.run(
        opts, input=pix.tobytes("png"), capture_output=True, timeout=90, env=env
    )
    if out.returncode:
        raise ValueError("Ошибка русского OCR.")
    return out.stdout.decode(errors="replace")


class Builder:
    def __init__(self, doc):
        self.doc = doc
        self.parts = []
        self.lines = []
        self.clause = ""
        self.parent = ""
        self.first = 0
        self.last = 0
        self.method = "текст"

    def add(self, text, loc, **kw):
        if clean(text):
            self.parts.append(
                Part(
                    f"{self.doc.id[:16]}-{len(self.parts) + 1:04}",
                    self.doc.id,
                    clean(text),
                    loc,
                    **kw,
                )
            )

    def flush(self):
        if not self.lines:
            return
        pdf = self.doc.kind == "PDF"
        span = str(self.first) + (f"–{self.last}" if self.last != self.first else "")
        loc = (
            (f"п. {self.clause} · " if self.clause else "")
            + ("стр. " if pdf else "абз. ")
            + span
        )
        self.add(
            " ".join(self.lines),
            loc,
            clause=self.clause,
            parent=self.parent,
            page=self.first if pdf else 0,
            para=0 if pdf else self.first,
            end=self.last,
            method=self.method,
        )
        self.lines = []

    def feed(self, line, position, source="текст"):
        line = clean(line)
        if not line:
            return
        n = numbers(line)
        h = bool(TITLE.match(line))
        if (
            n
            or h
            or self.method != source
            or (not self.clause and self.last != position)
        ) and self.lines:
            self.flush()
        if n or h:
            self.clause = n
        if h:
            self.parent = line
        if not self.lines:
            self.first = position
        self.last = position
        self.method = source
        self.lines.append(line)


def parse(name, data):
    kind = Path(name).suffix.lower()
    if kind not in FORMATS or not data or len(data) > SIZE:
        raise ValueError("Нужен непустой PDF, DOCX или XLSX до 35 МБ.")
    digest = sha(data)
    doc = Doc(
        digest, Path(name.replace("\\", "/")).name, kind[1:].upper(), Path(name).stem
    )
    b = Builder(doc)
    if kind == ".pdf":
        cached_path = ROOT / "data/ocr" / (digest + ".json")
        cached = (
            json.loads(cached_path.read_text(encoding="utf-8"))
            if cached_path.is_file()
            else {}
        )
        with pymupdf.open(stream=data, filetype="pdf") as f:
            if f.needs_pass or len(f) > 450:
                raise ValueError("PDF закрыт паролем или содержит более 450 страниц.")
            doc.pages = len(f)
            for page in f:
                i = page.number + 1
                text = page.get_text(sort=True)
                method = "текст"
                cover = max(
                    (
                        pymupdf.Rect(x["bbox"]).get_area() / page.rect.get_area()
                        for x in page.get_image_info()
                    ),
                    default=0,
                )
                if len(clean(text)) < 48 or (cover > 0.75 and len(clean(text)) < 1600):
                    try:
                        result = cached.get(str(i))
                        if result is None:
                            result = ocr(page)
                        if len(clean(result)) >= 20:
                            text = result
                            method = "OCR"
                            doc.ocr.append(i)
                    except (ValueError, OSError, subprocess.SubprocessError) as exc:
                        doc.warnings.append(f"Стр. {i} · {exc}")
                if not clean(text):
                    doc.warnings.append(f"Стр. {i} · нет извлекаемого текста.")
                    b.flush()
                    b.clause = ""
                for l in text.splitlines():
                    b.feed(l, i, method)
            b.flush()
            for page in f:
                if page.number + 1 in doc.ocr:
                    continue
                for j, t in enumerate(page.find_tables().tables, 1):
                    doc.tables += 1
                    for k, row in enumerate(t.extract(), 1):
                        vals = [clean(v) for v in row if v is not None and clean(v)]
                        if vals:
                            b.add(
                                " | ".join(vals),
                                f"стр. {page.number + 1}, табл. {j}, строка {k}",
                                page=page.number + 1,
                                end=page.number + 1,
                                parent="Таблица PDF",
                            )
    elif kind == ".docx":
        with safe_zip(data):
            pass
        f = Word(BytesIO(data))
        rownum = 0
        for el in f.element.body.iterchildren():
            if el.tag.endswith("}p"):
                rownum += 1
                b.feed(Paragraph(el, f).text, rownum)
            elif el.tag.endswith("}tbl"):
                b.flush()
                doc.tables += 1
                for j, row in enumerate(Table(el, f).rows, 1):
                    seen = set()
                    vals = []
                    for cell in row.cells:
                        if cell._tc in seen:
                            continue
                        seen.add(cell._tc)
                        if clean(cell.text):
                            vals.append(clean(cell.text))
                    b.add(
                        " | ".join(vals),
                        f"табл. {doc.tables}, строка {j}",
                        parent=b.parent,
                    )
                b.clause = ""
        b.flush()
    else:
        with safe_zip(data):
            pass
        w = load_workbook(
            BytesIO(data), read_only=True, data_only=False, keep_links=False
        )
        try:
            for sheet in w:
                if (sheet.max_row or 0) * (sheet.max_column or 0) > 2_000_000:
                    raise ValueError("Слишком большой лист.")
                doc.sheets.append(sheet.title)
                for row in sheet.iter_rows():
                    vals = [
                        (c.coordinate, clean(c.value), c.data_type)
                        for c in row
                        if c.value is not None and clean(c.value)
                    ]
                    if vals:
                        rg = vals[0][0] + (":" + vals[-1][0] if len(vals) > 1 else "")
                        b.add(
                            " | ".join(pos + " " + txt for pos, txt, _ in vals),
                            f"«{sheet.title}»!{rg}",
                            sheet=sheet.title,
                            cells=rg,
                        )
                        doc.formulas += sum(t == "f" for _, _, t in vals)
        finally:
            w.close()
    if not b.parts:
        raise ValueError("В документе нет распознаваемого текста.")
    # Uploaded documents have no curated dates. Accept only explicit approval/signing marks.
    header = " ".join(p.text for p in b.parts[:4])[:950]
    stamp = re.search(
        r"\b(?:утвержден[ао]?|подписан[ао]?)\s*(?:от\s*)?[:–-]?\s*(\d{2}\.\d{2}\.\d{4})\b",
        header.replace("ё", "е"),
        re.I,
    )
    if stamp and digest not in OFFICIAL.values():
        try:
            doc.signed = datetime.strptime(stamp[1], "%d.%m.%Y").date().isoformat()
        except ValueError:
            pass
    for prefix, key, title, num, signed, registered, effective, note in META:
        if digest in OFFICIAL.values() and clean(
            next(n for n, h in OFFICIAL.items() if h == digest)
        ).lower().startswith(prefix.lower()):
            doc.key = key
            doc.title = title
            doc.note = note
            # Metadata for a replacement file requires verification against its own content.
            sample = " ".join(p.text for p in b.parts[:36])
            if key in SOURCE_KEYS.values() and not any(
                digest.startswith(h) and v == key for h, v in SOURCE_KEYS.items()
            ):
                doc.note = "Тематическое совпадение по имени. Реквизиты этой версии требуют проверки."
            elif not num or num in sample or key in SOURCE_KEYS.values():
                doc.number = num
                doc.signed = signed
                doc.registered = registered
                doc.effective = effective
            break
    if digest in SAMPLES:
        spec = SAMPLES[digest]
        doc.key = spec["key"]
        doc.title = spec["title"]
        doc.note = "Синтетический пример для демонстрации."
    return doc, b.parts


class Store:
    def __init__(self, folder=None):
        self.root = Path(folder or os.getenv("NAV_STATE", ROOT / "state")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = self.root / "files"
        self.files.mkdir(exist_ok=True)
        self.path = self.root / "index.sqlite"
        with self.db() as c:
            c.executescript(
                "PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY,meta TEXT); CREATE TABLE IF NOT EXISTS parts(id TEXT PRIMARY KEY, doc TEXT, seq INTEGER, item TEXT);CREATE INDEX IF NOT EXISTS parts_doc ON parts(doc,seq);"
            )

    def db(self):
        return sqlite3.connect(self.path, timeout=50)

    def load(self, name, data):
        identity = sha(data)
        name = Path(name.replace("\\", "/")).name
        with self.db() as c:
            old = c.execute("SELECT meta FROM docs WHERE id=?", (identity,)).fetchone()
        if old:
            olddoc = Doc(**json.loads(old[0]))
            if self.original(olddoc).is_file():
                return identity
        doc, parts = parse(name, data)
        with tempfile.NamedTemporaryFile(dir=self.files, delete=False) as tmp:
            tmp.write(data)
            td = Path(tmp.name)
        try:
            os.replace(td, self.original(doc))
        finally:
            td.unlink(missing_ok=True)
        with self.db() as c:
            c.execute(
                "INSERT OR IGNORE INTO docs VALUES(?,?)",
                (doc.id, json.dumps(asdict(doc), ensure_ascii=False)),
            )
            c.executemany(
                "INSERT OR IGNORE INTO parts VALUES(?,?,?,?)",
                (
                    (p.id, doc.id, i, json.dumps(asdict(p), ensure_ascii=False))
                    for i, p in enumerate(parts)
                ),
            )
        return identity

    def original(self, doc):
        return self.files / (doc.id + "." + doc.kind.lower())

    def docs(self):
        with self.db() as c:
            return {
                k: Doc(**json.loads(v))
                for k, v in c.execute("SELECT id,meta FROM docs ORDER BY id")
            }

    def parts(self):
        with self.db() as c:
            return [
                Part(**json.loads(r[0]))
                for r in c.execute("SELECT item FROM parts ORDER BY doc,seq")
            ]

    def seed(self, folder=None):
        folder = Path(folder or ROOT / "corpus")
        failures = []
        if not folder.exists():
            return failures
        for f in sorted(folder.glob("*")):
            if f.is_file() and f.suffix.lower() in FORMATS:
                try:
                    self.load(f.name, f.read_bytes())
                except Exception as e:
                    failures.append((f.name, str(e)))
        return failures
