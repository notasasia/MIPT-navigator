from datetime import date
from pathlib import Path
from html import escape
import os
import tempfile
import shutil
import weakref
import pymupdf
import streamlit as st
from navigator.corpus import ROOT, Store, SAMPLES, unzip_case
from navigator.search import Search
from navigator.answer import answer, structured

st.set_page_config(page_title="Физтех.Навигатор", page_icon="🧭", layout="centered")
st.markdown(
    """<style>
.block-container{max-width:960px;padding-top:2.7rem} h1,h2,h3{letter-spacing:-.03em}
.brand{color:#174d46;font-size:1.3rem;font-weight:760;letter-spacing:-.04em;margin:0 0 1.6rem}
.evidence{padding:.9rem 1.1rem;background:#f0f5ec;border-left:3px solid #16816f;line-height:1.6;overflow-wrap:anywhere}
[data-testid="stSidebar"]{border-right:1px solid #d8e4da}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(max_entries=8, show_spinner="Индексируем документы…")
def database(path):
    s = Store(path)
    return s, s.seed()


@st.cache_resource(max_entries=3, show_spinner="Подготавливаем локальный поиск…")
def lookup(path, stamp):
    return Search(Store(path))


class SessionStore:
    def __init__(self, directory):
        self.path = Path(tempfile.mkdtemp(prefix="navigator-session-", dir=directory))
        self.cleanup = weakref.finalize(self, shutil.rmtree, self.path, True)


if "session_store" not in st.session_state:
    parent = os.getenv("NAV_STATE")
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)
    st.session_state.session_store = SessionStore(parent)
session_dir = st.session_state.session_store.path
s, errors = database(str(session_dir))
docs = s.docs()
x = lookup(str(s.root), tuple(sorted(docs)))
ss = st.session_state
for key, default in {
    "messages": [],
    "pending": "",
    "role": "",
    "scope": list(docs),
    "notice": "",
}.items():
    if key not in ss:
        ss[key] = default


def dateline(d):
    if d.id in SAMPLES:
        return "учебный пример"
    bits = []
    if d.number:
        bits.append("№ " + d.number)
    if d.signed:
        bits.append("подписан " + date.fromisoformat(d.signed).strftime("%d.%m.%Y"))
    elif d.registered:
        bits.append(
            "регистрация " + date.fromisoformat(d.registered).strftime("%d.%m.%Y")
        )
    else:
        bits.append("дата утверждения не установлена")
    if d.effective:
        bits.append(
            "действует с " + date.fromisoformat(d.effective).strftime("%d.%m.%Y")
        )
    return " · ".join(bits)


@st.dialog("Источник", width="large")
def show(ids):
    options = [x.ids[i] for i in ids if i in x.ids]
    if not options:
        st.info("Фрагмент недоступен.")
        return
    f = st.selectbox(
        "Основание шага",
        options,
        format_func=lambda p: docs[p.doc].title + " · " + p.loc,
    )
    d = docs[f.doc]
    st.caption(d.name + " · " + dateline(d))
    st.markdown(
        '<div class="evidence">' + escape(f.text) + "</div>", unsafe_allow_html=True
    )
    if f.page:
        with pymupdf.open(s.original(d)) as pdf:
            n = st.number_input(
                "Страница исходного PDF", 1, len(pdf), f.page, key="page_" + f.id
            )
            st.image(
                pdf[n - 1]
                .get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3), alpha=False)
                .tobytes("png"),
                caption="Оригинал · стр. " + str(n),
            )
    st.caption(f.loc + " · " + f.method + " · " + f.id)
    st.download_button(
        "Скачать оригинал",
        s.original(d).read_bytes(),
        file_name=d.name,
        on_click="ignore",
    )


with st.sidebar:
    st.markdown('<div class="brand">Физтех.Навигатор · 2026.11.14-r3</div>', unsafe_allow_html=True)
    screen = st.radio(
        "Раздел", ["Навигатор", "Библиотека"], label_visibility="collapsed"
    )
    day = st.date_input(
        "Проверить на дату",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date(2100, 1, 1),
        format="DD.MM.YYYY",
    )
    if docs:
        with st.expander("Документы для поиска"):
            selected = st.multiselect(
                "Источники",
                list(docs),
                default=[k for k in ss.scope if k in docs],
                format_func=lambda k: docs[k].title,
            )
            if selected != ss.scope:
                ss.scope = selected
                ss.pending = ""
    st.caption(f"Документов в базе: {len(docs)}")
    if ss.messages and st.button("Новый диалог"):
        ss.messages = []
        ss.pending = ""
        ss.role = ""
        st.rerun()
if errors:
    for name, message in errors:
        st.warning(name + " · " + message)

if screen == "Библиотека" or not docs:
    st.title("Библиотека документов")
    st.write(
        "Загрузите исходный ZIP организаторов целиком или выберите PDF, DOCX и XLSX."
    )
    uploads = st.file_uploader(
        "Добавить документы",
        type=["zip", "pdf", "docx", "xlsx"],
        accept_multiple_files=True,
    )
    if st.button("Индексировать", disabled=not uploads, type="primary"):
        todo = []
        warnings = []
        added = []
        for u in uploads:
            try:
                todo.extend(
                    unzip_case(u.getvalue())
                    if u.name.lower().endswith(".zip")
                    else [(u.name, u.getvalue())]
                )
            except Exception as e:
                warnings.append(u.name + " · " + str(e))
        progress = st.progress(0)
        for i, (name, raw) in enumerate(todo):
            progress.progress(
                i / max(len(todo), 1), text=f"Документ {i + 1} из {len(todo)} · {name}"
            )
            try:
                added.append(s.load(name, raw))
            except Exception as e:
                warnings.append(name + " · " + str(e))
        ss.scope = list(dict.fromkeys(ss.scope + added))
        ss.notice = (
            "Обработано: "
            + str(len(added))
            + (". " + " / ".join(warnings) if warnings else ".")
        )
        st.rerun()
    if ss.notice:
        st.info(ss.notice)
    for d in sorted(docs.values(), key=lambda d: d.title):
        with st.expander(d.title + " · " + d.kind):
            st.caption(d.name + " · " + dateline(d))
            st.write(d.note or "Проиндексирован")
            if d.pages:
                st.write(f"Страниц: {d.pages}. OCR: {len(d.ocr)}.")
            if d.sheets:
                st.write("Листы: " + ", ".join(d.sheets) + f". Формул: {d.formulas}.")
            if d.warnings:
                st.warning(" ".join(d.warnings))
            st.download_button(
                "Скачать файл",
                s.original(d).read_bytes(),
                d.name,
                key="dl" + d.id,
                on_click="ignore",
            )
else:
    st.title("Навигатор сотрудника")
    if all(d.id in SAMPLES for d in docs.values()):
        st.caption("Публичная демонстрация на четырёх учебных документах.")
    if not ss.messages:
        st.write(
            "Опишите задачу. Откройте основание каждого шага, чтобы проверить пункт в оригинале."
        )
        buttons = st.columns(3)
        for col, title, prompt in zip(
            buttons,
            ("Командировка", "Заявка в ИТ", "Смета РЦНИ"),
            (
                "Хочу поехать в командировку. Что мне делать?",
                "Сломался рабочий ноутбук. Куда написать?",
                "Как оформить смету по договору РЦНИ с физическим лицом?",
            ),
        ):
            if col.button(title, width="stretch"):
                ss.nextq = prompt
    for i, (q, a) in enumerate(ss.messages):
        with st.chat_message("user"):
            st.write(q)
        st.subheader(a.summary)
        for j, step in enumerate(a.steps, 1):
            with st.container(border=True):
                st.checkbox(f"{j}. " + step.text, key=f"ck{i}-{j}")
                if step.condition:
                    if step.condition == step.text:
                        st.caption("Условие применения указано в тексте шага.")
                    else:
                        st.write("Условие применения", step.condition)
                if step.after:
                    st.caption("После шага " + ", ".join(str(n) for n in step.after))
                for did in dict.fromkeys(f.doc for f in step.sources):
                    locs = ", ".join(
                        dict.fromkeys(f.loc for f in step.sources if f.doc == did)
                    )
                    st.caption(
                        docs[did].title + " · " + dateline(docs[did]) + " · " + locs
                    )
                if st.button("Открыть основание", key=f"cite{i}-{j}", type="tertiary"):
                    show([f.id for f in step.sources])
        if a.steps:
            st.download_button(
                "Скачать инструкцию в JSON",
                structured(a, x).model_dump_json(indent=2),
                file_name="instruction.json",
                mime="application/json",
                key=f"json-{i}",
                on_click="ignore",
            )
        for n in a.notes:
            st.info(n)
        if (
            i == len(ss.messages) - 1
            and a.pending
            and "студент" in a.summary.lower()
            and "сотрудник" in a.summary.lower()
        ):
            cols = st.columns(2)
            for col, label in zip(cols, ["Как сотрудник", "Как студент"]):
                if col.button(label):
                    ss.nextq = label
    q = st.chat_input("Ваш вопрос о процедуре", max_chars=2400)
    q = ss.pop("nextq", None) or q
    if q:
        a = answer(
            q,
            x,
            selected=ss.scope,
            known=ss.role,
            pending=ss.pending,
            day=day.isoformat(),
        )
        ss.messages.append((q, a))
        ss.role = a.role
        ss.pending = a.pending
        st.rerun()
