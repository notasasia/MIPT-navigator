"""Dynamic planning over retrieved evidence; no per-procedure route registry."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
import re
from . import generator, selection
from .search import tokens
from .rerank import local_reranker


@dataclass
class Step:
    text: str
    sources: list
    condition: str = ""
    after: list[int] = field(default_factory=list)


@dataclass
class Answer:
    status: str
    summary: str
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    pending: str = ""
    role: str = ""


@dataclass(frozen=True)
class Evidence:
    id: int
    part: object
    text: str


def norm(s):
    return " ".join(s.lower().replace("ё", "е").split())


def role(q):
    v = norm(q)
    employee = bool(
        re.search(r"сотрудни[кц]|работни[кц]|работаю\b|трудов.{0,6}договор", v)
    ) and not re.search("без трудов|не работаю|не сотрудник", v)
    student = bool(re.search("студент|обучающ|аспирант", v))
    if "как сотрудник" in v:
        return "employee"
    if "как студент" in v:
        return "student"
    return (
        "both"
        if employee and student
        else "employee"
        if employee
        else "student"
        if student
        else ""
    )


def procedural_passage(p):
    """Exclude approval logs, unfilled order templates and broken PDF table lines."""
    if p.sheet:
        return True
    if re.search(
        r"_{5,}|согласовано в ДО|согласование инициировано|срок согласования\s*\|",
        p.text,
        re.I,
    ):
        return False
    return not (p.parent == "Таблица PDF" and p.text[:1].islower() and not p.clause)


def retrieve(q, x, allowed):
    qt = set(tokens(q))
    # Split independently requested infinitive actions; each gets its own retrieval budget.
    parts = re.split(r"\s+и\s+(?=[а-яё-]+(?:ть|ти|ться)\b)", q, flags=re.I)[:3]
    lanes = []
    for query in parts:
        hits = x.find(query, allowed, 45)
        query_terms = set(tokens(query))
        supported = [
            (p, sim)
            for p, lex, sim in hits
            if sim >= 0.85
            or (
                lex > 0
                and sim > 0.15
                and query_terms & set(tokens(x.docs[p.doc].title + " " + p.text))
            )
        ]
        if not supported:
            continue
        rr = local_reranker()
        if rr and len(supported) > 1:
            scores = rr.score(
                query,
                [x.docs[p.doc].title + ". " + p.text[:1100] for p, _ in supported],
            )
            ordered = sorted(range(len(scores)), key=lambda j: -scores[j])
            # Drop distant document families; retain variants belonging to a credible source.
            best = scores[ordered[0]]
            credible = {supported[j][0].doc for j in ordered if scores[j] >= best - 2.2}
            supported = [
                supported[j] for j in ordered if supported[j][0].doc in credible
            ]
        lanes.append([p for p, sim in supported])
    if not lanes:
        return [], False
    relevant = []
    ranked_ids = set()
    for k in range(max(map(len, lanes))):
        for lane in lanes:
            if k < len(lane) and lane[k].id not in ranked_ids:
                relevant.append(lane[k])
                ranked_ids.add(lane[k].id)
    priority = {}
    for rank, p in enumerate(relevant[:24]):
        priority[p.doc] = priority.get(p.doc, 0) + 1 / (rank + 2)
    doc_ids = list(dict.fromkeys(lane[0].doc for lane in lanes))
    # A second document needs its own intent or a direct reference from the primary.
    # Overlapping words such as "сотрудник" do not establish that two procedures apply.
    primary_hits = [p.text.casefold() for p in relevant[:8] if p.doc == doc_ids[0]]
    for did in sorted(priority, key=priority.get, reverse=True):
        if len(lanes) == 1 and not any(
            x.docs[did].title.casefold() in t for t in primary_hits
        ):
            continue
        if len(doc_ids) >= 3:
            break
        if did not in doc_ids:
            doc_ids.append(did)
    groups = {did: [] for did in doc_ids}
    for p in x.rows:
        if p.doc in groups:
            groups[p.doc].append(p)
    # A relevant document may have ranked via an approval sheet. Search its numbered
    # provisions by content too, so that the sheet cannot exhaust the fragment budget.
    rr = local_reranker()
    if rr and len(lanes) == 1 and any(len(group) > 18 for group in groups.values()):
        topic = set(tokens(q))
        numbered = [
            p
            for group in groups.values()
            for p in group
            if p.clause and procedural_passage(p) and len(p.text) >= 45
        ]
        numbered.sort(key=lambda p: -len(topic & set(tokens(p.text))))
        pool = {p.id: p for p in relevant[:16] if procedural_passage(p)}
        for p in numbered[:80]:
            pool.setdefault(p.id, p)
        passages = list(pool.values())
        if passages:
            scores = rr.score(q, [p.text[:1600] for p in passages])
            ranked = [
                p for _, p in sorted(zip(scores, passages), key=lambda pair: -pair[0])
            ]
            relevant = ranked
            lanes = [ranked]
    # Full short regulations retain prerequisites/follow-ups that do not repeat the user's terms.
    # A section with several independently retrieved subclauses defines a coherent scope.
    # Keep independently retrieved material from other documents for cross-document answers.
    primary = doc_ids[0]
    topic_terms = qt - set(
        tokens("порядок действие оформить сделать получить заявка сколько срок")
    )
    headings = []
    if len(groups[primary]) > 18:
        for h in groups[primary]:
            if (
                not h.clause
                or len(h.text) > 165
                or not re.search(
                    r"(?:Порядок|Правила|Процедура|Требования)\s", h.text, re.I
                )
            ):
                continue
            overlap = topic_terms & set(tokens(h.text))
            hits = sum(
                1
                for f in relevant[:9]
                if f.doc == primary and f.clause.startswith(h.clause + ".")
            )
            if len(overlap) >= 2 and hits >= 2:
                headings.append((len(overlap), hits, h))
    if headings:
        heading = max(headings, key=lambda row: (row[0], row[1]))[2]
        groups[primary] = [
            p for p in groups[primary] if p.clause.startswith(heading.clause + ".")
        ]
        permitted = {p.id for p in groups[primary]}
        relevant = [p for p in relevant if p.doc != primary or p.id in permitted]
    candidates = []
    seen = set()

    def include(part):
        if part.id not in seen:
            candidates.append(part)
            seen.add(part.id)

    all_primary = groups[primary]
    branch = selection.matching_branch(q, all_primary)
    if branch:
        p, excerpt = branch
        return [Evidence(1, p, excerpt)], True
    scoped_route = selection.scope_and_followups(q, all_primary)
    if scoped_route and len(scoped_route) <= 10 and len(all_primary) > 18:
        return [Evidence(i, p, p.text) for i, p in enumerate(scoped_route, 1)], True
    after_event = selection.after_event(q, all_primary)
    if after_event and sum(len(p.text) for p in after_event) <= 7000:
        return [Evidence(i, p, p.text) for i, p in enumerate(after_event, 1)], True
    anchors = selection.anchor_parts(q, all_primary)
    focused = selection.passages_for_question(q, all_primary)
    # For a question about responsibility or amendment, include the complete operative
    # provision before generic context. The absence of a directly applicable rule is a gap.
    if focused and (re.search(r"\bкто\b|корректировк", q, re.I)):
        return [Evidence(i, p, p.text) for i, p in enumerate(focused[:7], 1)], True
    for p in anchors:
        include(p)
    for p in focused[:4]:
        include(p)
    # Mark explicit audience restrictions early for the primary retrieved document.
    for p in groups[doc_ids[0]]:
        if re.search(
            r"в трудовых отношениях|применяется (?:только )?к|распространяется (?:только )?на",
            p.text,
            re.I,
        ):
            include(p)
            break
    for lane in lanes:
        for p in lane[:3]:
            if p.doc in groups and (p.doc != primary or p in groups[primary]):
                include(p)
    # A matching section heading is a strong retrieval anchor for all numbered subclauses.
    for hit in relevant[:8]:
        if hit.doc not in groups or not hit.clause or len(hit.text) > 200:
            continue
        if len(qt & set(tokens(hit.text))) < 2:
            continue
        for sibling in groups[hit.doc]:
            if sibling.clause.startswith(hit.clause + "."):
                include(sibling)
    for p in relevant[:8]:
        if p.doc in groups:
            include(p)
    for did in doc_ids:
        group = groups[did]
        if len(group) <= 18:
            for p in group:
                include(p)
        else:
            for i in [i for i, f in enumerate(group) if f in relevant[:9]]:
                for p in group[max(i - 1, 0) : i + 2]:
                    include(p)
            for p in group:
                if re.search(
                    r"распространяется|регламент определяет|трудовых отношениях",
                    p.text,
                    re.I,
                ):
                    include(p)
                    break
    pieces = []
    used = 0
    clipped = False
    for p in candidates:
        # Preserve whole sentences. A partial clause must never silently lose a qualifier.
        if not procedural_passage(p):
            continue
        raw = p.text
        body = display_text(raw)
        if selection.ADMIN.search(body):
            continue
        if (
            not p.sheet
            and not p.clause
            and (
                len(body) < 45
                or re.match(
                    r"^(?:утвержден[ао]?|подписан[ао]?)\s*[:–-]?\s*\d", body, re.I
                )
            )
        ):
            continue
        if not p.sheet and (
            len(tokens(body)) < 4
            or re.fullmatch(
                r"(?:регламент|порядок|инструкция|утвержден[ао]?)(?:\s+[^.!?]+)?",
                body,
                re.I,
            )
            and len(tokens(body)) < 9
        ):
            continue
        if len(raw) > (1700 if p in anchors else 1100):
            spans = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁ])", raw)
            if len(spans) == 1:
                clipped = True
                continue
            selected = sorted(
                range(len(spans)), key=lambda i: -len(qt & set(tokens(spans[i])))
            )[:2]
            excerpts = [
                spans[i] for i in sorted(selected) if 20 <= len(spans[i]) <= 1100
            ]
            clipped = True
        else:
            excerpts = [raw]
        for t in excerpts:
            if len(t) < 14:
                continue
            if used + len(t) > 6200 or len(pieces) >= 22:
                clipped = True
                continue
            pieces.append((p, t))
            used += len(t)
    order = {p.id: i for i, p in enumerate(x.rows)}
    # For compound requests keep the first result from each requested action in reach of the decoder.
    if len(lanes) > 1:
        schedules = []
        for did in doc_ids:
            batch = [(p, t) for p, t in pieces if p.doc == did]
            batch.sort(key=lambda item: order[item[0].id])
            schedules.append(batch)
        pieces = [
            batch[k]
            for k in range(max(map(len, schedules), default=0))
            for batch in schedules
            if k < len(batch)
        ]
        return [Evidence(i, p, t) for i, (p, t) in enumerate(pieces, 1)], clipped
    pieces.sort(
        key=lambda item: (
            doc_ids.index(item[0].doc),
            order[item[0].id],
            item[0].text.find(item[1]),
        )
    )
    return [Evidence(i, p, t) for i, (p, t) in enumerate(pieces, 1)], clipped


COND = re.compile(
    r"\b(?:если|при условии|при наличии|в случае|после|до получения|при отсутствии|при положительном|в течение|при необходимости)\b",
    re.I,
)
PRECEDENCE = re.compile(
    r"\b(?:после|предварительно|затем|до получения|по получении|на основании|по итогам|если)\b",
    re.I,
)


def display_text(text):
    return re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", text).strip()


def requested_relation(question):
    """Extract the core terms of a nominal action, not a hand-maintained topic list."""
    match = re.search(
        r"\b(?:как|нужно|требуется)\s+([а-яё-]+(?:ть|ти|ться))\s+([а-яё-]+)\s+([а-яё-]+)",
        question,
        re.I,
    )
    if not match:
        return set()
    verb, head, complement = match.groups()
    # Genitive complements and "права доступа" restrict the specific operation being asked about.
    if not (
        head.lower().endswith(("цию", "ние", "ния", "права"))
        and complement.lower().endswith(("а", "я", "ов", "ий"))
    ):
        return set()
    return set(tokens(head + " " + complement))


def has_requested_relation(q, evidence):
    terms = requested_relation(q)
    return not terms or any(terms <= set(tokens(e.text)) for e in evidence)


def assemble(plan, evidence, x, allowed, role_name, pending, clipped):
    by_id = {
        e.id: e
        for e in evidence
        if e.part.doc in allowed
        and x.ids.get(e.part.id) == e.part
        and e.text in e.part.text
    }
    notes = []
    steps = []
    accepted = {}

    def conflict(items):
        return Answer(
            "partial",
            "В источниках обнаружено расхождение.",
            [Step(p.text, [p]) for p in items],
            ["Сверьте обе редакции до выполнения действия."],
            role=role_name,
        )

    # Check the active corpus, even when the model elected to omit one alternative.
    for e in by_id.values():
        pair = x.conflicting([e.part], allowed)
        if pair:
            return conflict(pair)
    if plan.status == "conflict" or plan.conflicting:
        pair = list(dict.fromkeys(plan.conflicting))
        if len(pair) >= 2 and all(i in by_id for i in pair):
            return Answer(
                "partial",
                "Модель отметила возможное противоречие.",
                [Step(by_id[i].text, [by_id[i].part]) for i in pair],
                ["Сопоставьте область действия указанных положений."],
                role=role_name,
            )
        return Answer(
            "partial",
            "Противоречие требует проверки.",
            notes=["Не удалось подтвердить обе ссылки."],
            role=role_name,
        )
    if plan.status == "clarify":
        ask = plan.question.strip()
        if not ask.endswith("?") or len(ask) > 180:
            ask = "Уточните вашу роль или условия задачи?"
        return Answer("clarify", ask, pending=pending, role=role_name)
    if plan.status == "missing":
        return Answer(
            "abstain",
            "В найденных документах нет достаточных оснований.",
            role=role_name,
        )
    # A conditional step may explicitly require a numbered action the decoder omitted.
    # Resolve only a unique, whole clause in the same document. Ambiguous references
    # (for example repeated clause numbers in annexes) must not supply a prerequisite.
    represented = {row.action for row in plan.steps}
    scheduled = set()
    ordered_rows = []

    def schedule(row, path):
        e = by_id.get(row.action)
        if not e or row.action in path:
            return
        opening = display_text(e.text).split(",", 1)[0]
        if scheduled and re.match(
            r"(?:после|если|по получении|по итогам)\b", opening, re.I
        ):
            for clause in re.findall(
                r"(?:пункт(?:а|у|е|ом)?|п\.)\s*(\d+(?:\.\d+)*)", opening, re.I
            ):
                matches = [
                    other
                    for other in by_id.values()
                    if other.part.doc == e.part.doc
                    and other.part.clause == clause
                    and other.id != e.id
                ]
                if (
                    len(matches) == 1
                    and matches[0].id not in represented
                    and matches[0].id not in scheduled
                ):
                    prev = matches[0]
                    if prev.text == prev.part.text and prev.id not in path:
                        schedule(generator.ProposedStep(action=prev.id), path | {e.id})
        scheduled.add(row.action)
        ordered_rows.append(row)

    for row in plan.steps:
        if row.action not in by_id:
            notes.append("Пропущено неподтверждённое или повторное действие.")
        else:
            schedule(row, set())
    for row in ordered_rows:
        e = by_id.get(row.action)
        if not e or row.action in accepted:
            notes.append("Пропущено неподтверждённое или повторное действие.")
            continue
        condition = ""
        sources = [e.part]
        deps = []
        if row.condition:
            c = by_id.get(row.condition)
            # A cross-clause condition requires an explicit reference in the action.
            if (
                c
                and COND.search(e.text)
                and (c.part.id == e.part.id or links_clause(e, c))
            ):
                condition = display_text(e.text)
                if c.part.id != e.part.id:
                    sources.append(c.part)
            else:
                notes.append("Отдельное условие шага не подтверждено ссылкой.")
        if not condition and COND.search(e.text):
            condition = display_text(e.text)
        if row.after:
            pred = by_id.get(row.after)
            if (
                row.after in accepted
                and pred
                and PRECEDENCE.search(e.text)
                and links_clause(e, pred)
            ):
                deps = [accepted[row.after]]
                if pred.part not in sources:
                    sources.append(pred.part)
            else:
                notes.append("Зависимость между шагами не подтверждена явной отсылкой.")
        if not deps and PRECEDENCE.search(e.text):
            for earlier, position in accepted.items():
                previous = by_id[earlier]
                if links_clause(e, previous):
                    deps.append(position)
                    if previous.part not in sources:
                        sources.append(previous.part)
        steps.append(Step(display_text(e.text), sources, condition, deps))
        accepted[row.action] = len(steps)
    if plan.unresolved:
        for i in dict.fromkeys(plan.unresolved):
            if i in by_id:
                notes.append(
                    "Недостающие сведения или приложение. "
                    + x.docs[by_id[i].part.doc].title
                    + " · "
                    + by_id[i].part.loc
                )
            else:
                notes.append("Ссылка на недостающие сведения не прошла проверку.")
    if plan.status == "partial" or clipped:
        notes.append(
            "Ответ охватывает найденные фрагменты. Полнота процедуры требует сверки."
        )
    if not steps:
        return Answer(
            "abstain",
            "Не удалось подтвердить действия по предоставленным источникам.",
            notes=list(dict.fromkeys(notes)),
            role=role_name,
        )
    return Answer(
        "partial" if notes else "answer",
        "Порядок действий",
        steps,
        list(dict.fromkeys(notes)),
        role=role_name,
    )


def links_clause(action, other):
    if action.part.doc != other.part.doc or not other.part.clause:
        return False
    return bool(
        re.search(
            r"(?:пункт(?:а|у|е|ом)?|п\.)\s*"
            + re.escape(other.part.clause)
            + r"(?![\d.]\d|\d)",
            action.text,
            re.I,
        )
    )


def answer(q, x, *, selected=None, day=None, known="", pending="", _split=True):
    q = q.strip()
    r = role(q) or known
    if not q or len(q) > 2400:
        return Answer("abstain", "Введите вопрос длиной от 1 до 2400 символов.")
    current_query = q
    if pending:
        q = (
            pending
            if role(q) and len(q.split()) <= 3
            else pending + "\nУточнение пользователя: " + q
        )
    dt = re.search(
        r"(?:на|по состоянию на) (\d\d\.\d\d\.\d{4})", current_query, re.I
    ) or re.search(r"(?:на|по состоянию на) (\d\d\.\d\d\.\d{4})", q, re.I)
    try:
        day = (
            datetime.strptime(dt[1], "%d.%m.%Y").date().isoformat()
            if dt
            else (day or date.today().isoformat())
        )
    except ValueError:
        return Answer("clarify", "Уточните календарную дату?", pending=q)
    allowed = x.available(
        selected, day, bool(re.search(r"истори|редакци|отмен", q, re.I))
    )
    if not allowed:
        return Answer(
            "abstain",
            "В выбранных источниках на эту дату нет доступных документов.",
            role=r,
        )
    doc_request = re.search(
        r"\b(?:приказ|распоряжени[ея])\s*(?:№|N)?\s*(\d+(?:-\d+)?)\b", q, re.I
    )
    if doc_request:
        exact = {did for did in allowed if x.docs[did].number == doc_request[1]}
        if not exact:
            return Answer(
                "abstain",
                "В выбранных источниках на эту дату документ с таким номером не найден.",
                role=r,
            )
        allowed = exact
        if re.match(
            r"\s*(найдите|найти|покажите|показать)\b", q, re.I
        ) and not re.search(r"как|порядок|действи", q, re.I):
            matches = [p for p in x.rows if p.doc in exact]
            refs = []
            for did in sorted(exact):
                d = x.docs[did]
                first = next(p for p in matches if p.doc == did)
                refs.append(
                    Step(
                        f"{d.title}. № {d.number}. Дата подписания: {d.signed or 'не установлена'}.",
                        [first],
                    )
                )
            return Answer(
                "document", "Найден документ на указанную дату.", refs, role=r
            )
    # Generate independent parts separately. The LLM must not spend the whole response
    # budget on the first document or infer dependencies across unrelated regulations.
    if _split and (not pending or role(current_query)):
        intents = re.split(
            r"\s+и\s+(?=[а-яё-]+(?:ть|ти|ться)\b)", q, maxsplit=2, flags=re.I
        )
        if len(intents) > 1:
            # If both requests concern the same regulation, retain shared roles, objects
            # and circumstances in the original question (e.g. after a trip).
            scopes = [retrieve(part, x, allowed)[0] for part in intents]
            docs = [{e.part.doc for e in scope} for scope in scopes]
            if all(len(ids) == 1 for ids in docs) and all(
                ids == docs[0] for ids in docs
            ):
                return answer(q, x, selected=selected, day=day, known=r, _split=False)
            answers = [
                answer(part, x, selected=selected, day=day, known=r, _split=False)
                for part in intents
            ]
            clarification = next((a for a in answers if a.status == "clarify"), None)
            if clarification:
                return Answer("clarify", clarification.summary, pending=q, role=r)
            steps = []
            notes = []
            for part, a in zip(intents, answers):
                offset = len(steps)
                steps.extend(
                    Step(
                        step.text,
                        step.sources,
                        step.condition,
                        [i + offset for i in step.after],
                    )
                    for step in a.steps
                )
                if not a.steps:
                    notes.append(f"По части запроса «{part.strip()}»: {a.summary}")
                notes.extend(a.notes)
            status = (
                "partial"
                if steps and any(a.status != "answer" for a in answers)
                else "answer"
                if steps
                else "abstain"
            )
            return Answer(
                status,
                "Результат по частям запроса",
                steps,
                list(dict.fromkeys(notes)),
                role=r,
            )
    evidence, clipped = retrieve(q, x, allowed)
    if not evidence:
        return Answer(
            "abstain", "Для вопроса не найдено подтверждающих фрагментов.", role=r
        )

    # Proper names following a location preposition restrict the requested application.
    # An unseen named location must not silently turn into a procedure for a different one.
    locations = re.findall(r"\b(?:на|из|в)\s+([А-ЯЁ][а-яё]{2,})\b", q)
    corpus_terms = set(
        t
        for item in evidence
        for t in tokens(item.text + " " + x.docs[item.part.doc].title)
    )
    if any(not (set(tokens(name)) & corpus_terms) for name in locations):
        return Answer(
            "abstain",
            "Найденные источники не подтверждают применение процедуры для указанного объекта.",
            role=r,
        )

    if re.search(
        r"\b(?:изменить|скорректировать|пересмотреть)\b", q, re.I
    ) and re.search(r"\b(?:19|20)\d{2}\s+год", q, re.I):
        if not any(
            re.search(r"изменени|корректир|пересмотр|внесен", e.text, re.I)
            and set(tokens(q)) & set(tokens(e.text))
            for e in evidence
        ):
            return Answer(
                "abstain",
                "Источники не подтверждают процедуру изменения на указанный год.",
                role=r,
            )
    if not has_requested_relation(q, evidence):
        return Answer(
            "abstain",
            "В найденных пунктах не подтверждён порядок запрошенной операции.",
            role=r,
        )

    def scopes(e):
        s = norm(e.text)
        if re.search(
            r"(распространяется|применяется|в трудовых отношениях|состоит в трудовых отношениях|состоящ.{0,9}в трудовых отношениях)",
            s,
        ):
            if re.search(r"студент|обучающ", s):
                return "student"
            if re.search(r"работник|сотрудник|трудовых отношениях", s):
                return "employee"
        return ""

    scoped = [
        (e, scopes(e))
        for e in evidence
        if e.part.doc == evidence[0].part.doc and scopes(e)
    ]
    if (
        scoped
        and r in ("", "both")
        and (
            r == "both"
            or re.search(r"\b(я|мне|хочу|могу|нам|мы)\b", q, re.I)
            or re.search(r"как (оформить|поехать|получить|подать)", q, re.I)
        )
    ):
        return Answer(
            "clarify",
            "Вы едете как студент или как сотрудник?"
            if r == "both"
            else "Вы студент или сотрудник?",
            pending=q,
            role=r,
        )
    if r in ("student", "employee"):
        excluded = {e.part.doc for e, s in scoped if s != r} - {
            e.part.doc for e, s in scoped if s == r
        }
        evidence = [e for e in evidence if e.part.doc not in excluded]
        if not evidence:
            return Answer(
                "abstain",
                "В выбранных документах нет процедуры для указанной роли.",
                [Step(e.text, [e.part]) for e, s in scoped[:1]],
                role=r,
            )
    if evidence:
        ask = selection.regulated_object(
            q, [e.part for e in evidence], x.docs[evidence[0].part.doc].title
        )
        if ask:
            title = x.docs[evidence[0].part.doc].title
            if pending and not selection.confirmed_scope(current_query, title):
                return Answer(
                    "abstain",
                    "Применимость документа «"
                    + title
                    + "» к указанному помещению не подтверждена.",
                    role=r,
                )
            if not pending:
                return Answer("clarify", ask, pending=q, role=r)
    conflicts = x.conflicting([e.part for e in evidence], allowed)
    if conflicts:
        return Answer(
            "partial",
            "В выбранных документах указаны разные значения.",
            [Step(p.text, [p]) for p in conflicts],
            ["Сверьте редакции и область применения."],
            role=r,
        )
    fragments = [
        {
            "id": e.id,
            "document": x.docs[e.part.doc].title,
            "location": e.part.loc,
            "text": e.text,
        }
        for e in evidence
    ]
    try:
        plan = generator.generate(
            {
                "question": (
                    q + "\nУточнение пользователя: " + current_query
                    if pending and q == pending
                    else q
                ),
                "role": r,
                "as_of": day,
                "fragments": fragments,
            }
        )
    except generator.ModelUnavailable as exc:
        return Answer(
            "unavailable",
            "Пошаговая инструкция не сформирована.",
            notes=[str(exc)],
            role=r,
        )
    # A whole short document already fits into the decoder. Do not turn a
    # single relevant selection in that document into an inferred wider workflow.
    if (
        plan.status in ("steps", "partial")
        and evidence
        and (clipped or len(evidence) > 6)
    ):
        primary = evidence[0].part.doc
        group = list(
            {e.part.id: e.part for e in evidence if e.part.doc == primary}.values()
        )
        branch = selection.matching_branch(q, group)
        required = selection.anchor_parts(q, group)
        scoped_route = selection.scope_and_followups(q, group)
        required.extend(scoped_route)
        required.extend(e.part for e in selection.event_obligations(q, evidence))
        focused = selection.passages_for_question(q, group)
        if branch:
            required = [branch[0]]
        elif focused and (re.search(r"\bкто\b|корректировк", q, re.I)):
            required = focused
        wanted = {p.id for p in required}
        # A complete quoted obligation survives even if the decoder omitted its id.
        ids = {row.action for row in plan.steps}
        for e in evidence:
            if (
                e.part.id in wanted
                and (e.text == e.part.text or branch)
                and e.id not in ids
            ):
                plan.steps.append(generator.ProposedStep(action=e.id))
                ids.add(e.id)
        if wanted:
            position = {e.id: i for i, e in enumerate(evidence)}
            plan.steps.sort(key=lambda row: position.get(row.action, len(position)))
        result = assemble(plan, evidence, x, allowed, r, q, clipped)
        if branch and result.steps:
            result.status = "partial"
            result.notes.append(
                "Пункт определяет применимую ветвь. Подробные этапы для этой категории в выбранных основаниях не установлены."
            )
        if result.steps and re.search(r"\bкто\b", q, re.I) and focused:
            # The category of equipment may be regulated without explicitly naming the item.
            subject = re.search(r"(?:неисправность|поломка)\s+([а-яё-]+)", q, re.I)
            if subject and not any(
                set(tokens(subject[1])) <= set(tokens(e.text)) for e in evidence
            ):
                result.status = "partial"
                result.notes.append(
                    "Конкретный объект из вопроса не назван в источнике. Проверьте, входит ли он в перечисленные категории."
                )
        return result
    return assemble(plan, evidence, x, allowed, r, q, clipped)


class Citation(generator.Strict):
    document_id: str
    fragment_id: str
    title: str
    filename: str
    signed: str
    registered: str
    effective: str
    location: str
    clause: str
    quote: str


class InstructionStep(generator.Strict):
    number: int
    action: str
    condition: str
    after: list[int]
    evidence: list[Citation]


class Instruction(generator.Strict):
    status: str
    summary: str
    steps: list[InstructionStep]
    notes: list[str]


def structured(result, index):
    """User-facing JSON. Source metadata comes from the index, never from the LLM."""
    steps = []
    for i, step in enumerate(result.steps, 1):
        citations = []
        for ref in step.sources:
            f = index.ids[ref.id]
            d = index.docs[f.doc]
            citations.append(
                Citation(
                    document_id=d.id,
                    fragment_id=f.id,
                    title=d.title,
                    filename=d.name,
                    signed=d.signed,
                    registered=d.registered,
                    effective=d.effective,
                    location=f.loc,
                    clause=f.clause,
                    quote=f.text,
                )
            )
        steps.append(
            InstructionStep(
                number=i,
                action=step.text,
                condition=step.condition,
                after=step.after,
                evidence=citations,
            )
        )
    return Instruction(
        status=result.status, summary=result.summary, steps=steps, notes=result.notes
    )
