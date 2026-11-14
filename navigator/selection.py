"""Select operative clauses within one applicable regulation. No procedure identifiers."""

import re
from .search import tokens


def words(text):
    return set(tokens(text))


def body(part):
    return re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", part.text).strip()


# Cover document submissions, review and authorisation regardless of the type of procedure.
RECORD = re.compile(
    r"заяв[кло]|обращени|записк|распоряжени|смет[ауы]|отч[её]т|карточк|договор", re.I
)
OPERATIVE = re.compile(
    r"пода[еёю]|подать|представи|представля|направля|составля|рассматрив|согласу|согласов|утвержда|утверди|подпис|инициир|заполня|доводится|заключить|регистрир|принимает|присваив|вносится",
    re.I,
)
RESTRICTION = re.compile(
    r"возможн[оаы]? только|при выполнении следующих условий|не выдаются в следующих случаях|в (?:таком|этом) случае.{0,50}не |за исключением|кроме\s",
    re.I,
)
ADMIN = re.compile(
    r"^(?:министерство |считать утратившим|отменить действие|контроль за исполнением|ввести в (?:действие|промышленную эксплуатацию)|утвердить (?:регламент|бланки)|заведующ.{0,25}канцеляр|бланк |форма приказа|приложение\s|лист согласования)|место для ввода|выберите элемент",
    re.I,
)
GENERIC = words(
    "как оформить организация организовать новый какие действия порядок нужен нужный получить проект сотрудник работник для между подразделениями делать делать провести"
)
WORKFLOW = re.compile(
    r"\b(?:какие действия|весь порядок|вся процедура|все этапы|пошагов\w*)\b", re.I
)
# Numbered branch lists separate alternative audiences in a single provision.
EXCLUSION = re.compile(r"\b(?:кроме|за исключением)\s+([^);.:]+)", re.I)


def matching_branch(query, parts):
    query_terms = words(query) - words("как организовать оформить")
    for p in parts:
        if not p.clause or not EXCLUSION.search(p.text):
            continue
        alternatives = re.split(r";\s*[-–—]\s*", p.text)
        if len(alternatives) < 2:
            continue
        for alternative in alternatives:
            for exclusion in EXCLUSION.findall(alternative):
                target = words(exclusion)
                if len(target) < 2 or not target <= query_terms:
                    continue
                for branch in alternatives:
                    if (
                        branch != alternative
                        and target <= words(branch)
                        and not EXCLUSION.search(branch)
                    ):
                        return p, branch.strip()
    return None


def anchor_parts(query, parts):
    """Find explicit limits and the document workflow that the decoder must cover."""
    qt = words(query)
    topic = qt - GENERIC
    anchors = {}
    for p in parts:
        t = body(p)
        if not p.clause or ADMIN.search(t) or len(t) > 1700:
            continue
        terms = words(t)
        relevant = bool(topic & terms)
        if (
            RESTRICTION.search(t)
            and (qt & terms or WORKFLOW.search(query))
            and (
                relevant
                or WORKFLOW.search(query)
                or re.search(
                    r"возможн[оаы]? только|при выполнении следующих условий", t, re.I
                )
            )
        ):
            anchors[p.id] = p
        if (
            WORKFLOW.search(query)
            and relevant
            and re.search(
                r"может предоставляться|может составлять|при принятии решения|рассматриваются",
                t,
                re.I,
            )
        ):
            anchors[p.id] = p
        if (
            re.search(r"согласова(?:ть|ни)", query, re.I)
            and re.search(r"запускается процесс согласования", t, re.I)
            and relevant
        ):
            anchors[p.id] = p
        if WORKFLOW.search(query) and RECORD.search(t) and OPERATIVE.search(t):
            if relevant or re.match(
                r"(?:заявк|заявлен|при принятии положительн)", t, re.I
            ):
                anchors[p.id] = p
    return list(anchors.values())


def passages_for_question(query, parts):
    """For a narrow question favor clauses naming the operation or the responsible actor."""
    specific = words(query) - GENERIC
    if not specific:
        return []
    who = bool(re.search(r"\bкто\b", query, re.I))
    selected = []
    for p in parts:
        t = body(p)
        if not p.clause or len(t) > 1600 or ADMIN.search(t):
            continue
        matched = words(t) & specific
        if who:
            # Stemmed predicate plus the same object, or an explicit service restriction.
            if any(
                v.startswith(("принима", "прием", "приём", "обрабатыва"))
                for v in specific
            ):
                if re.search(
                    r"принимает и обрабатывает|осуществляет при[её]м обращений|не обрабатывает обращения",
                    t,
                    re.I,
                ):
                    selected.append(p)
        elif re.search(r"корректировк|изменени", query, re.I):
            if (
                re.search(
                    r"корректировк.{0,70}(?:внос|осуществля|соглас)|изменени.{0,50}внос",
                    t,
                    re.I,
                )
                and len(matched) >= 2
            ):
                selected.append(p)
        elif specific <= words(t) and OPERATIVE.search(t):
            selected.append(p)
    return selected


def scope_and_followups(query, parts):
    """Eligibility provisions can be decisive even if users use a different name for the service."""
    audience = words(query)
    limits = [
        p
        for p in parts
        if p.clause
        and re.search(
            r"возможн[оаы]? только|при выполнении следующих условий", body(p), re.I
        )
        and audience & words(p.text)
    ]
    if not limits:
        return []
    operative = [
        p
        for p in parts
        if p.clause
        and not ADMIN.search(body(p))
        and re.search(
            r"пода[её]тся|организует сбор заявок|формирует списки|решение.{0,35}доводится|должен заключить",
            body(p),
            re.I,
        )
    ]
    return list({p.id: p for p in limits + operative}.values())


def regulated_object(query, parts, title):
    """Ask about an unverified adjective when the named facility differs from the scope."""
    phrases = re.findall(
        r"\b([а-яё]+(?:ый|ой|ий|ую|ое))\s+([а-яё]{3,})\b", query.casefold()
    )
    title_terms = words(title)
    for modifier, noun in phrases:
        noun_terms = words(noun)
        if (
            not noun_terms
            or not noun_terms <= title_terms
            or words(modifier) <= title_terms
        ):
            continue
        # The restriction concerns a named physical facility, not e.g. a "new" document.
        if noun not in {"зал", "кабинет", "корпус", "лабораторию", "помещение"}:
            continue
        if not any(words(modifier + " " + noun) <= words(p.text) for p in parts):
            return f"Вы имеете в виду объект из регламента «{title}»? Уточните название помещения."
    return ""


def after_event(query, group):
    match = re.search(r"\bпосле\s+([а-яё]{4,})", query, re.I)
    if not match or len(group) < 18:
        return []
    event = words(match[1])
    at = next(
        (
            i
            for i, p in enumerate(group)
            if p.clause
            and event & words(p.text)
            and re.search(
                r"(?:по возвращении|после (?:окончания|завершения|возвращения)).{0,90}(?:обязан|представ|направ)",
                p.text,
                re.I,
            )
        ),
        None,
    )
    if at is None:
        return []
    block = []
    for p in group[at : at + 14]:
        if not p.clause or not (
            RECORD.search(p.text)
            or re.search(
                r"задолжен|возврат|расч[её]т|проверя|удостовер|утвержд", p.text, re.I
            )
        ):
            break
        block.append(p)
    return block


def confirmed_scope(reply, title):
    if re.search(r"\b(?:нет|не)\b", reply, re.I):
        return False
    if re.fullmatch(r"да[.!]?", reply.strip(), re.I):
        return True
    named = words(title)
    return len(words(reply) & named) >= 2


def event_obligations(query, evidence):
    if not re.search(r"\bпосле\s+[а-яё]{4,}", query, re.I):
        return []
    return [
        e
        for e in evidence
        if re.search(
            r"по возвращении.{0,85}обязан|к отчету.{0,60}прилагаются|(?:погашени|погашении|погасить|удержани).{0,100}(?:задолжен|аванс)|задолжен.{0,120}удержан|окончательный расчет",
            e.text.replace("ё", "е"),
            re.I,
        )
    ]
