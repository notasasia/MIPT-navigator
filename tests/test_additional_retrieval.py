from types import SimpleNamespace
from navigator.answer import retrieve, role
from navigator.corpus import Part


def test_semantic_only_candidate_enters_planner(monkeypatch):
    part = Part(
        "fragment",
        "doc",
        "1. Для проведения опыта лаборатория выдаёт сотруднику измерительное оборудование по описи.",
        "п. 1",
        clause="1",
    )
    docs = {"doc": SimpleNamespace(title="Учебная инструкция")}
    index = SimpleNamespace(
        docs=docs, rows=[part], find=lambda q, allowed, limit: [(part, 0.0, 0.92)]
    )
    evidence, _ = retrieve("Инвентаризация оснащения", index, {"doc"})
    assert [e.part.id for e in evidence] == ["fragment"]
    index.find = lambda q, allowed, limit: [(part, 0.0, 0.41)]
    assert retrieve("Инвентаризация оснащения", index, {"doc"})[0] == []


def test_simultaneous_roles_with_adverb():
    assert role("Я аспирант и ещё работаю в лаборатории") == "both"
    assert role("Я аспирант и не работаю в лаборатории") == "student"


def test_compound_question_preserves_two_sources(monkeypatch):
    first = Part(
        "open",
        "one",
        "1. Лаборант размещает извещение о выдаче оборудования в журнале закупок.",
        "п. 1",
        clause="1",
    )
    second = Part(
        "close",
        "two",
        "5. Чтобы вернуть оборудование, подпишите акт передачи на склад.",
        "п. 5",
        clause="5",
    )
    index = SimpleNamespace(
        docs={
            "one": SimpleNamespace(title="Разместить извещение"),
            "two": SimpleNamespace(title="Вернуть оборудование"),
        },
        rows=[first, second],
    )
    index.find = lambda q, allowed, limit: (
        [(first, 5.0, 0.71)] if "разместить" in q.lower() else [(second, 7.0, 0.66)]
    )
    found, _ = retrieve(
        "Как разместить извещение и вернуть оборудование?", index, set(index.docs)
    )
    assert {e.part.id for e in found} == {"open", "close"}


def test_rights_are_distinct_from_signing_and_service_access():
    from navigator.answer import Evidence, has_requested_relation

    p = Part(
        "f", "d", "Входящие соглашения подписываются электронной подписью.", "п. 7"
    )
    q = "Как назначить права доступа к системе учёта?"
    assert not has_requested_relation(q, [Evidence(1, p, p.text)])
    assert has_requested_relation(
        q, [Evidence(1, p, "Права доступа назначает администратор.")]
    )
    q = "Как заказать регистрацию пропуска в ЦОД для подрядчика?"
    assert not has_requested_relation(
        q,
        [
            Evidence(
                1,
                p,
                "Заявка на виртуальную машину служит основанием для доступа к сервису.",
            )
        ],
    )


def test_named_location_not_substituted(monkeypatch):
    from navigator.answer import Evidence, answer
    import importlib

    mod = importlib.import_module("navigator.answer")
    p = Part(
        "f",
        "d",
        "1. Начальник отдела выдаёт ключ от кабинета на складе.",
        "п. 1",
        clause="1",
    )
    ix = SimpleNamespace(
        docs={"d": SimpleNamespace(title="Правила выдачи ключей")},
        available=lambda *a: {"d"},
    )
    monkeypatch.setattr(mod, "retrieve", lambda *a: ([Evidence(1, p, p.text)], False))
    a = answer("Кто разрешит запуск ускорителя на Марсе?", ix)
    assert a.status == "abstain" and not a.steps


def test_split_preserves_actions_and_local_dependency_numbering(monkeypatch):
    import importlib
    from navigator.generator import Plan

    mod = importlib.import_module("navigator.answer")
    parts = [
        Part(
            str(i),
            "d1" if i <= 2 else "d2",
            f"{i}. Сотрудник фиксирует действие по документу {i}.",
            f"п. {i}",
            clause=str(i),
        )
        for i in range(1, 5)
    ]
    ix = SimpleNamespace(
        docs={key: SimpleNamespace(title="Порядок") for key in ("d1", "d2")},
        ids={p.id: p for p in parts},
        available=lambda *a: {"d1", "d2"},
        conflicting=lambda *a: [],
    )

    def collect(q, *_):
        pair = parts[:2] if "открыть" in q else parts[2:]
        return [mod.Evidence(i, p, p.text) for i, p in enumerate(pair, 1)], False

    monkeypatch.setattr(mod, "retrieve", collect)
    monkeypatch.setattr(
        mod.generator,
        "generate",
        lambda payload: Plan(status="steps", steps=[{"action": 1}, {"action": 2}]),
    )
    a = mod.answer("Как открыть журнал и закрыть ведомость?", ix)
    assert [st.sources[0].clause for st in a.steps] == ["1", "2", "3", "4"]


def test_exception_uses_positive_branch_of_new_rule():
    from navigator.selection import matching_branch

    part = Part(
        "f",
        "doc",
        "1. Без калибровки принимают оборудование (кроме оптических датчиков); - оптические датчики проходят калибровку до подачи заявки.",
        "п. 1",
        clause="1",
    )
    match = matching_branch("Как принять оптический датчик?", [part])
    assert match and match[0] is part
    assert match[1] in part.text and "проходят калибровку" in match[1]
    assert matching_branch("Как принять оптический фильтр?", [part]) is None


def test_obligations_collected_without_golden_route_or_clause_numbers():
    from navigator.selection import anchor_parts

    parts = [
        Part(str(n), "d", s, f"п. {n}", clause=str(n))
        for n, s in [
            (
                42,
                "42. Заявка на резервирование составляется по форме и направляется координатору.",
            ),
            (57, "57. Заявки рассматриваются в техническом отделе."),
            (
                93,
                "93. При принятии положительного решения заявки резервирование утверждается координатором до 25 числа.",
            ),
            (101, "101. Контроль за исполнением приказа возложить на канцелярию."),
        ]
    ]
    result = anchor_parts(
        "Какие действия нужны для резервирования оборудования?", parts
    )
    assert [p.clause for p in result] == ["42", "57", "93"]


def test_unknown_room_needs_clarification_even_with_similar_source():
    from navigator.selection import regulated_object

    p = Part("z", "d", "Помещение оснащено проектором для показа фильмов.", "стр. 1")
    assert regulated_object(
        "Как забронировать учебный зал?", [p], "Бронирование концертного зала"
    )
    assert not regulated_object(
        "Как забронировать концертный зал?", [p], "Бронирование концертного зала"
    )
