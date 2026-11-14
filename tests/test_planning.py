import os, json
from io import BytesIO
from pathlib import Path
from docx import Document
from pydantic import ValidationError
import pytest
from navigator import generator
from navigator.answer import answer
from navigator.corpus import ROOT, Store
from navigator.search import Search


@pytest.fixture
def policy(tmp_path):
    case = json.loads((ROOT / "tests/fixtures/unseen.json").read_text())
    doc = Document()
    for line in case["paragraphs"]:
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    store = Store(tmp_path)
    store.load(case["name"], buf.getvalue())
    return Search(store), case


def test_reference_validation_and_dependencies(policy, monkeypatch):
    ix, c = policy

    def propose(payload):
        ids = {f["location"].split(" · ")[0]: f["id"] for f in payload["fragments"]}
        return generator.Plan(
            status="steps",
            steps=[
                {"action": ids["п. 1"]},
                {"action": ids["п. 2"], "after": ids["п. 1"]},
                {"action": ids["п. 3"], "condition": ids["п. 3"]},
                {"action": 999},
                {"action": ids["п. 4"], "after": 999},
            ],
        )

    monkeypatch.setattr(generator, "generate", propose)
    a = answer(c["question"], ix, day="2026-11-14")
    assert len(a.steps) == 4 and a.status == "partial"
    assert a.steps[1].after == [1] and a.steps[2].condition.startswith("Если")
    assert a.steps[3].after == [3] and any("не подтверждена" in s for s in a.notes)
    assert next(iter(ix.docs.values())).signed == "2026-11-12"
    assert answer(c["question"], ix, day="2026-11-10").status == "abstain"


def test_missing_weights_do_not_produce_ungrounded_answer(
    policy, monkeypatch, tmp_path
):
    ix, c = policy
    monkeypatch.setenv("NAV_LLM_MODEL", str(tmp_path / "absent.gguf"))
    a = answer(c["question"], ix)
    assert a.status == "unavailable" and not a.steps


def test_schema_rejects_extra_hallucinated_action_text():
    with pytest.raises(ValidationError):
        generator.Plan.model_validate(
            {"status": "steps", "steps": [{"action": 1, "text": "Перечислить деньги"}]}
        )


def test_actual_local_llm_on_unseen_regulation(policy):
    model = Path(
        os.environ.get(
            "NAV_LLM_MODEL",
            ROOT
            / "models"
            / json.loads((ROOT / "models/generator.json").read_text())["filename"],
        )
    )
    if not model.exists():
        pytest.skip("Установите GGUF для интеграционного теста")
    ix, c = policy
    out = answer(c["question"], ix, day="2026-11-14")
    assert [st.sources[0].clause for st in out.steps] == ["1", "2", "3", "4"], out
    assert out.steps[2].condition and out.steps[3].after == [3]
    assert all(st.text in st.sources[0].text for st in out.steps)


def test_missing_second_conflict_source_does_not_advertise_a_verified_conflict(
    policy, monkeypatch
):
    ix, c = policy
    monkeypatch.setattr(
        generator,
        "generate",
        lambda payload: generator.Plan(
            status="conflict", steps=[], conflicting=[1, 889]
        ),
    )
    out = answer(c["question"], ix)
    assert (
        out.status == "partial"
        and not out.steps
        and any("обе ссылки" in n for n in out.notes)
    )


def test_role_clarification_and_other_role_exclusion(tmp_path, monkeypatch):
    doc = Document()
    doc.add_paragraph(
        "1. Регламент применяется только к сотрудникам, состоящим в трудовых отношениях с организацией."
    )
    doc.add_paragraph(
        "2. Сотрудник отправляет заявку на получение образца кварца в отдел геологии."
    )
    buf = BytesIO()
    doc.save(buf)
    store = Store(tmp_path)
    store.load("Образец кварца.docx", buf.getvalue())
    ix = Search(store)
    a = answer("Как мне получить образец кварца?", ix)
    assert a.status == "clarify" and a.pending
    student = answer("Как студент", ix, pending=a.pending)
    assert student.status == "abstain" and "роли" in student.summary

    def propose(payload):
        assert "Как сотрудник" in payload["question"]
        f = next(i for i in payload["fragments"] if "заявку" in i["text"])
        return generator.Plan(status="steps", steps=[{"action": f["id"]}])

    monkeypatch.setattr(generator, "generate", propose)
    emp = answer("Как сотрудник", ix, pending=a.pending)
    assert emp.steps and "отдел геологии" in emp.steps[0].text


def test_steps_are_grounded_after_inserting_unrelated_paragraph(policy, monkeypatch):
    ix, c = policy
    monkeypatch.setattr(
        generator,
        "generate",
        lambda payload: generator.Plan(
            status="partial",
            steps=[
                {
                    "action": next(
                        f["id"]
                        for f in payload["fragments"]
                        if "подписывает акт" in f["text"]
                    )
                }
            ],
        ),
    )
    from navigator.answer import structured

    result = structured(answer(c["question"], ix), ix)
    ref = result.steps[0].evidence[0]
    assert ref.signed == "2026-11-12" and ref.clause == "3" and "абз." in ref.location
    assert result.steps[0].action in ref.quote
