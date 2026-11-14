"""Check a live Streamlit deployment in a fresh browser using only a synthetic DOCX."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from docx import Document
from playwright.sync_api import sync_playwright

VERSION = (
    Path(__file__).resolve().parents[1].joinpath("VERSION.txt").read_text().strip()
)


def probe(url, output, executable=None):
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Укажите полный HTTP(S) адрес приложения.")
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as playwright:
        fixture = Path(tmp) / "jury-spectrometer.docx"
        doc = Document()
        case = json.loads(
            (
                Path(__file__).resolve().parents[1] / "tests/fixtures/unseen.json"
            ).read_text()
        )
        for paragraph in case["paragraphs"]:
            doc.add_paragraph(paragraph)
        doc.save(fixture)
        browser = playwright.chromium.launch(
            executable_path=executable, args=["--disable-dev-shm-usage"]
        )
        try:
            page = browser.new_page(locale="ru-RU")
            page.set_default_timeout(180_000)
            page.goto(url, wait_until="domcontentloaded")
            page.get_by_text("Физтех.Навигатор · " + VERSION, exact=True).wait_for(
                timeout=600_000
            )
            page.get_by_text("Библиотека", exact=True).click()
            page.locator("input[type=file]").set_input_files(fixture)
            page.get_by_role("button", name="Индексировать", exact=True).click()
            page.get_by_text("Обработано: 1", exact=False).first.wait_for()
            page.get_by_text("Навигатор", exact=True).click()
            page.get_by_text("Документы для поиска", exact=True).click()
            tags = page.locator('[data-testid="stMultiSelect"] [data-baseweb="tag"]')
            for name in tags.all_text_contents():
                if fixture.stem not in name:
                    tags.filter(has_text=name).locator("[role=button],svg").last.click()
            page.get_by_text("Документы для поиска", exact=True).click()
            input_box = page.get_by_role("textbox", name="Ваш вопрос о процедуре")
            input_box.fill(case["question"])
            input_box.press("Enter")
            download = page.get_by_role("button", name="Скачать инструкцию в JSON")
            download.wait_for()
            with page.expect_download() as transfer:
                download.click()
            path = Path(tmp) / "answer.json"
            transfer.value.save_as(path)
            answer = json.loads(path.read_text())
            steps = answer["steps"]
            assert [s["evidence"][0]["clause"] for s in steps] == [
                "1",
                "2",
                "3",
                "4",
            ], answer["status"]
            assert all(
                s["evidence"][0]["filename"] == fixture.name
                and s["evidence"][0]["signed"] == "2026-11-12"
                and s["action"] in s["evidence"][0]["quote"]
                for s in steps
            )
            assert steps[2]["condition"] and steps[3]["after"] == [3]
            page.get_by_role("button", name="Открыть основание").last.click()
            page.get_by_role("dialog").get_by_text(
                fixture.name, exact=False
            ).first.wait_for()
            output.write_text(
                json.dumps(
                    {
                        "url": url,
                        "version": VERSION,
                        "question": case["question"],
                        "answer": answer,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(
                "Проверены сборка, загрузка DOCX, четыре шага, зависимости и открытие источника.",
                flush=True,
            )
        finally:
            browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("--out", type=Path, default=Path("deployment-check.json"))
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"))
    args = ap.parse_args()
    probe(args.url, args.out, args.chromium)
