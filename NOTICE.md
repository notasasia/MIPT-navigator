# Компоненты и исходные материалы

- Папка corpus содержит четыре синтетических документа. Полный набор организаторов используется в локальном комплекте.
- Базовая языковая модель — RuBERT-tiny автора cointegrated. В карточке модели указана лицензия MIT. [Карточка базовой модели](https://huggingface.co/cointegrated/rubert-tiny)
- Квантованная ONNX-версия и токенизатор взяты из onnx-community/rubert-tiny-ONNX, ревизия `9a8d2ed31c37246d2a8d4d9f500dc1ca66563114`. Из графа удалена голова предобучения и сохранён выход последнего скрытого слоя. Для представления текста используется нормированный вектор токена CLS размерности 312. Контрольные суммы зафиксированы в `data/model/weights.json`. [Карточка ONNX-версии](https://huggingface.co/onnx-community/rubert-tiny-ONNX)
- Файл русского языка Tesseract входит в поставку пакета `tesseract-ocr-rus`. Условия Apache License 2.0 приведены в `data/ocr/LICENSE`. [Данные Tesseract](https://github.com/tesseract-ocr/tessdata_fast)
- Интерфейс использует Streamlit. Документы читают PyMuPDF, python-docx и openpyxl. Поиск использует ONNX Runtime, tokenizers, snowballstemmer и NumPy. Версии зафиксированы в `requirements.txt`. При установке зависимости сохраняют свои лицензии.

Генеративная модель Qwen2.5-1.5B-Instruct разработана Qwen. Для установки используется официальная сборка GGUF Q4_K_M. Веса загружаются отдельно на условиях карточки модели. [Модель и условия](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

Локальное повторное ранжирование использует cross-encoder/mmarco-mMiniLMv2-L12-H384-v1. При установке загружаются ONNX INT8 и токенизатор из одной зафиксированной ревизии. URL, размеры и SHA-256 приведены в models/reranker.json. https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
