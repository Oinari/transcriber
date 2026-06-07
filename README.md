# transcriber

Локальный транскрибатор стримов и подкастов: аудио/видео → русскоязычный конспект + таймкодированный транскрипт в Markdown.

**Что делает:**
- Транскрибирует аудио/видео через [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (EN / JA / ID и другие языки)
- Переводит транскрипт на русский через локальную LLM (Ollama) или OpenAI-совместимый сервер
- Генерирует структурированный Markdown: вступление, тематические блоки с прозой, хронологический транскрипт с таймкодами
- Опционально: построчный перевод каждого сегмента и субтитры (`.ass`)

---

## Зависимости

- Python 3.11+
- [Ollama](https://ollama.com/) с загруженной моделью (по умолчанию `qwen3:14b`) **или** любой OpenAI-совместимый сервер
- NVIDIA GPU (рекомендуется; CPU тоже работает, но медленно)
- CUDA-библиотеки (cublas, cudnn) — устанавливаются автоматически через pip

---

## Подготовка окружения

```bash
# Создать виртуальное окружение
python3 -m venv .venv

# Активировать
source .venv/bin/activate

# Установить зависимости
pip install faster-whisper typer rich pydantic ollama httpx

# Деактивировать когда не нужно
deactivate
```

Скрипт `transcribe` автоматически подхватывает `.venv` и выставляет нужные `LD_LIBRARY_PATH` для CUDA — активировать venv вручную перед запуском не нужно.

---

## Запуск

### Базовый — транскрипция + конспект

```bash
./transcribe audio.mp3
```

Создаёт `audio.md` рядом с входным файлом.

### С указанием языка

```bash
./transcribe stream.mp4 --lang ja
./transcribe podcast.m4a --lang en
```

### Быстрый режим (whisper-medium вместо large-v3)

```bash
./transcribe stream.mp4 --fast
```

### Указать модель Whisper вручную

```bash
./transcribe audio.mp3 --model medium
./transcribe audio.mp3 --model large-v3-turbo
```

### С профилем стримера

Профили хранятся в `streamers/<name>.md` — помогают LLM правильно расставить имена, местоимения и контекст.

```bash
# Создать профиль из шаблона
cp streamers/_template.md streamers/korone.md
# Отредактировать профиль, затем:
./transcribe vod.mp4 --streamer korone
```

### С построчным переводом сегментов

Добавляет колонку с русским переводом в таблицу транскрипта:

```bash
./transcribe stream.mp4 --translate-segments
```

### Генерация субтитров (.ass)

```bash
./transcribe stream.mp4 --subtitles
```

Создаёт `stream.orig.ass` (оригинал) и `stream.ru.ass` (русский перевод).

### Использовать OpenAI-совместимый сервер вместо Ollama

```bash
./transcribe audio.mp3 --llm-host http://localhost:8080 --llm-model gpt-4o-mini
```

### Переиспользовать кеш транскрипта (пропустить Whisper)

Если `audio.segments.json` уже существует — пропустить транскрипцию и сразу перейти к LLM:

```bash
./transcribe audio.mp3 --reuse-transcript
```

### Перезаписать существующий output

```bash
./transcribe audio.mp3 --force
```

### Глоссарий (термины не переводить / переводить особым образом)

```bash
./transcribe audio.mp3 --glossary glossary.txt
```

Формат файла: одна строка — одна пара `оригинал = перевод`.

---

## Все опции

```
./transcribe --help
```

| Опция | По умолчанию | Описание |
|---|---|---|
| `--model` / `-m` | `large-v3` | Модель Whisper |
| `--lang` / `-l` | авто | Язык оригинала (`en`, `ja`, `id`, ...) |
| `--fast` | false | Использовать `whisper-medium` |
| `--beam-size` | 1 | Beam size (1 = жадный, быстрый; 5 = точный) |
| `--vad/--no-vad` | vad | Voice activity detection |
| `--force` / `-f` | false | Перезаписать существующий `.md` |
| `--ollama-model` | `qwen3:14b` | Модель Ollama для LLM-шагов |
| `--llm-host` | — | URL OpenAI-совместимого сервера |
| `--llm-model` | `gpt-4o-mini` | Модель для `--llm-host` |
| `--streamer` / `-s` | — | Имя профиля или путь к `.md` файлу |
| `--glossary` | — | Путь к файлу глоссария |
| `--translate-segments` / `-t` | false | Построчный перевод сегментов |
| `--batch-size` | 50 | Сегментов на батч при `--translate-segments` |
| `--reuse-transcript` / `-r` | false | Загрузить сегменты из `.segments.json` |
| `--subtitles` | false | Генерировать `.ass`-субтитры |
| `--debug-llm` | false | Печатать сырые ответы LLM |
