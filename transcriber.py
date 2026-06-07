#!/usr/bin/env python3
"""Local multilingual transcriber: audio → Russian summary + timestamped transcript."""

import gc
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic import BaseModel
from rich.console import Console
import time
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

app = typer.Typer(help="Transcribe audio (EN/JA/ID) → Russian summary + timestamped Markdown.")
console = Console()


class TranslationOnly(BaseModel):
    translation_russian: str


class BatchTranslation(BaseModel):
    translations: list[str]


class ThemeItem(BaseModel):
    name: str


class ThemeList(BaseModel):
    themes: list[ThemeItem]


class NarrativeBlock(BaseModel):
    prose: str


class SubtitleLines(BaseModel):
    lines: list[str]


BATCH_PROMPT = """\
Переведи каждую строку транскрипта на русский язык дословно.
Верни JSON с массивом "translations" той же длины, что и входные строки.
Сохраняй порядок. Не объединяй и не разбивай строки.
Имена людей, географические названия, бренды, названия продуктов питания и напитков (matcha, hojicha, yerba mate и т.п.), игры, фильмы — оставляй без перевода.
Никакого текста до или после JSON. Никаких пояснений."""

INTRO_SYSTEM = """\
/no_think
Ты — редактор подкастов и стримов. Пиши ТОЛЬКО на русском языке.
Тебе дан транскрипт. Напиши короткое вступление (2–3 предложения):
кто ведёт, о чём стрим/подкаст, общая атмосфера или формат.
Конкретика важнее общих слов. Названия оставляй на языке оригинала.
Верни JSON с полем "prose"."""

OUTRO_SYSTEM = """\
/no_think
Ты — редактор подкастов и стримов. Пиши ТОЛЬКО на русском языке.
Тебе дан транскрипт. Напиши короткое заключение (2–3 предложения):
чем закончился стрим/подкаст, что запомнилось, какое общее впечатление.
Конкретика важнее общих слов. Названия оставляй на языке оригинала.
Верни JSON с полем "prose"."""

_STREAMER_CONTEXT_PREFIX = """\
Контекст об авторе контента (используй эти данные при написании — имя, местоимения, тематика):
{context}

---

"""

THEME_EXTRACT_SYSTEM = """\
/no_think
Ты — редактор подкастов и стримов. Тебе дан транскрипт.
Найди 6–8 главных тем. Верни JSON объект с полем "themes" — массивом объектов {"name": "<тема на русском>"}.
Только JSON, никакого другого текста."""

THEME_NARRATIVE_SYSTEM = """\
/no_think
Ты — профессиональный редактор. Пиши ТОЛЬКО на русском языке — никакого английского текста.
Тебе дан транскрипт и название одной темы.
Напиши нарративную прозу об этой теме от третьего лица.
Объём: столько предложений, сколько нужно чтобы передать ВСЕ конкретные детали из транскрипта — 
минимум 4, максимум 10. Не растягивай если деталей мало, не обрезай если их много.
Приоритет конкретики:
- Конкретные названия, цифры, примеры из оригинала важнее общих наблюдений
- Если автор назвал игру, цену, дату — включи это
- Если была шутка или характерная реплика — сохрани её суть
Правила:
- Названия игр, фильмов, брендов, продуктов питания и напитков оставляй на языке оригинала.
- Не начинай текст с названия темы.
- Без маркированных списков и вводных фраз.
- Сохраняй голос и юмор оригинала.
Верни JSON с полем "prose" — строка с готовым текстом."""

TRANSLATION_SYSTEM = """\
/no_think
Ты — профессиональный переводчик.
Сделай дословный перевод ВСЕГО транскрипта на русский язык.
Это полный перевод, а не пересказ — каждое предложение оригинала должно быть переведено.
Имена людей, географические названия, бренды, названия продуктов и напитков — оставляй без перевода.
Сохраняй структуру речи.
Верни только перевод в поле translation_russian."""

SUB_TRANSLATE_SYSTEM = """\
/no_think
Переводи субтитры на русский язык. Входные данные — JSON-массив строк.
Верни JSON объект {"lines": [...]} с переводами строго в том же порядке и количестве.
Имена людей, географические названия, бренды, названия продуктов и напитков, игры, фильмы — оставляй на языке оригинала.
Переводи кратко и точно."""

# Minimum subtitle display time — BBC recommends ≥1s, Netflix ≥5/6s; 1.2s is a safe default.
MIN_SUB_DURATION = 1.2


def translate_segments_batch(
    ollama_client,
    model: str,
    texts: list[str],
    source_lang: str,
    timeout: int = 120,
) -> list[str]:
    """Translate a batch of segment texts, return same-length list of Russian strings."""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_msg = f"Язык оригинала: {source_lang}\n\n{numbered}"

    for attempt in (1, 2):
        try:
            resp = ollama_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": BATCH_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format=BatchTranslation.model_json_schema(),
                options={"timeout": timeout},
                think=False,
            )
            content = strip_thinking(resp.message.content)
            if not content and hasattr(resp.message, "thinking") and resp.message.thinking:
                content = resp.message.thinking.strip()
            result = BatchTranslation.model_validate_json(content)
            translations = result.translations
            if len(translations) < len(texts):
                translations += [""] * (len(texts) - len(translations))
            return translations[: len(texts)]
        except Exception as e:
            if attempt == 1:
                continue  # retry once silently
            console.print(f"[yellow]  Batch failed after 2 attempts ({e}), skipping.[/yellow]")
            return [""] * len(texts)
    return [""] * len(texts)

def llm_call(ollama_client, model: str, system: str, user: str, num_predict: int = 1024) -> str:
    """Simple Ollama chat call, returns content string."""
    resp = ollama_client.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"num_predict": num_predict, "temperature": 0.7},
    )
    return resp.message.content


def extract_themes(ollama_client, model: str, transcript_text: str) -> list[tuple[str, str]]:
    """Pass 1: ask model to identify 6-8 themes. Returns list of (name, description)."""
    raw = llm_call(ollama_client, model, THEME_EXTRACT_SYSTEM,
                   f"Транскрипт:\n\n{transcript_text}\n\nВыдели 6–8 главных тем.",
                   num_predict=600)
    themes = []
    for line in raw.splitlines():
        line = line.strip().lstrip("*•-0123456789. ")
        if line.upper().startswith("ТЕМА:"):
            content = line.split(":", 1)[-1].strip()
            name = content.split("|")[0].strip()
            desc = content.split("|")[1].strip() if "|" in content else ""
            if name:
                themes.append((name, desc))
    if not themes:
        # Fallback: treat each non-empty line as a theme
        themes = [(l.strip(), "") for l in raw.splitlines() if l.strip()][:8]
    return themes


def write_theme_block(ollama_client, model: str, transcript_text: str,
                      theme_name: str, theme_desc: str, glossary_text: str = "") -> str:
    """Pass 2: write 4-5 sentence narrative prose for one theme."""
    user = (
        f"Транскрипт:\n\n{transcript_text}\n\n"
        f"Тема: {theme_name}\n{theme_desc}\n\n"
        f"Напиши 4–5 предложений об этой теме."
    )
    if glossary_text:
        user = f"Глоссарий:\n{glossary_text}\n\n" + user
    return llm_call(ollama_client, model, THEME_NARRATIVE_SYSTEM, user, num_predict=500).strip()


def fmt_timestamp(seconds: float) -> str:
    """Convert seconds float to HH:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def fmt_ass_time(seconds: float) -> str:
    """Convert seconds to ASS subtitle timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def group_subtitle_segments(
    segments,
    min_duration: float = MIN_SUB_DURATION,
    max_gap: float = 2.0,
    max_chars: int = 84,
) -> list[tuple[float, float, str, list[int]]]:
    """Merge short Whisper segments into cards with minimum display time.

    Returns list of (start, end, combined_text, [original_indices]).
    """
    groups: list[tuple[float, float, str, list[int]]] = []
    buf_texts: list[str] = []
    buf_idx: list[int] = []
    buf_start: float = 0.0
    buf_end: float = 0.0

    def flush() -> None:
        if not buf_texts:
            return
        groups.append((buf_start, max(buf_end, buf_start + min_duration), " ".join(buf_texts), buf_idx[:]))

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue

        gap = (seg.start - buf_end) if buf_texts else 0.0
        merged_len = len(" ".join(buf_texts + [text]))

        if buf_texts and (gap > max_gap or merged_len > max_chars):
            flush()
            buf_texts = [text]
            buf_idx = [i]
            buf_start = seg.start
            buf_end = seg.end
        else:
            if not buf_texts:
                buf_start = seg.start
            buf_texts.append(text)
            buf_idx.append(i)
            buf_end = seg.end

        # Flush when duration is sufficient and text ends a sentence
        if buf_end - buf_start >= min_duration and text[-1:] in ".?!":
            flush()
            buf_texts = []
            buf_idx = []

    flush()
    return groups


_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""


def write_ass(path: Path, groups: list[tuple[float, float, str, list[int]]]) -> None:
    lines = [_ASS_HEADER]
    for start, end, text, _idx in groups:
        safe = text.replace("\\", "\\\\").replace("{", "\\{")
        lines.append(f"Dialogue: 0,{fmt_ass_time(start)},{fmt_ass_time(end)},Default,,0,0,0,,{safe}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. qwen3)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: LLM returned invalid JSON.[/red]\nRaw response:\n{raw[:500]}")
        raise typer.Exit(1) from e


def _parse_narrative(raw: str, label: str = "") -> str:
    """Extract prose string from NarrativeBlock JSON with fallback attempts."""
    def clean(text: str) -> str:
        return strip_thinking(text).strip()

    # 1. Direct parse (happy path — json_schema gives clean JSON)
    try:
        return clean(NarrativeBlock.model_validate_json(raw).prose)
    except Exception:
        pass
    # 2. Strip markdown fences and retry
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        return clean(NarrativeBlock.model_validate_json(cleaned).prose)
    except Exception:
        pass
    # 3. Regex: extract first "prose" value from partial/truncated JSON
    m = re.search(r'"prose"\s*:\s*"(.*)', raw, re.DOTALL)
    if m:
        prose = m.group(1)
        prose = re.sub(r'"\s*\}?\s*$', "", prose)
        prose = clean(prose)
        if prose:
            console.print(f'[yellow]  Warning: narrative JSON truncated for "{label}", used partial prose.[/yellow]')
            return prose
    console.print(f'[yellow]  Warning: could not parse narrative JSON for "{label}".[/yellow]')
    console.print(f'[dim]  Raw ({len(raw)} chars): {raw[:200]}[/dim]')
    return ""


def render_markdown(
    audio_path: Path,
    info,
    whisper_model_name: str,
    ollama_model_name: str,
    segments: list,
    summary_blocks: list[tuple[str, str]] | None = None,  # (theme_name, narrative_prose)
    translation: str = "",  # kept for API compat, no longer rendered
    segment_translations: list[str] | None = None,
    stage: str = "done",
    intro: str = "",
    outro: str = "",
    sub_groups: list | None = None,
) -> str:
    """Render the output .md file content."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    stage_note = {
        "transcribed": " *(транскрипция готова, саммари в процессе)*",
        "summarizing": " *(темы определяются...)*",
        "done": "",
    }.get(stage, "")

    lines = [
        "---",
        f"source: {audio_path.name}",
        f"language: {info.language}",
        f"duration: {info.duration:.1f}",
        f"whisper_model: {whisper_model_name}",
        f"ollama_model: {ollama_model_name}",
        f"processed_at: {now}",
        f"stage: {stage}",
        "---",
        "",
        f"# Пересказ{stage_note}",
        "",
    ]

    if not summary_blocks and not intro:
        lines += ["*Саммари будет добавлено после обработки.*", ""]
    else:
        if intro:
            lines += ["## Вступление", "", intro, ""]
        if summary_blocks:
            for name, prose in summary_blocks:
                lines += [f"## {name}", "", prose, ""]
        if outro:
            lines += ["## Заключение", "", outro, ""]

    lines += ["---", "", "## Оригинальный транскрипт", ""]

    has_seg_tr = bool(segment_translations)
    if has_seg_tr:
        lines += ["| Время | Оригинал | Перевод |", "|-------|----------|---------|"]
    else:
        lines += ["| Время | Текст |", "|-------|-------|"]
    groups = sub_groups if sub_groups is not None else group_subtitle_segments(segments)
    for start, _end, text, indices in groups:
        ts = fmt_timestamp(start)
        text = text.replace("|", "\\|")
        if has_seg_tr:
            ru = " ".join(
                segment_translations[j] for j in indices if j < len(segment_translations)
            ).replace("|", "\\|")
            lines.append(f"| {ts} | {text} | {ru} |")
        else:
            lines.append(f"| {ts} | {text} |")

    return "\n".join(lines) + "\n"


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Audio/video file to transcribe (mp3/wav/m4a/ogg/mp4/mkv/...)."),
    model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model size."),
    lang: str = typer.Option(None, "--lang", "-l", help="Force source language: en, ja, id."),
    fast: bool = typer.Option(False, "--fast", help="Use whisper-medium for speed (lower accuracy)."),
    beam_size: int = typer.Option(1, "--beam-size", help="Whisper beam size (1=greedy, less VRAM; 5=accurate, more VRAM)."),
    vad: bool = typer.Option(True, "--vad/--no-vad", help="Voice activity detection: skip silence before transcribing."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output file."),
    ollama_model: str = typer.Option("qwen3:14b", "--ollama-model", help="Ollama model name (ignored when --llm-host is set)."),
    llm_host: str = typer.Option(None, "--llm-host", help="OpenAI-compatible LLM server URL, e.g. http://localhost:8080 (llama-server). Bypasses Ollama."),
    llm_model: str = typer.Option("gpt-4o-mini", "--llm-model", help="Model name for --llm-host (e.g. gpt-4o-mini, gpt-4.1)."),
    streamer: str = typer.Option(None, "--streamer", "-s", help="Streamer name (looks up streamers/<name>.md) or path to a profile .md file."),
    glossary: Path = typer.Option(None, "--glossary", help="Path to glossary file (term = translation, one per line)."),
    translate_segments: bool = typer.Option(False, "--translate-segments", "-t", help="Translate each segment individually (slower, adds column to transcript table)."),
    batch_size: int = typer.Option(50, "--batch-size", help="Segments per translation batch (used with --translate-segments)."),
    reuse_transcript: bool = typer.Option(False, "--reuse-transcript", "-r", help="Skip Whisper: load segments from cached .segments.json if it exists."),
    debug_llm: bool = typer.Option(False, "--debug-llm", help="Print raw Ollama response objects for debugging."),
    subtitles: bool = typer.Option(False, "--subtitles", help="Generate .orig.ass (original) and .ru.ass (Russian) subtitle files."),
):
    """Transcribe audio → translate to Russian → structured Markdown summary."""
    # --- Validate input ---
    if not audio_file.exists():
        console.print(f"[red]Error: file not found: {audio_file}[/red]")
        raise typer.Exit(1)

    output_path = audio_file.with_suffix(".md")
    if output_path.exists() and not force:
        console.print(f"[yellow]Output already exists: {output_path}[/yellow]")
        console.print("Use --force to overwrite.")
        raise typer.Exit(0)

    whisper_model_name = "medium" if fast else model

    # --- GPU availability check: block if Ollama has models loaded in VRAM ---
    # Skip when reusing transcript — Whisper won't run, so no VRAM conflict.
    using_cache = reuse_transcript and (audio_file.with_suffix(".segments.json").exists())
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/ps", timeout=3) if not using_cache else None
        if resp is not None and resp.status_code == 200:
            running = resp.json().get("models", [])
            if running:
                lines = []
                for m in running:
                    vram_gb = m.get("size_vram", 0) / 1024**3
                    lines.append(f"  • {m['name']}  ({vram_gb:.1f} GB VRAM)")
                console.print("[yellow]Ollama has models loaded in GPU VRAM:[/yellow]")
                for l in lines:
                    console.print(l)
                console.print("[yellow]Running Whisper in parallel may cause OOM. Waiting for Ollama to unload...[/yellow]")
                names = " / ".join(m["name"] for m in running)
                console.print(f"  To unload manually:  [bold]ollama stop {names}[/bold]")
                console.print("  (Press Ctrl-C to skip waiting and proceed anyway)\n")
                try:
                    while True:
                        time.sleep(5)
                        resp2 = httpx.get("http://localhost:11434/api/ps", timeout=3)
                        still = resp2.json().get("models", []) if resp2.status_code == 200 else []
                        if not still:
                            console.print("[green]Ollama VRAM cleared. Starting.[/green]\n")
                            break
                        names = ", ".join(m["name"] for m in still)
                        console.print(f"  Still loaded: {names} — waiting...")
                except KeyboardInterrupt:
                    console.print("\n[yellow]Skipping wait — proceeding anyway.[/yellow]\n")
    except Exception:
        pass  # Ollama not reachable or httpx not installed — skip check

    # --- Load streamer profile ---
    streamer_text = ""
    if streamer:
        ctx_path = Path(streamer)
        if not ctx_path.exists():
            for base in [Path(__file__).parent, Path.cwd()]:
                candidate = base / "streamers" / f"{streamer}.md"
                if candidate.exists():
                    ctx_path = candidate
                    break
        if ctx_path.exists():
            streamer_text = ctx_path.read_text(encoding="utf-8").strip()
            console.print(f"  Streamer profile: [cyan]{ctx_path.name}[/cyan]")
        else:
            console.print(f"[yellow]Warning: streamer profile not found for '{streamer}' — continuing without context.[/yellow]")

    def _inject_context(system: str) -> str:
        if not streamer_text:
            return system
        return _STREAMER_CONTEXT_PREFIX.format(context=streamer_text) + system

    # --- Load glossary ---
    glossary_text = ""
    if glossary:
        if not glossary.exists():
            console.print(f"[red]Error: glossary file not found: {glossary}[/red]")
            raise typer.Exit(1)
        lines = [
            l.strip() for l in glossary.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if lines:
            glossary_text = "\n".join(lines)

    # --- Step 1: Transcribe (or load from cache) ---
    cache_path = audio_file.with_suffix(".segments.json")

    if using_cache:
        console.print(f"[bold]Loading cached transcript:[/bold] {cache_path.name}")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

        class _FakeInfo:
            language = cached["language"]
            duration = cached["duration"]

        class _FakeSeg:
            def __init__(self, d):
                self.start = d["start"]
                self.end = d["end"]
                self.text = d["text"]

        info = _FakeInfo()
        segments = [_FakeSeg(s) for s in cached["segments"]]
        whisper_model_name = cached.get("whisper_model", whisper_model_name)
        console.print(f"  Loaded [cyan]{len(segments)}[/cyan] segments  |  lang={info.language}  |  duration={info.duration:.1f}s")
    else:
        console.print(f"[bold]Transcribing:[/bold] {audio_file.name}")
        console.print(f"  Model: whisper-{whisper_model_name}  |  Language: {lang or 'auto-detect'}")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            console.print("[red]Error: faster-whisper not installed. Run: pip install faster-whisper[/red]")
            raise typer.Exit(1)

    if not using_cache:
        # --- Load model ---
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Loading Whisper model...", total=None)
            try:
                import ctranslate2
                n_gpu = ctranslate2.get_cuda_device_count()
            except Exception:
                n_gpu = 0
            if n_gpu > 0:
                device, compute_type = "cuda", "float16"
                progress.update(task, description=f"Loading Whisper model (GPU ×{n_gpu})...")
            else:
                device, compute_type = "cpu", "int8"
                progress.update(task, description="Loading Whisper model (CPU int8)...")
            whisper = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)

        # --- Transcribe with auto language detection per ~30s chunk ---
        # language=None lets Whisper detect per chunk, which handles code-switching
        # (e.g. en/ja/id/ko mixed within a single stream).
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Detecting language...", total=None)
            raw_segments, info = whisper.transcribe(
                str(audio_file), language=lang, beam_size=beam_size, vad_filter=vad,
                condition_on_previous_text=(lang is not None),
            )
            detected = info.language

        console.print(f"  Detected language: [cyan]{detected}[/cyan]  |  Duration: [cyan]{info.duration:.1f}s[/cyan]  |  Device: [cyan]{device}[/cyan]")

        # --- Transcribe with live progress ---
        segments = []
        t0 = time.monotonic()
        cancelled = False
        with Progress(
            SpinnerColumn(), BarColumn(bar_width=30), TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[pos]}[/cyan]"),
            TextColumn("[yellow]{task.fields[speed]}[/yellow]"),
            TimeElapsedColumn(), console=console,
        ) as progress:
            task = progress.add_task("Transcribing...", total=info.duration, pos="0:00 / 0:00", speed="")
            try:
                for seg in raw_segments:
                    segments.append(seg)
                    elapsed = time.monotonic() - t0
                    audio_pos = seg.end
                    speed = audio_pos / elapsed if elapsed > 0 else 0
                    pos_str = f"{fmt_timestamp(audio_pos)} / {fmt_timestamp(info.duration)}"
                    progress.update(task, completed=audio_pos, pos=pos_str, speed=f"{speed:.1f}× realtime")
            except KeyboardInterrupt:
                cancelled = True

        if cancelled:
            console.print(f"\n[yellow]Cancelled at {fmt_timestamp(segments[-1].end if segments else 0)}. Exiting.[/yellow]")
            raise typer.Exit(0)

        elapsed_total = time.monotonic() - t0
        speed_avg = info.duration / elapsed_total if elapsed_total > 0 else 0
        console.print(f"  Segments: [cyan]{len(segments)}[/cyan]  |  Speed: [cyan]{speed_avg:.1f}× realtime[/cyan]  |  Time: [cyan]{elapsed_total:.1f}s[/cyan]")

        # --- Save transcript cache ---
        cache_path.write_text(
            json.dumps({"language": info.language, "duration": info.duration,
                        "whisper_model": whisper_model_name,
                        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"  [dim]Transcript cached: {cache_path.name}[/dim]")

        # --- Unload Whisper VRAM before loading LLM ---
        if hasattr(whisper, "model") and hasattr(whisper.model, "unload_model"):
            whisper.model.unload_model()
        del whisper
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- Pre-compute subtitle groups once for all saves ---
    sub_groups = group_subtitle_segments(segments)

    # --- Stage 1 save: transcript only (no summary yet) ---
    output_path.write_text(
        render_markdown(audio_path=audio_file, info=info, whisper_model_name=whisper_model_name,
                        ollama_model_name=ollama_model, segments=segments,
                        summary_blocks=None, translation="", stage="transcribed",
                        sub_groups=sub_groups),
        encoding="utf-8",
    )
    console.print(f"  [dim]Saved (stage 1/3): {output_path.name}[/dim]")

    # --- Build transcript text for LLM ---
    # Estimate token count conservatively: Latin ~4 chars/token, CJK/Cyrillic ~2 chars/token.
    # Using 2.5 chars/token as a safe middle ground to avoid underestimating num_ctx.
    transcript_text = " ".join(seg.text.strip() for seg in segments)
    char_count = len(transcript_text)
    est_tokens = int(char_count / 2.5)
    # qwen3:30b context: 128k tokens.
    MAX_TRANSCRIPT_TOKENS = 120_000
    if est_tokens > MAX_TRANSCRIPT_TOKENS:
        # Keep first 60% and last 40% to preserve both intro and conclusion
        keep_chars = int(MAX_TRANSCRIPT_TOKENS * 2.5)
        cut_front = int(keep_chars * 0.6)
        cut_back = keep_chars - cut_front
        transcript_text = (
            transcript_text[:cut_front]
            + f"\n\n[... транскрипт обрезан: {char_count} → {keep_chars} символов ...]\n\n"
            + transcript_text[-cut_back:]
        )
        console.print(f"  [yellow]Transcript truncated: ~{est_tokens:,} tokens → {MAX_TRANSCRIPT_TOKENS:,} (model limit)[/yellow]")
    else:
        console.print(f"  Transcript: [cyan]{len(segments)} segments[/cyan], ~[cyan]{est_tokens:,} tokens[/cyan]")

    # --- Step 2: Two-pass summarization + translation ---
    try:
        import ollama as ollama_client
    except ImportError:
        console.print("[red]Error: ollama not installed. Run: pip install ollama[/red]")
        raise typer.Exit(1)

    # num_ctx: transcript tokens + 16k headroom for system prompts and output.
    # 16k headroom covers system prompts (~200 tokens) and generous output space.
    _num_ctx = max(16384, min(est_tokens + 16384, 131072))
    _backend = f"{llm_model} @ {llm_host}" if llm_host else f"Ollama ({ollama_model})"
    console.print(f"  LLM backend: [cyan]{_backend}[/cyan]  |  context: [cyan]{_num_ctx:,} tokens[/cyan]")

    def _llm_call(system: str, user: str, num_predict: int | None = None, json_schema: dict | None = None) -> str:
        """Dispatch to llama-server (OpenAI API) or Ollama depending on --llm-host."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            if llm_host:
                # llama-server: OpenAI-compatible REST API via httpx
                import httpx
                payload: dict = {
                    "model": llm_model,
                    "messages": messages,
                    "temperature": 0.7,
                }
                if num_predict is not None:
                    payload["max_tokens"] = num_predict
                if json_schema:
                    payload["response_format"] = {"type": "json_object"}
                import os
                headers = {}
                if api_key := os.environ.get("OPENAI_API_KEY"):
                    headers["Authorization"] = f"Bearer {api_key}"
                resp = httpx.post(
                    f"{llm_host.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=None,
                )
                resp.raise_for_status()
                return strip_thinking(resp.json()["choices"][0]["message"]["content"])
            else:
                # Ollama
                kwargs: dict = dict(
                    model=ollama_model,
                    messages=messages,
                    options={"num_ctx": _num_ctx, "num_predict": num_predict if num_predict is not None else -1, "temperature": 0.7},
                    think=False,  # disable thinking mode (qwen3 and similar)
                )
                if json_schema:
                    kwargs["format"] = json_schema
                resp = ollama_client.chat(**kwargs)
                if debug_llm:
                    console.print(f"[dim]── DEBUG ollama raw ──[/dim]")
                    console.print(f"[dim]message.content:  {repr(resp.message.content[:300])}[/dim]")
                    thinking = getattr(resp.message, "thinking", None)
                    console.print(f"[dim]message.thinking: {repr((thinking or '')[:300])}[/dim]")
                content = strip_thinking(resp.message.content)
                # fallback: if content empty, model put answer in thinking field
                if not content and hasattr(resp.message, "thinking") and resp.message.thinking:
                    content = resp.message.thinking.strip()
                return content
        except KeyboardInterrupt:
            console.print("\n[yellow]LLM call cancelled.[/yellow]")
            raise typer.Exit(0)
        except ConnectionRefusedError:
            target = llm_host or "localhost:11434 (Ollama)"
            console.print(f"[red]Error: LLM server not running at {target}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            raise typer.Exit(1)

    # --- Pass 1: extract themes (structured JSON) ---
    console.print(f"\n[bold]Analysing:[/bold] {ollama_model} — extracting themes")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        t = p.add_task("Pass 1: identifying themes...", total=None)
        themes_raw = _llm_call(
            THEME_EXTRACT_SYSTEM,
            f"Транскрипт:\n\n{transcript_text}\n\nНазвания тем — только на русском языке.",
            json_schema=ThemeList.model_json_schema(),
        )
        p.update(t, description="Themes extracted")

    themes: list[str] = []
    try:
        theme_data = ThemeList.model_validate_json(themes_raw)
        themes = [t.name for t in theme_data.themes if t.name]
    except Exception as e:
        console.print(f"[yellow]Warning: could not parse themes JSON ({e}). Raw:[/yellow]\n{themes_raw[:1000]}")

    if not themes:
        console.print("[red]Could not extract themes. Exiting.[/red]")
        raise typer.Exit(1)

    console.print(f"  Found [cyan]{len(themes)}[/cyan] themes: {', '.join(themes)}")

    # --- Intro ---
    console.print(f"[bold]Writing:[/bold] intro")
    intro = ""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        p.add_task("Intro...", total=None)
        raw_intro = _llm_call(
            _inject_context(INTRO_SYSTEM),
            f"Транскрипт:\n\n{transcript_text}",
            json_schema=NarrativeBlock.model_json_schema(),
        )
    intro = _parse_narrative(raw_intro, "Вступление")

    # --- Pass 2: write narrative block per theme (save after each) ---
    console.print(f"[bold]Writing:[/bold] narrative blocks")
    summary_blocks: list[tuple[str, str]] = []
    with Progress(
        SpinnerColumn(), BarColumn(bar_width=25), TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[theme]}[/cyan]"), TimeElapsedColumn(), console=console,
    ) as p:
        t = p.add_task("Pass 2...", total=len(themes), theme="")
        for name in themes:
            p.update(t, theme=name[:40])
            user = (
                f"{'Глоссарий:\n' + glossary_text + chr(10)*2 if glossary_text else ''}"
                f"Транскрипт:\n\n{transcript_text}\n\n"
                f"Тема: {name}\n\n"
                f"Напиши 4–5 предложений об этой теме. ОБЯЗАТЕЛЬНО на русском языке."
            )
            raw_prose = _llm_call(_inject_context(THEME_NARRATIVE_SYSTEM), user,
                                   json_schema=NarrativeBlock.model_json_schema())
            prose = _parse_narrative(raw_prose, name)
            summary_blocks.append((name, prose))
            p.advance(t)
            # Save progressively after each block
            output_path.write_text(
                render_markdown(audio_path=audio_file, info=info, whisper_model_name=whisper_model_name,
                                ollama_model_name=ollama_model, segments=segments,
                                summary_blocks=summary_blocks, stage="summarizing", intro=intro,
                                sub_groups=sub_groups),
                encoding="utf-8",
            )

    # --- Outro ---
    console.print(f"[bold]Writing:[/bold] outro")
    outro = ""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        p.add_task("Outro...", total=None)
        raw_outro = _llm_call(
            _inject_context(OUTRO_SYSTEM),
            f"Транскрипт:\n\n{transcript_text}",
            json_schema=NarrativeBlock.model_json_schema(),
        )
    outro = _parse_narrative(raw_outro, "Заключение")

    # --- Stage 2 save: full summary ---
    output_path.write_text(
        render_markdown(audio_path=audio_file, info=info, whisper_model_name=whisper_model_name,
                        ollama_model_name=ollama_model, segments=segments,
                        summary_blocks=summary_blocks, intro=intro, outro=outro,
                        stage="done", sub_groups=sub_groups),
        encoding="utf-8",
    )
    console.print(f"  [dim]Saved (stage 2/3): {output_path.name}[/dim]")

    # --- Optional: per-segment translation ---
    segment_translations: list[str] | None = None
    if translate_segments:
        seg_texts = [seg.text.strip() for seg in segments]
        batches = [seg_texts[i:i + batch_size] for i in range(0, len(seg_texts), batch_size)]
        total_batches = len(batches)
        console.print(f"\n[bold]Translating segments:[/bold] {len(seg_texts)} segments → {total_batches} batches of {batch_size}")
        segment_translations = []
        with Progress(
            SpinnerColumn(),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[info]}[/cyan]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Translating...", total=total_batches, info="")
            for idx, batch in enumerate(batches):
                progress.update(task, info=f"batch {idx+1}/{total_batches} ({len(batch)} segs)")
                try:
                    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
                    user_msg = f"Язык оригинала: {info.language}\n\n{numbered}"
                    translated: list[str] = []
                    for attempt in (1, 2):
                        try:
                            raw = _llm_call(BATCH_PROMPT, user_msg,
                                            json_schema=BatchTranslation.model_json_schema())
                            result = BatchTranslation.model_validate_json(raw)
                            translated = result.translations
                            if len(translated) < len(batch):
                                translated += [""] * (len(batch) - len(translated))
                            translated = translated[:len(batch)]
                            break
                        except Exception as e:
                            if attempt == 1:
                                continue
                            console.print(f"[yellow]  Batch {idx+1} failed ({e}), skipping.[/yellow]")
                            translated = [""] * len(batch)
                    segment_translations.extend(translated)
                    progress.advance(task)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Segment translation cancelled — saving partial results.[/yellow]")
                    segment_translations.extend([""] * len(batch))
                    segment_translations.extend([""] * sum(len(b) for b in batches[idx + 1:]))
                    break

        # --- Stage 3 save: full output with segment translations ---
        output_path.write_text(
            render_markdown(audio_path=audio_file, info=info, whisper_model_name=whisper_model_name,
                            ollama_model_name=ollama_model, segments=segments,
                            summary_blocks=summary_blocks, intro=intro, outro=outro,
                            segment_translations=segment_translations, stage="done",
                            sub_groups=sub_groups),
            encoding="utf-8",
        )

    if subtitles:
        console.print(f"\n[bold]Subtitles:[/bold] grouping segments (min {MIN_SUB_DURATION}s per card)...")
        sub_groups = group_subtitle_segments(segments)
        console.print(f"  {len(segments)} segments → {len(sub_groups)} subtitle cards")

        orig_ass = audio_file.with_suffix(".orig.ass")
        write_ass(orig_ass, sub_groups)
        console.print(f"  [dim]Saved: {orig_ass.name}[/dim]")

        ru_ass = audio_file.with_suffix(".ru.ass")
        orig_texts = [text for _, _, text, _ in sub_groups]
        ru_texts: list[str] = []
        sub_batch_size = 20
        sub_batches = [orig_texts[i:i + sub_batch_size] for i in range(0, len(orig_texts), sub_batch_size)]
        console.print(f"  Translating {len(orig_texts)} cards → {len(sub_batches)} batches...")
        with Progress(
            SpinnerColumn(),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[info]}[/cyan]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Translating subtitles...", total=len(sub_batches), info="")
            for idx, batch in enumerate(sub_batches):
                progress.update(task, info=f"batch {idx+1}/{len(sub_batches)}")
                raw = _llm_call(
                    SUB_TRANSLATE_SYSTEM,
                    json.dumps(batch, ensure_ascii=False),
                    json_schema=SubtitleLines.model_json_schema(),
                )
                try:
                    parsed = SubtitleLines.model_validate_json(raw).lines
                    if len(parsed) >= len(batch):
                        ru_texts.extend(parsed[:len(batch)])
                    else:
                        ru_texts.extend(parsed)
                        ru_texts.extend([""] * (len(batch) - len(parsed)))
                except Exception:
                    ru_texts.extend([""] * len(batch))
                progress.advance(task)

        ru_groups = [(s, e, ru_texts[i], idx) for i, (s, e, _, idx) in enumerate(sub_groups) if i < len(ru_texts)]
        write_ass(ru_ass, ru_groups)
        console.print(f"  [dim]Saved: {ru_ass.name}[/dim]")

    console.print(f"\n[green]Done:[/green] {output_path}")
    console.print(f"  Segments: {len(segments)}  |  Output: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    app()
