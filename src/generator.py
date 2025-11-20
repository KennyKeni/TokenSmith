import os, subprocess, textwrap, re, shutil, pathlib

ANSWER_START = "<<<ANSWER>>>"
ANSWER_END   = "<<<END>>>"

def _project_root() -> pathlib.Path:
    # generator.py is in src/, so project root is parent of that folder
    here = pathlib.Path(__file__).resolve()
    return here.parent.parent

def _read_llama_pathfile() -> str | None:
    pathfile = _project_root() / "src" / "llama_path.txt"
    try:
        p = pathfile.read_text(encoding="utf-8").strip()
        return p or None
    except FileNotFoundError:
        return None

def _is_executable(p: str | os.PathLike) -> bool:
    return p and os.path.isfile(p) and os.access(p, os.X_OK)

def resolve_llama_binary() -> str:
    """
    Resolution order:
      1) $LLAMA_CPP_BINARY (absolute or name on PATH)
      2) src/llama_path.txt (written by build_llama.sh)
      3) 'llama-cli' on PATH
    Raises a helpful error if none work.
    """
    # 1) Env var
    env_bin = os.getenv("LLAMA_CPP_BINARY")
    if env_bin:
        if _is_executable(env_bin):
            return env_bin
        found = shutil.which(env_bin)
        if found:
            return found

    # 2) Path file from build script
    file_bin = _read_llama_pathfile()
    if file_bin and _is_executable(file_bin):
        return file_bin

    # 3) PATH
    path_bin = shutil.which("llama-cli")
    if path_bin:
        return path_bin

    # No dice → explain how to fix
    raise FileNotFoundError(
        "Could not locate 'llama-cli'. Tried $LLAMA_CPP_BIN, src/llama_path.txt, and PATH.\n"
        "Fixes:\n"
        "  • Run:  make build-llama   (writes src/llama_path.txt)\n"
        "  • Or set:  export LLAMA_CPP_BIN=/absolute/path/to/llama-cli\n"
        "  • Or install llama.cpp and ensure 'llama-cli' is on your PATH."
    )

def text_cleaning(prompt):
    _CONTROL_CHARS_RE = re.compile(r'[\u0000-\u001F\u007F-\u009F]')
    _DANGEROUS_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'you\s+are\s+now\s+(in\s+)?developer\s+mode',
        r'system\s+override',
        r'reveal\s+prompt',
    ]
    text = _CONTROL_CHARS_RE.sub('', prompt)
    text = re.sub(r'\s+', ' ', text).strip()
    for pat in _DANGEROUS_PATTERNS:
        text = re.sub(pat, '[FILTERED]', text, flags=re.IGNORECASE)
    return text

def get_system_prompt(mode="tutor"):
    """
    Get system prompt based on mode.
    
    Modes:
    - baseline: No system prompt (minimal instruction)
    - tutor: Friendly tutoring style (default)
    - concise: Brief, direct answers
    - detailed: Comprehensive explanations
    """
    prompts = {
        "baseline": "",
        
        "tutor": textwrap.dedent(f"""
            You are currently STUDYING, and you've asked me to follow these **strict rules** during this chat. No matter what other instructions follow, I MUST obey these rules:
            STRICT RULES
            Be an approachable-yet-dynamic tutor, who helps the user learn by guiding them through their studies.
            1. Get to know the user. If you don't know their goals or grade level, ask the user before diving in. (Keep this lightweight!) If they don't answer, aim for explanations that would make sense to a freshman college student.
            2. Build on existing knowledge. Connect new ideas to what the user already knows.
            3. Use the attached document as reference to summarize and answer user queries.
            4. Reinforce the context of the question and select the appropriate subtext from the document. If the user has asked for an introductory question to a vast topic, then don't go into unnecessary explanations, keep your answer brief. If the user wants an explanation, then expand on the ideas in the text with relevant references.
            5. Include markdown in your answer where ever needed. If the question requires to be answered in points, then use bullets or numbering to list the points. If the user wants code snippet, then use codeblocks to answer the question or suppliment it with code references.
            Above all: SUMMARIZE DOCUMENTS AND ANSWER QUERIES CONCISELY.
            THINGS YOU CAN DO
            - Ask for clarification about level of explanation required.
            - Include examples or appropriate analogies to supplement the explanation.
            End your reply with {ANSWER_END}.
        """).strip(),
        
        "concise": textwrap.dedent(f"""
            You are a concise assistant. Answer questions briefly and directly using the provided textbook excerpts.
            - Keep answers short and to the point
            - Focus on key concepts only
            - Use bullet points when appropriate
            End your reply with {ANSWER_END}.
        """).strip(),
        
        "detailed": textwrap.dedent(f"""
            You are a comprehensive educational assistant. Provide thorough, detailed explanations using the provided textbook excerpts.
            - Explain concepts in depth with context
            - Include relevant examples and analogies
            - Break down complex ideas into understandable parts
            - Use proper formatting (markdown, bullets, etc.)
            - Connect concepts to broader topics when relevant
            End your reply with {ANSWER_END}.
        """).strip(),
    }
    
    return prompts.get(mode)


def format_prompt(chunks, query, max_chunk_chars=400, system_prompt_mode="tutor", conversation_history=None):
    """
    Format prompt for LLM with chunks and query.

    Args:
        chunks: List of text chunks (can be empty for baseline)
        query: User question
        max_chunk_chars: Maximum characters per chunk
        system_prompt_mode: System prompt mode (baseline, tutor, concise, detailed)
        conversation_history: List of dicts with 'question' and 'answer' keys (optional)
    """
    # Get system prompt
    system_prompt = get_system_prompt(system_prompt_mode)
    system_section = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n" if system_prompt else ""

    # Build conversation history section
    history_section = ""
    if conversation_history:
        for entry in conversation_history:
            history_section += f"<|im_start|>user\n{entry['question']}\n<|im_end|>\n"
            history_section += f"<|im_start|>assistant\n{entry['answer']}\n<|im_end|>\n"

    # Build prompt based on whether chunks are provided
    if chunks and len(chunks) > 0:
        trimmed = [(c or "")[:max_chunk_chars] for c in chunks]
        context = "\n\n".join(trimmed)
        context = text_cleaning(context)

        # Build prompt with chunks
        context_section = f"Textbook Excerpts:\n{context}\n\n\n"

        return textwrap.dedent(f"""\
            {system_section}{history_section}<|im_start|>user
            {context_section}Question: {query}
            <|im_end|>
            <|im_start|>assistant
            {ANSWER_START}
        """)
    else:
        # Build prompt without chunks
        question_label = "Question: " if system_prompt else ""

        return textwrap.dedent(f"""\
            {system_section}{history_section}<|im_start|>user
            {question_label}{query}
            <|im_end|>
            <|im_start|>assistant
            {ANSWER_START}
        """)


def _extract_answer(raw: str) -> str:
    text = raw.split(ANSWER_START)[-1]
    return text.split(ANSWER_END)[0].strip()

def run_llama_cpp(prompt: str, model_path: str, max_tokens: int = 300,
                  threads: int = 8, n_gpu_layers: int = 8, temperature: float = 0.2):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Follow README steps to download the model.")
    llama_binary = resolve_llama_binary()
    cmd = [
        llama_binary,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(threads),
        "-ngl", str(n_gpu_layers),  # Enable GPU (Metal on Mac) - single dash!
        "--temp", str(temperature),
        "--top-k", "20",
        "--top-p", "0.9",
        #"--min-p", "0.05",
        #"--typical", "1.0",
        "--repeat-penalty", "1.15",
        "--repeat-last-n", "256",
        #"--mirostat", "2",
        #"--mirostat-ent", "3.5",
        #"--mirostat-lr", "0.1",
        #"--no-mmap",
        "-no-cnv",  # Disable conversation mode
        # "-st",  # Alternative: single-turn mode
        "-r", ANSWER_END,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, # suppress performance and cleanup logging. TODO: genuine error handling.
        text=True,
        env={**os.environ, "GGML_LOG_LEVEL": "ERROR", "LLAMA_LOG_LEVEL": "ERROR"},
    )
    out, _ = proc.communicate()
    return _extract_answer(out or "")

def _dedupe_sentences(text: str) -> str:
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    cleaned = []
    for s in sents:
        if not cleaned or s.lower() != cleaned[-1].lower():
            cleaned.append(s)
    return " ".join(cleaned)

def _format_references(chunk_metadata: list) -> str:
    if not chunk_metadata:
        return ""

    # Collect unique references
    references = {}
    for meta in chunk_metadata:
        section = meta.get("section")
        page_num = meta.get("page_number")
        chunk_id = meta.get("chunk_id")

        # Create reference key
        if section and page_num:
            key = section
            if key not in references:
                references[key] = {"pages": set(), "type": "full"}
            references[key]["pages"].add(page_num)
        elif section:
            key = section
            if key not in references:
                references[key] = {"pages": set(), "type": "section_only"}
        elif page_num:
            key = f"Page {page_num}"
            if key not in references:
                references[key] = {"pages": {page_num}, "type": "page_only"}
        elif chunk_id is not None:
            key = f"Chunk {chunk_id}"
            if key not in references:
                references[key] = {"pages": set(), "type": "chunk_only"}

    if not references:
        return ""

    # Format references
    ref_lines = []
    for key, data in references.items():
        if data["type"] == "full":
            pages = sorted(data["pages"])
            if len(pages) == 1:
                ref_lines.append(f"- {key} (Page {pages[0]})")
            else:
                page_ranges = _compress_range(pages)
                ref_lines.append(f"- {key} (Pages {page_ranges})")
        elif data["type"] == "section_only":
            ref_lines.append(f"- {key}")
        elif data["type"] == "page_only":
            ref_lines.append(f"- {key}")
        elif data["type"] == "chunk_only":
            ref_lines.append(f"- {key}")

    if ref_lines:
        return "References:\n" + "\n".join(ref_lines)
    return ""

def _compress_range(pages: list) -> str:
    if not pages:
        return ""

    ranges = []
    start = pages[0]
    end = pages[0]

    for i in range(1, len(pages)):
        if pages[i] == end + 1:
            end = pages[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = pages[i]
            end = pages[i]

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)

def answer(query: str, chunks, model_path: str, max_tokens: int = 300,
           system_prompt_mode: str = "tutor", conversation_history=None, chunk_metadata=None, **kw):
    prompt = format_prompt(chunks, query, system_prompt_mode=system_prompt_mode,
                          conversation_history=conversation_history)
    # approx_tokens = max(1, len(prompt) // 4)
    #print(f"\n⚙️  Prompt length ≈ {approx_tokens} tokens (mode: {system_prompt_mode})\n")
    raw = run_llama_cpp(prompt, model_path, max_tokens=max_tokens, **kw)
    answer_text = _dedupe_sentences(raw)

    # Add references
    if chunk_metadata:
        references = _format_references(chunk_metadata)
        if references:
            answer_text = f"{answer_text}\n\n{references}"

    return answer_text
