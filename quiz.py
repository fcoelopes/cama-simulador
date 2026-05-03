"""Geração de questões CAMA-style via RAG, com cache em SQLite."""
from __future__ import annotations
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass

from config import QUESTIONS_DB, TOP_K
from llm import LLMClient


# ---------- modelo ----------

@dataclass
class Question:
    id: str
    stem: str               # enunciado
    options: list[str]      # 4 alternativas
    answer_idx: int         # índice da correta (0-3)
    explanation: str        # justificativa
    sources: list[str]      # arquivos fonte
    topic: str              # tópico/área (ex: "ISO 55001 - Liderança")


# ---------- cache ----------

def _conn():
    conn = sqlite3.connect(QUESTIONS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY,
            stem TEXT NOT NULL,
            options TEXT NOT NULL,
            answer_idx INTEGER NOT NULL,
            explanation TEXT NOT NULL,
            sources TEXT NOT NULL,
            topic TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            user_answer INTEGER,
            correct_answer INTEGER NOT NULL,
            is_correct INTEGER NOT NULL,
            answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES questions(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_topic ON attempts(topic)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_qid ON attempts(question_id)")
    return conn


def record_attempts(
    session_id: str,
    questions: list[Question],
    answers: dict[str, int],
) -> None:
    """Persiste todas as respostas de uma sessão de simulado."""
    rows = []
    for q in questions:
        ua = answers.get(q.id)
        is_correct = 1 if ua == q.answer_idx else 0
        rows.append((session_id, q.id, q.topic, ua, q.answer_idx, is_correct))
    with _conn() as conn:
        conn.executemany(
            "INSERT INTO attempts (session_id, question_id, topic, user_answer, "
            "correct_answer, is_correct) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )


def save_question(q: Question) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (q.id, q.stem, json.dumps(q.options, ensure_ascii=False),
             q.answer_idx, q.explanation,
             json.dumps(q.sources, ensure_ascii=False), q.topic),
        )


def load_cached(n: int = 10, topic: str | None = None) -> list[Question]:
    with _conn() as conn:
        if topic:
            rows = conn.execute(
                "SELECT id, stem, options, answer_idx, explanation, sources, topic "
                "FROM questions WHERE topic LIKE ? ORDER BY RANDOM() LIMIT ?",
                (f"%{topic}%", n),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, stem, options, answer_idx, explanation, sources, topic "
                "FROM questions ORDER BY RANDOM() LIMIT ?",
                (n,),
            ).fetchall()
    return [
        Question(
            id=r[0], stem=r[1], options=json.loads(r[2]), answer_idx=r[3],
            explanation=r[4], sources=json.loads(r[5]), topic=r[6],
        )
        for r in rows
    ]


def count_cached() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]


def list_topics() -> list[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT DISTINCT topic FROM questions ORDER BY topic").fetchall()
    return [r[0] for r in rows]


# ---------- geração ----------

SYSTEM_PROMPT = """Você é um especialista certificado CAMA (Certified Asset Management Assessor) \
elaborando questões para simulado da prova oficial WPiAM/GFMAM, baseadas na família ISO 55000.
O material literário para conhecimento e a realização do Exame CAMA, estão contidos nos \ 
seguintes documentos: 
- Normas família ISO 55000.
- Especificação de competência GFMAM para um auditor/avaliador de sistema de gerenciamento \ 
de ativos ISO 55001 (download em site GFMAM) 
- GFMAM Landscape. 

# REGRA FUNDAMENTAL DE GROUNDING
Toda afirmação no enunciado, alternativa correta e justificativa DEVE estar DIRETAMENTE \
sustentada por trecho LITERAL do contexto fornecido. Você NÃO pode usar conhecimento externo. \
Se o contexto for ambíguo, incompleto ou tratar do tópico apenas tangencialmente, \
RETORNE LISTA VAZIA. É preferível retornar zero questões a inventar uma.

# QUALIDADE DAS QUESTÕES
- Sempre 4 alternativas (A, B, C, D), exatamente UMA correta.
- Nível CAMA: questão TESTA COMPREENSÃO de um conceito, não apenas memorização literal.
- Enunciado claro, não-ambíguo, sem pegadinhas linguísticas.

# QUALIDADE DOS DISTRATORES (esta é a parte que diferencia uma questão boa)
Os 3 distratores DEVEM:
- Ser conceitos REAIS e DISTINTOS da gestão de ativos (não invenções, não absurdos).
- Representar erros conceituais COMUNS — coisas que candidatos genuinamente confundem.
- Ter comprimento e nível de detalhe similares à correta (não delatar a resposta).
- Ser MUTUAMENTE EXCLUSIVOS — nunca duas alternativas que dizem essencialmente a mesma coisa.
- NÃO usar "Todas as anteriores" / "Nenhuma das anteriores" / "A e C estão corretas".

# JUSTIFICATIVA
A justificativa deve:
1. Citar literalmente entre aspas o trecho do contexto que sustenta a resposta correta.
2. Indicar a fonte (nome do arquivo) e, se possível, cláusula/seção.
3. Explicar por que cada distrator está errado (em uma frase curta cada).

# IDIOMA
Português brasileiro técnico, alinhado com a tradução ABNT NBR ISO 55000-series \
quando o termo tiver tradução consagrada.

# AUTOVALIDAÇÃO ANTES DE RETORNAR
Para cada questão que você gerar, verifique mentalmente:
1. A resposta correta tem suporte LITERAL no contexto? (Se não → descarte)
2. Os 4 enunciados das alternativas são realmente distintos? (Se não → descarte)
3. Um especialista olharia essa questão e diria "essa é justa"? (Se não → descarte)
Só inclua questões que passem nas três checagens."""


PROMPT_TEMPLATE_STRICT = """Gere até {n} questões de múltipla escolha sobre o tópico: "{topic}".

Use EXCLUSIVAMENTE o contexto abaixo, extraído da base de conhecimento de Gestão de Ativos. \
Cada bloco vem com [fonte: arquivo] indicando de onde foi extraído.

---CONTEXTO---
{context}
---FIM CONTEXTO---

INSTRUÇÕES DE SAÍDA (MODO RIGOROSO — qualidade > quantidade):
- Retorne APENAS um array JSON válido, sem markdown, sem comentários, sem texto antes/depois.
- Se o contexto não cobrir o tópico adequadamente, retorne array vazio [].
- É MELHOR retornar 2 questões excelentes do que {n} questões medianas.

Formato JSON esperado:
[
  {{
    "stem": "enunciado da questão",
    "options": ["alternativa A", "alternativa B", "alternativa C", "alternativa D"],
    "answer_idx": 0,
    "explanation": "Citação literal do contexto entre aspas + fonte + por que cada distrator está errado"
  }}
]

answer_idx é o índice (0 a 3) da alternativa correta."""


PROMPT_TEMPLATE_LENIENT = """Gere {n} questões de múltipla escolha sobre o tópico: "{topic}".

Use COMO BASE PRINCIPAL o contexto abaixo, extraído da base de conhecimento de Gestão de Ativos. \
Cada bloco vem com [fonte: arquivo] indicando de onde foi extraído.

---CONTEXTO---
{context}
---FIM CONTEXTO---

INSTRUÇÕES DE SAÍDA (MODO AMPLO — atingir a quantidade pedida):
- Gere {n} questões cobrindo DIFERENTES ASPECTOS do tópico (definição, aplicação prática, \
exceções, relação com outras cláusulas/conceitos, exemplos, terminologia).
- Você PODE gerar variações que abordam o mesmo conceito sob ângulos distintos (ex: "qual é X?" \
e "em qual situação X se aplica?" e "X difere de Y como?").
- A justificativa ainda DEVE se basear no contexto fornecido — não invente fatos não suportados.
- Se você só conseguir gerar menos que {n} questões boas, gere o máximo possível e retorne.
- Mantenha qualidade dos distratores e validação interna conforme regras do system prompt.
- Retorne APENAS um array JSON válido, sem markdown, sem comentários.

Formato JSON esperado:
[
  {{
    "stem": "enunciado da questão",
    "options": ["alternativa A", "alternativa B", "alternativa C", "alternativa D"],
    "answer_idx": 0,
    "explanation": "Citação literal do contexto entre aspas + fonte + por que cada distrator está errado"
  }}
]

answer_idx é o índice (0 a 3) da alternativa correta."""


# Mantém alias pra compatibilidade
PROMPT_TEMPLATE = PROMPT_TEMPLATE_STRICT


def _retrieve_context(topic: str, k: int = TOP_K) -> tuple[str, list[str]]:
    """Recupera os top-k chunks mais relevantes para o tópico."""
    from ingest import get_collection  # lazy: chromadb é pesado de carregar
    col = get_collection()
    res = col.query(query_texts=[topic], n_results=k)

    docs_raw = res["documents"]
    docs = docs_raw[0] if docs_raw else []

    metas_raw = res["metadatas"]
    metas = metas_raw[0] if metas_raw else []

    sources = sorted({str(m["source"]) for m in metas})
    context = "\n\n---\n\n".join(
        f"[fonte: {m['source']}]\n{d}" for d, m in zip(docs, metas)
    )
    return context, sources


def inspect_retrieval(topic: str, k: int = TOP_K) -> list[dict]:
    """
    Versão de debug do retrieval — retorna chunks + metadata + distância.
    Distância é cosine: 0 = idêntico, 2 = oposto. Bom retrieval geralmente < 0.7.
    """
    from ingest import get_collection
    col = get_collection()
    res = col.query(query_texts=[topic], n_results=k)

    docs_raw = res["documents"]
    docs = docs_raw[0] if docs_raw else []

    metas_raw = res["metadatas"]
    metas = metas_raw[0] if metas_raw else []

    dists_raw = res.get("distances")
    dists = dists_raw[0] if dists_raw else [None] * len(docs)

    return [
        {
            "rank": i + 1,
            "distance": d,
            "source": str(m["source"]),
            "chunk_idx": m.get("chunk_idx", "?"),
            "text": doc,
        }
        for i, (doc, m, d) in enumerate(zip(docs, metas, dists))
    ]


def build_prompt_preview(topic: str, n: int = 5, k: int = TOP_K) -> dict:
    """Retorna o que seria enviado ao LLM, sem chamar o LLM. Útil pra debug."""
    context, sources = _retrieve_context(topic, k)
    user_prompt = PROMPT_TEMPLATE.format(n=n, topic=topic, context=context)
    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "context_chars": len(context),
        "sources": sources,
    }


def _parse_json(raw: str) -> list[dict]:
    """Extrai JSON do output do LLM, tolerando cercas markdown."""
    # remove ```json ... ``` se existir
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    # tenta achar o array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def generate_questions(
    llm: LLMClient,
    topic: str,
    n: int = 5,
    persist: bool = True,
    mode: str = "strict",
) -> list[Question]:
    """
    Gera questões sobre o tópico via RAG e (opcional) salva no cache.

    mode:
      - 'strict' (default): qualidade > quantidade. LLM pode retornar menos que n
        se o contexto não suportar mais. Recomendado pra simulado real.
      - 'lenient': prioriza atingir n questões. LLM gera variações sob ângulos
        diferentes do mesmo conceito quando necessário. Útil pra revisão/treino
        de tópicos estreitos.
    """
    # Escala TOP_K conforme n: mais questões precisam de mais contexto.
    # Cap em 20 pra não estourar context window de modelos pequenos.
    k = min(max(TOP_K, n), 20)

    context, sources = _retrieve_context(topic, k=k)
    if not context:
        return []

    template = PROMPT_TEMPLATE_LENIENT if mode == "lenient" else PROMPT_TEMPLATE_STRICT
    prompt = template.format(n=n, topic=topic, context=context)
    # Modo permissivo usa temperature ligeiramente maior pra gerar variações
    temperature = 0.4 if mode == "lenient" else 0.2
    raw = llm.generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
    items = _parse_json(raw)

    questions: list[Question] = []
    for it in items:
        try:
            q = Question(
                id=str(uuid.uuid4()),
                stem=it["stem"].strip(),
                options=[o.strip() for o in it["options"]],
                answer_idx=int(it["answer_idx"]),
                explanation=it["explanation"].strip(),
                sources=sources,
                topic=topic,
            )
        except (KeyError, ValueError, TypeError):
            continue
        if len(q.options) != 4 or not (0 <= q.answer_idx <= 3):
            continue
        questions.append(q)
        if persist:
            save_question(q)
    return questions


def generate_batch(
    llm: LLMClient,
    topics: list[str],
    n_per_topic: int = 5,
    mode: str = "strict",
    progress_cb=None,
    max_retries: int = 2,
) -> dict:
    """
    Gera questões pra múltiplos tópicos em sequência. Ideal pra cobertura ampla
    de um documento denso (ex: as 39 subject areas do AM Landscape).

    Retry automático: se uma chamada retornar lista vazia ou levantar exceção,
    tenta de novo até `max_retries` vezes (default 2 = 3 tentativas no total).
    Cobre falhas pontuais como JSON truncado, rate limit momentâneo, ou flag
    de safety. Espera 2s, 4s entre tentativas (backoff linear simples).

    progress_cb(idx, total, topic, n_generated) opcional pra UI.

    Retorna {topic: [Question, ...], errors: [str, ...], total_generated: int,
    retries: int}.
    """
    import time

    result = {"by_topic": {}, "errors": [], "total_generated": 0, "retries": 0}

    for i, topic in enumerate(topics):
        if progress_cb:
            progress_cb(i + 1, len(topics), topic, 0)

        qs: list[Question] = []
        last_error: str | None = None

        for attempt in range(max_retries + 1):  # 0, 1, 2 = 3 tentativas
            try:
                qs = generate_questions(llm, topic, n=n_per_topic, mode=mode, persist=True)
                if qs:  # sucesso real (não apenas exceção evitada)
                    break
                # lista vazia = provável JSON malformado ou safety filter
                last_error = "lista vazia (provável JSON truncado ou safety filter)"
            except Exception as e:
                last_error = str(e)
                qs = []

            if attempt < max_retries:
                result["retries"] += 1
                time.sleep(2 * (attempt + 1))  # 2s, 4s
                if progress_cb:
                    progress_cb(i + 1, len(topics), f"{topic} (tentativa {attempt + 2})", 0)

        result["by_topic"][topic] = qs
        result["total_generated"] += len(qs)

        if not qs and last_error:
            result["errors"].append(f"{topic}: {last_error}")

        if progress_cb:
            progress_cb(i + 1, len(topics), topic, len(qs))

    return result