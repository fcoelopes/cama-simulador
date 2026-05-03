# Simulador CAMA — RAG sobre Base de Conhecimento de Gestão de Ativos

App Streamlit que indexa sua base de conhecimento (normas ISO 55000/55001/55002, slides, anotações) e gera questões de múltipla escolha estilo prova **CAMA (Certified Asset Management Assessor)** via RAG.

## Stack

- **Embeddings**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (local, multilíngue)
- **Vector store**: ChromaDB persistente (`db/`)
- **LLM**: plugável — Gemini, OpenAI ou Anthropic (escolhido na sidebar)
- **Cache de questões + histórico de tentativas**: SQLite (`cache/questions.sqlite`)
- **Export**: Markdown e PDF (reportlab)
- **UI**: Streamlit

## Estrutura

```
cama_simulator/
├── app.py            # Streamlit (3 abas)
├── config.py         # paths e constantes
├── ingest.py         # leitura, chunking, embeddings
├── llm.py            # abstração dos providers
├── quiz.py           # RAG + geração + cache + tracking
├── export.py         # Markdown e PDF
├── dashboard.py      # queries de métricas
├── requirements.txt
├── data/             # COLOQUE SEUS PDFs/DOCX/MD AQUI
├── db/               # ChromaDB (gerado)
└── cache/            # SQLite com questões + histórico (gerado)
```

## Setup

```bash
cd cama_simulator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Na primeira execução o `sentence-transformers` baixa o modelo (~470 MB).

## Uso

1. Coloque arquivos (PDF/DOCX/MD/TXT) em `data/` — pode usar subpastas.
2. `streamlit run app.py`
3. Sidebar: escolha provider, modelo e cole sua API key (não é persistida).
4. Clique em **🔄 Indexar** — só processa o que ainda não foi indexado.
5. Aba **✨ Gerar novas questões**: digite o tópico (ex: "ISO 55001 cláusula 6 — Planejamento") e gere.
6. Aba **🎯 Simulado**: filtra por tópico, escolhe o tamanho e responde. Pode exportar:
   - Antes de finalizar: prova em branco (Markdown ou PDF) pra estudar offline
   - Depois de finalizar: gabarito completo com justificativas
7. Aba **📊 Dashboard**: KPIs gerais, evolução por sessão, desempenho por tópico (ordenado pelo pior), questões mais erradas.

## Pontos de arquitetura

- **Chunking** por caracteres (800 com overlap 150), com corte preferencial em fim de parágrafo/frase.
- **ID estável** dos chunks: `caminho_relativo + mtime`, então editar um arquivo invalida e reprocessa só ele.
- **RAG**: top-6 chunks por tópico → contexto do prompt → JSON estruturado.
- **Validação** das questões: descarta qualquer item com ≠ 4 alternativas ou `answer_idx` fora de [0,3].
- **Cache híbrido**: questões geradas ficam no SQLite e podem ser reusadas; gera novas a qualquer hora.
- **Tracking**: tabela `attempts` registra cada resposta com `session_id`, timestamp, tópico e acerto. Permite agregações por sessão (evolução temporal), por tópico (pontos fracos) e por questão (mais erradas).
- **Import lazy do ChromaDB**: dashboard e export não precisam carregar embeddings, então essas abas abrem instantaneamente.
- **PDF via reportlab**: puro Python, sem deps de sistema (não precisa wkhtmltopdf nem chromium).

## Customização rápida

- Trocar modelo de embedding: `EMBED_MODEL` em `config.py`.
- Mexer no chunking: `CHUNK_SIZE` / `CHUNK_OVERLAP`.
- Mais/menos contexto por questão: `TOP_K`.
- Adicionar provider: nova classe em `llm.py` + entrada em `PROVIDERS`.
- Cutoff de aprovação: hard-coded em 70% no `app.py` (mude o `if pct >= 70` se a CAMA mudar).

## CLI alternativo

Pra indexar fora do Streamlit:

```bash
python ingest.py
```