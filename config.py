"""Configurações centrais do projeto."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "db"
CACHE_DIR = ROOT / "cache"

DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Embeddings
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "cama_kb"

# Chunking
CHUNK_SIZE = 800        # caracteres por chunk
CHUNK_OVERLAP = 150     # sobreposição entre chunks

# Retrieval
TOP_K = 6               # nº de chunks recuperados por questão

# Cache de questões
QUESTIONS_DB = CACHE_DIR / "questions.sqlite"

# Formatos suportados
SUPPORTED_EXT = {".pdf", ".docx", ".md", ".txt"}

# Identificação do autor (aparece no rodapé dos PDFs exportados)
AUTHOR_NAME = "Francisco Edson Lopes da Silva"
AUTHOR_CONTACT = "www.linkedin.com/in/franciscoedsonlopessilva"
AUTHOR_EMAIL = "contato@fcoelds.dev.br"
AUTHOR_GITHUB = "https://github.com/fcoelds"
AUTHOR_DISCLAIMER = "Material de estudo pessoal · Não redistribuir"