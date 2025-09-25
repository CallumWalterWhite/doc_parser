from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from rich import print as rprint
from rich.table import Table
from pydantic_settings import BaseSettings

# LangChain & friends
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable
from langchain.chains import RetrievalQA

# PDF & OCR
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

#env
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    rprint("[red]Error: OPENAI_API_KEY environment variable not set. Please set it in your environment or .env file.[/red]")
    raise typer.Exit(code=1)

# Tables (optional)
try:
    import camelot  # type: ignore
    HAS_CAMELOT = True
except Exception:
    HAS_CAMELOT = False

# Add poppler path if needed (Windows)
if "POPPLER_PATH" in os.environ:
    poppler_path = os.environ["POPPLER_PATH"]
    if not Path(poppler_path).exists():
        rprint(f"[red]Warning: POPPLER_PATH is set to '{poppler_path}' but that path does not exist.[/red]")
    else:
        os.environ["PATH"] += os.pathsep + poppler_path
else:
    os.environ["PATH"] = "C:/tools/poppler-25.07.0/Library/bin"

if "TESSDATA_PREFIX" in os.environ:
    tessdata_prefix = os.environ["TESSDATA_PREFIX"]
    if not Path(tessdata_prefix).exists():
        rprint(f"[red]Warning: TESSDATA_PREFIX is set to '{tessdata_prefix}' but that path does not exist.[/red]")
    else:
        pytesseract.pytesseract.tesseract_cmd = str(Path(tessdata_prefix) / "tesseract.exe")
else:
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

app = typer.Typer(add_completion=False)


class Settings(BaseSettings):
    # Embedding model
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    openai_model: str = "o4-mini"

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 150

    # OCR threshold – if extracted text is shorter than this, we try OCR
    ocr_trigger_chars: int = 800

    class Config:
        env_prefix = "PARSER_"

def get_embeddings(settings: Settings):
    return HuggingFaceEmbeddings(model_name=settings.embed_model)


def get_llm(settings: Settings):
    """Return an LLM or None. Try OpenAI first; fallback to Ollama."""
    # Try OpenAI (chat)
    try:
        if OPENAI_API_KEY:
            return ChatOpenAI(model=settings.openai_model)
    except Exception as e:
        rprint(f"[yellow]OpenAI init failed: {e}[/yellow]")

    # No LLM available
    return None


# -----------------------------
# PDF Parsing & OCR
# -----------------------------

def extract_text_with_pymupdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_textpage().extractText())
    return "\n".join(parts).strip()


def ocr_pdf_to_text(pdf_path: Path, dpi: int = 300) -> str:
    images = convert_from_path(str(pdf_path), dpi=dpi)
    text_parts: List[str] = []
    for img in images:
        text_parts.append(pytesseract.image_to_string(img))
    return "\n".join(text_parts).strip()


def extract_tables_with_camelot(pdf_path: Path, flavor: str = "lattice") -> List[str]:
    if not HAS_CAMELOT:
        return []
    try:
        tables = camelot.read_pdf(str(pdf_path), flavor=flavor, pages="all")
        csv_snippets = []
        for i, t in enumerate(tables):
            csv_snippets.append(t.df.to_csv(index=False))
        return csv_snippets
    except Exception as e:
        rprint(f"[yellow]Camelot failed: {e}. Skipping tables for {pdf_path.name}[/yellow]")
        return []


# -----------------------------
# Pre‑processing & Chunking
# -----------------------------

def clean_text(text: str) -> str:
    # Light cleanup; keep it conservative to not harm semantics
    text = re.sub(r"\u00a0", " ", text)  # non‑breaking space
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_documents(
    raw_text: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": source, **(metadata or {}), "chunk": i},
        )
        for i, chunk in enumerate(chunks)
    ]
    return docs


# -----------------------------
# Index Build & Persist
# -----------------------------

def build_or_update_faiss(
    docs: List[Document], index_dir: Path, settings: Settings
) -> FAISS:
    embeddings = get_embeddings(settings)
    if index_dir.exists():
        vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
    else:
        vs = FAISS.from_documents(docs, embeddings)
        index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return vs


# -----------------------------
# High‑level ingest routine
# -----------------------------

def ingest_path(
    input_path: Path,
    index_dir: Path,
    settings: Settings,
    ocr: bool = False,
    include_tables: bool = True,
) -> None:
    all_docs: List[Document] = []

    pdf_files = []
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_files = [input_path]
    else:
        pdf_files = list(input_path.rglob("*.pdf"))

    if not pdf_files:
        rprint(f"[red]No PDFs found under {input_path}[/red]")
        raise typer.Exit(code=1)

    for pdf in pdf_files:
        rprint(f"[bold cyan]Parsing:[/bold cyan] {pdf}")
        text = extract_text_with_pymupdf(pdf)
        if (ocr or len(text) < settings.ocr_trigger_chars):
            rprint("[yellow]Running OCR fallback...[/yellow]")
            text = text + "\n\n" + ocr_pdf_to_text(pdf)

        text = clean_text(text)
        meta = {"filename": pdf.name, "path": str(pdf)}
        docs = make_documents(
            text,
            source=str(pdf),
            metadata=meta,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        all_docs.extend(docs)

        if include_tables:
            csvs = extract_tables_with_camelot(pdf)
            for ti, csv in enumerate(csvs):
                tdocs = make_documents(
                    csv,
                    source=f"{pdf}#table{ti}",
                    metadata={**meta, "table_index": ti, "modality": "table/csv"},
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                all_docs.extend(tdocs)

    rprint(f"[green]Total chunks:[/green] {len(all_docs)}")
    # build_or_update_faiss(all_docs, index_dir, settings)
    rprint(f"[green]Index saved to[/green] {index_dir}")


# -----------------------------
# Query / QA
# -----------------------------

def query_index(index_dir: Path, question: str, settings: Settings, k: int = 4) -> Dict[str, Any]:
    embeddings = get_embeddings(settings)
    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    llm = get_llm(settings)
    docs = retriever.get_relevant_documents(question)

    if llm is None:
        return {
            "answer": "(No LLM available. Showing top context chunks.)",
            "contexts": [d.page_content for d in docs],
            "metadata": [d.metadata for d in docs],
        }

    try:
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        result = chain.invoke({"query": question})
        return {
            "answer": result.get("result"),
            "contexts": [d.page_content for d in docs],
            "metadata": [d.metadata for d in docs],
        }
    except Exception as e:
        rprint(f"[yellow]QA failed ({e}). Falling back to retrieval-only.[/yellow]")
        return {
            "answer": "(LLM failed. Showing top context chunks.)",
            "contexts": [d.page_content for d in docs],
            "metadata": [d.metadata for d in docs],
        }



# -----------------------------
# CLI Commands
# -----------------------------

@app.command()
def ingest(
    input: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=True, help="PDF file or directory"),
    index: Path = typer.Option(..., help="Directory to store FAISS index"),
    ocr: bool = typer.Option(False, help="Force OCR for all PDFs"),
    no_tables: bool = typer.Option(False, help="Disable table extraction with Camelot"),
    reset: bool = typer.Option(False, help="Delete and rebuild the index from scratch")
):
    """Parse PDFs and (re)build the FAISS index."""
    if reset and index.exists():
        for p in index.glob("**/*"):
            if p.is_file():
                p.unlink()
        for p in sorted(index.glob("**/*"), reverse=True):
            if p.is_dir():
                p.rmdir()
    settings = Settings()
    ingest_path(input, index, settings, ocr=ocr, include_tables=(not no_tables))


@app.command()
def query(
    q: str = typer.Option(..., help="Your question"),
    index: Path = typer.Option(..., exists=True, help="Directory for FAISS index"),
    k: int = typer.Option(4, help="Top‑k chunks to retrieve"),
):
    """Ask questions against the indexed documents (RAG)."""
    settings = Settings()
    out = query_index(index, q, settings, k=k)
    rprint("\n[bold]Answer:[/bold]", out.get("answer"))
    tbl = Table(title="Top Context Chunks")
    tbl.add_column("#", width=4)
    tbl.add_column("Source", overflow="fold")
    tbl.add_column("Snippet", overflow="fold")
    for i, (m, c) in enumerate(zip(out["metadata"], out["contexts"])):
        src = f"{m.get('filename')} (chunk {m.get('chunk')})"
        snippet = (c[:500] + "…") if len(c) > 500 else c
        tbl.add_row(str(i + 1), src, snippet)
    rprint(tbl)


@app.command("peek-index")
def peek_index(
    index: Path = typer.Option(..., exists=True),
    k: int = typer.Option(3),
    query_text: str = typer.Option("index sanity check", help="Probe query to preview retrieved chunks"),
):
    """Quickly preview what's in the FAISS index by running a probe retrieval."""
    settings = Settings()
    res = query_index(index, query_text, settings, k=k)
    for i, (m, c) in enumerate(zip(res["metadata"], res["contexts"])):
        rprint(f"[bold]{i+1}. {m.get('source')}[/bold] -> chunk {m.get('chunk')}")
        rprint(c[:300] + ("…" if len(c) > 300 else ""))
        rprint("")


@app.command()
def rebuild(
    input: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=True),
    index: Path = typer.Option(...),
    ocr: bool = typer.Option(False),
    no_tables: bool = typer.Option(False),
):
    """Delete and rebuild the index from scratch."""
    if index.exists():
        for p in index.glob("**/*"):
            if p.is_file():
                p.unlink()
        for p in sorted(index.glob("**/*"), reverse=True):
            if p.is_dir():
                p.rmdir()
    ingest.callback(input=input, index=index, ocr=ocr, no_tables=no_tables)  # type: ignore


if __name__ == "__main__":
    app()
