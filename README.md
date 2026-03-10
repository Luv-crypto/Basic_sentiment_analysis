# Basic_sentiment_analysis
Sentiment analysis on 150 different articles
from __future__ import annotations
import os, glob, uuid, time, re
from pathlib import Path
from threading import Event
from typing import List, Dict, Tuple, Any

import chromadb
import nest_asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# -------------------- setup --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_GEN = "models/gemini-1.5-flash-latest"
MODEL_EMB = "models/text-embedding-004"

if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

nest_asyncio.apply()
genai.configure(api_key=GEMINI_API_KEY)
_gem = genai.GenerativeModel(MODEL_GEN)

client = chromadb.PersistentClient(path="chroma_generic_rag")
txt_col = client.get_or_create_collection("text_chunks", metadata={"hnsw:space": "cosine"})
img_col = client.get_or_create_collection("image_chunks", metadata={"hnsw:space": "cosine"})
tbl_col = client.get_or_create_collection("table_chunks", metadata={"hnsw:space": "cosine"})

IMG_DIR = Path("object_store/images")
TBL_DIR = Path("object_store/tables")
IMG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- docling --------------------
pipe_opts = PdfPipelineOptions(
    do_table_structure=True,
    generate_page_images=True,
    generate_picture_images=True,
    save_picture_images=True,
    images_scale=2.0
)
pipe_opts.table_structure_options.mode = TableFormerMode.ACCURATE

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)}
)

# -------------------- helpers --------------------
def _chat(prompt: str, retry: int = 3) -> str:
    for i in range(retry):
        try:
            return _gem.generate_content(prompt).text.strip()
        except Exception:
            if i == retry - 1:
                raise
            time.sleep(1 + i)
    return ""

def _embed(texts: List[str]) -> List[List[float]]:
    return genai.embed_content(
        model=MODEL_EMB,
        content=texts,
        task_type="retrieval_document"
    )["embedding"]

def _summarize_image(path: str, caption: str = "") -> str:
    with open(path, "rb") as f:
        parts = [
            {"mime_type": "image/png", "data": f.read()},
            f"Summarize this image for retrieval in under 120 words. Caption: {caption or 'N/A'}"
        ]
    return _gem.generate_content(parts).text.strip()

def _summarize_table(table_md: str, caption: str = "") -> str:
    prompt = f"""
Summarize this table for retrieval in under 120 words.
Mention what the table contains and any key values/comparisons if obvious.

Caption: {caption or 'N/A'}

Table:
{table_md[:4000]}
"""
    return _chat(prompt)

def _zip_meta(res) -> List[Dict]:
    if not res or not res.get("ids"):
        return []
    ids_raw = res["ids"][0] if isinstance(res["ids"][0], list) else res["ids"]
    metas_raw = res["metadatas"][0] if isinstance(res["metadatas"][0], list) else res["metadatas"]
    out = []
    for _id, meta in zip(ids_raw, metas_raw):
        if meta:
            d = dict(meta)
            d["id"] = _id
            out.append(d)
    return out

def _cosine_top(question_vec: List[float], items: Dict[str, Dict], top_n: int) -> List[str]:
    if not items:
        return []
    ids = list(items.keys())
    summaries = [items[i]["summary"] for i in ids]
    vecs = _embed(summaries)
    sims = [dot(question_vec, v) / (norm(question_vec) * norm(v) + 1e-9) for v in vecs]
    ranked = sorted(zip(ids, sims), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:top_n]]

def _fetch_linked_media(chunk_ids: List[str], user_id: int) -> Tuple[List[Dict], List[Dict]]:
    if not chunk_ids:
        return [], []
    where_clause = {"$and": [{"user_id": user_id}, {"parent_chunk_id": {"$in": chunk_ids}}]}
    imgs = _zip_meta(img_col.get(where=where_clause, include=["metadatas"]))
    tbls = _zip_meta(tbl_col.get(where=where_clause, include=["metadatas"]))
    return imgs, tbls

# -------------------- ingestion --------------------
def ingest_documents(pattern: str, user_id: int, chunk_size: int = 1500, stop_event: Event | None = None) -> None:
    stop_event = stop_event or Event()
    pdfs = glob.glob(pattern, recursive=True)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs matched: {pattern}")

    for pdf in pdfs:
        if stop_event.is_set():
            print("Ingestion cancelled.")
            return

        p = Path(pdf)
        print(f"Processing {p.name} ...")

        ddoc = converter.convert(p).document
        md = ddoc.export_to_markdown()
        file_meta = {"file_name": p.name, "file_path": str(p), "user_id": user_id}

        # text chunks
        chunks = [md[i:i + chunk_size] for i in range(0, len(md), chunk_size)]
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        for cid, chunk in zip(chunk_ids, chunks):
            txt_col.add(
                ids=[cid],
                embeddings=[_embed([chunk])[0]],
                documents=[chunk],
                metadatas=[{
                    **file_meta,
                    "chunk_id": cid,
                    "chunk_preview": chunk[:300]
                }]
            )

        # images
        page_numbers = [pic.prov[0].page_no for pic in ddoc.pictures if pic.prov]
        max_pg = max(page_numbers) if page_numbers else 1

        for pic in ddoc.pictures:
            img = pic.get_image(ddoc)
            if img is None:
                continue

            pg = pic.prov[0].page_no if pic.prov else 1
            idx = min(int((pg - 1) / max_pg * len(chunk_ids)), len(chunk_ids) - 1)
            parent_chunk_id = chunk_ids[idx]

            img_id = str(uuid.uuid4())
            img_path = IMG_DIR / f"{img_id}_{p.stem}_p{pg}.png"
            img.save(img_path, "PNG")

            caption = (pic.caption_text(ddoc) or "").strip()
            summary = _summarize_image(str(img_path), caption)
            emb_text = f"{caption}\n\n{summary}" if caption else summary

            img_col.add(
                ids=[img_id],
                embeddings=[_embed([emb_text])[0]],
                documents=[summary],
                metadatas=[{
                    **file_meta,
                    "parent_chunk_id": parent_chunk_id,
                    "path": str(img_path),
                    "caption": caption,
                    "summary": summary
                }]
            )

        # tables
        page_numbers_tbl = [t.prov[0].page_no for t in ddoc.tables if t.prov]
        max_pg_tbl = max(page_numbers_tbl) if page_numbers_tbl else 1

        for tbl in ddoc.tables:
            table_md = tbl.export_to_markdown(ddoc).strip()
            pg = tbl.prov[0].page_no if tbl.prov else 1
            idx = min(int((pg - 1) / max_pg_tbl * len(chunk_ids)), len(chunk_ids) - 1)
            parent_chunk_id = chunk_ids[idx]

            tbl_id = str(uuid.uuid4())
            tbl_path = TBL_DIR / f"{tbl_id}.md"
            tbl_path.write_text(table_md, encoding="utf-8")

            caption = (tbl.caption_text(ddoc) or "").strip()
            summary = _summarize_table(table_md, caption)
            emb_text = f"{caption}\n\n{summary}" if caption else summary

            tbl_col.add(
                ids=[tbl_id],
                embeddings=[_embed([emb_text])[0]],
                documents=[summary],
                metadatas=[{
                    **file_meta,
                    "parent_chunk_id": parent_chunk_id,
                    "path": str(tbl_path),
                    "caption": caption,
                    "summary": summary
                }]
            )

# -------------------- query --------------------
def smart_query(
    question: str,
    user_id: int,
    top_k: int = 3,
    return_media: bool = False
) -> str | Tuple[str, List[Tuple[str, str]]]:

    q_vec = _embed([question])[0]

    txt_hits = txt_col.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )

    if not txt_hits["ids"] or not txt_hits["ids"][0]:
        msg = "Sorry, I could not find relevant content for your question."
        return (msg, []) if return_media else msg

    docs = txt_hits["documents"][0]
    metas = txt_hits["metadatas"][0]
    chunk_ids = [m["chunk_id"] for m in metas]

    # linked media
    imgs_link, tbls_link = _fetch_linked_media(chunk_ids, user_id)

    # semantic media
    imgs_sem = _zip_meta(img_col.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        where={"user_id": user_id},
        include=["metadatas"]
    ))
    tbls_sem = _zip_meta(tbl_col.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        where={"user_id": user_id},
        include=["metadatas"]
    ))

    imgs_all = {m["id"]: m for m in imgs_link + imgs_sem}
    tbls_all = {t["id"]: t for t in tbls_link + tbls_sem}

    top_img_ids = _cosine_top(q_vec, imgs_all, 1)
    top_tbl_ids = _cosine_top(q_vec, tbls_all, 2)

    imgs_final = {i: imgs_all[i] for i in top_img_ids if i in imgs_all}
    tbls_final = {i: tbls_all[i] for i in top_tbl_ids if i in tbls_all}

    ctx = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        ctx.append(
            f"\n### Doc {i}"
            f"\nFile: {meta.get('file_name', '')}"
            f"\nChunk ID: {meta.get('chunk_id', '')[:8]}"
            f"\n---\n{doc[:1500]}\n"
        )

    if imgs_final:
        ctx.append("\n## Images")
        for im in imgs_final.values():
            ctx.append(f"* (img:{im['id']}) File: {im['file_name']} | {im['summary']}")

    if tbls_final:
        ctx.append("\n## Tables")
        for tb in tbls_final.values():
            ctx.append(f"* (tbl:{tb['id']}) File: {tb['file_name']} | {tb['summary']}")

    prompt = f"""
You are answering using only the provided material.

Rules:
- Use only the context below.
- If the answer is not present, say: "Sorry, the documents do not contain that information."
- Cite sources as (Doc 1), (Doc 2), etc.
- If an image or table is directly useful, output exactly:
  <<img:FULL_UUID>>
  or
  <<tbl:FULL_UUID>>
  on its own line.

--- MATERIAL ---
{''.join(ctx)}
--- END MATERIAL ---

Question: {question}
"""

    answer = _chat(prompt)

    # collect media referenced by model
    show = []
    refs = re.findall(r"<<(img|tbl):([0-9A-Fa-f\-]{32,36})>>", answer)

    for kind, media_id in refs:
        item = imgs_final.get(media_id) if kind == "img" else tbls_final.get(media_id)
        if item and Path(item["path"]).exists():
            pair = (kind, item["path"])
            if pair not in show:
                show.append(pair)

    return (answer, show) if return_media else answer
