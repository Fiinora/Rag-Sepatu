#!/usr/bin/env python3
"""
Codebase RAG / Code Semantic Search CLI
Mendukung Multi-Project / Multi-Repository
Menggunakan ChromaDB lokal dan HuggingFace Embedding (all-MiniLM-L6-v2)
"""

import os
import re
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Fix encoding issue on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("Error: Library 'chromadb' belum terinstall.")
    print("Silakan jalankan: pip install chromadb sentence-transformers")
    sys.exit(1)

def load_env_db_dir() -> Path:
    """Membaca CHROMA_DB_DIR dari .env di folder script, atau fallback ke default."""
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "CHROMA_DB_DIR":
                            clean_val = v.strip().strip("'\"")
                            if clean_val:
                                return Path(clean_val).expanduser().resolve()
        except Exception:
            pass

    env_val = os.getenv("CHROMA_DB_DIR")
    if env_val:
        return Path(env_val).expanduser().resolve()

    return Path.home() / ".code_rag_db"

# Lokasi penyimpanan ChromaDB
DEFAULT_DB_DIR = load_env_db_dir()
LAST_PROJECT_FILE = "last_project.txt"
COLLECTION_PREFIX = "codebase_"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ekstensi file yang didukung
SUPPORTED_EXTENSIONS = {
    ".kt", ".go", ".tsx", ".ts", ".jsx", ".js", ".py"
}

# Direktori yang harus diabaikan
IGNORED_DIRS = {
    "node_modules", "build", ".git", "venv", ".venv", "env",
    "vendor", ".idea", ".gradle", "bin", "obj", "__pycache__",
    ".pytest_cache", "dist", "out", "target", ".next"
}


def sanitize_project_name(name: str) -> str:
    """
    Sanitasi nama project agar sesuai dengan aturan penamaan ChromaDB:
    - 3-63 karakter
    - alphanumeric, underscore, hyphen
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    if not cleaned:
        cleaned = "default"
    return cleaned


def get_collection_name(project_name: str) -> str:
    """Mengembalikan nama collection ChromaDB dengan prefix standar."""
    sanitized = sanitize_project_name(project_name)
    coll_name = f"{COLLECTION_PREFIX}{sanitized}"
    # ChromaDB batas 63 karakter
    return coll_name[:63]


def get_hash_file_path(db_dir: Path, project_name: str) -> Path:
    """Mengembalikan path file metadata hash spesifik per project."""
    sanitized = sanitize_project_name(project_name)
    return db_dir / f"hashes_{sanitized}.json"


def get_last_project(db_dir: Path) -> Optional[str]:
    """Membaca nama project terakhir yang di-index."""
    last_file = db_dir / LAST_PROJECT_FILE
    if last_file.exists():
        try:
            return last_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None


def set_last_project(db_dir: Path, project_name: str):
    """Menyimpan nama project terakhir yang di-index."""
    try:
        last_file = db_dir / LAST_PROJECT_FILE
        last_file.write_text(project_name, encoding="utf-8")
    except Exception:
        pass


def get_embedding_function():
    """Menggunakan SentenceTransformerEmbeddingFunction lokal bawaan ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )


def get_chroma_collection(persist_dir: Path, collection_name: str):
    """Inisialisasi klien ChromaDB persistent dan ambil collection spesifik."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    embed_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection, client


def chunk_file_content(content: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split kode menjadi potongan 500-1000 karakter dengan overlap 100,
    sambil melacak nomor baris awal (start_line) dan akhir (end_line).
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    chunks = []
    current_chunk_lines = []
    current_chunk_len = 0
    start_line_idx = 1

    for line in lines:
        line_len = len(line)
        if current_chunk_len + line_len > chunk_size and current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            end_line_idx = start_line_idx + len(current_chunk_lines) - 1
            chunks.append({
                "text": chunk_text,
                "start_line": start_line_idx,
                "end_line": end_line_idx
            })

            overlap_lines = []
            overlap_len = 0
            for prev_line in reversed(current_chunk_lines):
                overlap_lines.insert(0, prev_line)
                overlap_len += len(prev_line)
                if overlap_len >= chunk_overlap:
                    break

            start_line_idx = end_line_idx - len(overlap_lines) + 1
            current_chunk_lines = list(overlap_lines)
            current_chunk_len = overlap_len

        current_chunk_lines.append(line)
        current_chunk_len += line_len

    if current_chunk_lines:
        chunk_text = "".join(current_chunk_lines)
        end_line_idx = start_line_idx + len(current_chunk_lines) - 1
        chunks.append({
            "text": chunk_text,
            "start_line": start_line_idx,
            "end_line": end_line_idx
        })

    return chunks


def calculate_file_hash(file_path: Path) -> str:
    """Menghitung hash MD5 dari isi file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_file_hashes(hash_file_path: Path) -> Dict[str, str]:
    """Memuat catatan hash file dari JSON."""
    if hash_file_path.exists():
        try:
            with open(hash_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Gagal membaca {hash_file_path}: {e}")
    return {}


def save_file_hashes(hash_file_path: Path, hashes: Dict[str, str]):
    """Menyimpan catatan hash file ke JSON."""
    try:
        with open(hash_file_path, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)
    except Exception as e:
        print(f"⚠️ Gagal menyimpan file hashes: {e}")


def list_projects(db_dir: Path = DEFAULT_DB_DIR):
    """Menampilkan daftar semua project yang sudah pernah di-index beserta total chunks-nya."""
    if not db_dir.exists():
        print(f"\n📂 Database belum dibuat di: {db_dir}")
        return

    client = chromadb.PersistentClient(path=str(db_dir))
    collections = client.list_collections()
    last_proj = get_last_project(db_dir)

    print(f"\n📂 Direktori Database: {db_dir}")
    print("=" * 60)
    print(f"{'PROJECT NAME':<25} | {'COLLECTION':<22} | {'CHUNKS':<8}")
    print("-" * 60)

    count_projects = 0
    for coll in collections:
        c_name = coll.name
        # Ekstrak project name dari prefix
        if c_name.startswith(COLLECTION_PREFIX):
            p_name = c_name[len(COLLECTION_PREFIX):]
        else:
            p_name = c_name

        marker = " (aktif)" if p_name == last_proj else ""
        item_count = coll.count()
        print(f"{p_name + marker:<25} | {c_name:<22} | {item_count:<8}")
        count_projects += 1

    if count_projects == 0:
        print("Belum ada project yang di-index.")
    print("=" * 60)


def resolve_target_project(
    user_project: Optional[str],
    db_dir: Path,
    client: Optional[chromadb.ClientAPI] = None
) -> Tuple[str, str]:
    """
    Menentukan project mana yang akan di-query dengan urutan fallback:
    1. Argumen eksplisit user (--project)
    2. Deteksi CWD saat ini (nama folder saat ini) jika collection cocok
    3. Project terakhir yang di-index (last_project.txt)
    """
    if client is None:
        client = chromadb.PersistentClient(path=str(db_dir))

    existing_collections = {c.name for c in client.list_collections()}

    # 1. User spesifik
    if user_project:
        p_name = sanitize_project_name(user_project)
        c_name = get_collection_name(p_name)
        return p_name, c_name

    # 2. Cek apakah CWD saat ini cocok dengan collection yang ada
    cwd_name = sanitize_project_name(Path.cwd().name)
    cwd_c_name = get_collection_name(cwd_name)
    if cwd_c_name in existing_collections:
        return cwd_name, cwd_c_name

    # 3. Cek project terakhir yang di-index
    last_proj = get_last_project(db_dir)
    if last_proj:
        last_c_name = get_collection_name(last_proj)
        if last_c_name in existing_collections:
            return last_proj, last_c_name

    # 4. Fallback ke CWD atau collection pertama yang ada
    for coll_name in existing_collections:
        if coll_name.startswith(COLLECTION_PREFIX):
            extracted = coll_name[len(COLLECTION_PREFIX):]
            return extracted, coll_name

    return cwd_name, cwd_c_name


def index_repo(repo_path: str, project_name: Optional[str] = None, db_dir: Path = DEFAULT_DB_DIR, reset: bool = False):
    """
    Scan repo secara incremental untuk project tertentu:
    - Nama collection spesifik per project
    - File hash tersimpan di hashes_<project>.json
    """
    root_path = Path(repo_path).resolve()
    if not root_path.exists() or not root_path.is_dir():
        print(f"[ERROR] Path repository tidak valid: {repo_path}")
        return

    # Tentukan nama project
    final_project_name = sanitize_project_name(project_name if project_name else root_path.name)
    collection_name = get_collection_name(final_project_name)
    hash_file_path = get_hash_file_path(db_dir, final_project_name)

    print(f"\n🚀 Memulai indexing project: [{final_project_name}]")
    print(f"📂 Direktori database: {db_dir}")
    print(f"📦 Collection name   : {collection_name}")

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    embed_fn = get_embedding_function()

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"🗑️  Koleksi [{collection_name}] berhasil di-reset.")
        except Exception:
            pass
        if hash_file_path.exists():
            try:
                hash_file_path.unlink()
                print("🗑️  Metadata hash file berhasil di-reset.")
            except Exception:
                pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    stored_hashes = {} if reset else load_file_hashes(hash_file_path)

    print(f"🔍 Memindai file kode di: {root_path} ...")
    current_files_map = {}

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        for file in files:
            ext = Path(file).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                f_path = Path(root) / file
                rel_path = f_path.relative_to(root_path).as_posix()
                current_files_map[rel_path] = f_path

    # Deteksi status file
    files_to_index = []
    files_skipped = 0
    new_hashes = dict(stored_hashes)

    # 1. Cek file yang ditambah atau diubah
    for rel_path, file_path in current_files_map.items():
        try:
            current_hash = calculate_file_hash(file_path)
        except Exception as e:
            print(f"⚠️ Gagal membaca {file_path}: {e}")
            continue

        prev_hash = stored_hashes.get(rel_path)
        if prev_hash == current_hash and not reset:
            files_skipped += 1
        else:
            files_to_index.append((rel_path, file_path, current_hash, prev_hash is not None))

    # 2. Cek file yang sudah dihapus dari codebase
    deleted_files = [rel_path for rel_path in stored_hashes.keys() if rel_path not in current_files_map]

    for rel_path in deleted_files:
        try:
            collection.delete(where={"file_path": rel_path})
        except Exception:
            pass
        new_hashes.pop(rel_path, None)

    print(f"✨ Total file ditemukan: {len(current_files_map)} (Perlu diproses: {len(files_to_index)}, Di-skip: {files_skipped}, Dihapus: {len(deleted_files)})")

    all_ids = []
    all_documents = []
    all_metadatas = []
    total_chunks_added = 0
    batch_size = 100

    for rel_path, file_path, file_hash, is_modified in files_to_index:
        if is_modified and not reset:
            try:
                collection.delete(where={"file_path": rel_path})
            except Exception:
                pass

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Gagal membaca {file_path}: {e}")
            continue

        if not content.strip():
            new_hashes[rel_path] = file_hash
            continue

        chunks = chunk_file_content(content, chunk_size=800, chunk_overlap=100)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}#chunk{idx}"
            all_ids.append(chunk_id)
            all_documents.append(chunk["text"])
            all_metadatas.append({
                "file_path": rel_path,
                "absolute_path": str(file_path),
                "extension": file_path.suffix.lower(),
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"]
            })
            total_chunks_added += 1

            if len(all_ids) >= batch_size:
                collection.upsert(
                    ids=all_ids,
                    documents=all_documents,
                    metadatas=all_metadatas
                )
                all_ids = []
                all_documents = []
                all_metadatas = []
                print(f"  -> {total_chunks_added} chunks diproses...", end="\r", flush=True)

        new_hashes[rel_path] = file_hash

    if all_ids:
        collection.upsert(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas
        )

    # Simpan state hash dan project aktif
    save_file_hashes(hash_file_path, new_hashes)
    set_last_project(db_dir, final_project_name)

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 RINGKASAN INDEXING [{final_project_name}]:")
    print(f"  • File Di-skip (Tidak berubah) : {files_skipped}")
    print(f"  • File Di-update / Baru         : {len(files_to_index)}")
    print(f"  • File Dihapus                 : {len(deleted_files)}")
    print(f"  • Chunk baru/update diproses   : {total_chunks_added}")
    print(f"  • Total item di Collection     : {collection.count()}")
    print("=" * 50)
    print(f"✅ Indexing project '{final_project_name}' selesai!")


def query_code(
    query_text: str,
    project_name: Optional[str] = None,
    top_k: int = 5,
    db_dir: Path = DEFAULT_DB_DIR
):
    """
    Mencari snippet kode yang paling relevan pada project tertentu.
    """
    resolved_proj, collection_name = resolve_target_project(project_name, db_dir)
    collection, _ = get_chroma_collection(db_dir, collection_name)
    count = collection.count()

    if count == 0:
        print(f"\n⚠️ Collection '{collection_name}' (Project: '{resolved_proj}') masih kosong.")
        print(f"Silakan jalankan: python code_rag.py index <repo_path> --project {resolved_proj}")
        return

    print(f"\n🔎 [{resolved_proj}] Mencari: '{query_text}' (Top {top_k} hasil dari {count} chunks)...")
    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, count)
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else [None] * len(documents)

    if not documents:
        print("❌ Tidak ditemukan snippet yang relevan.")
        return

    separator = "=" * 80
    print(f"\n{separator}")
    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        file_path = meta.get("file_path", "unknown")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        score_info = f" | Cosine Distance: {dist:.4f}" if dist is not None else ""

        print(f"[{idx}] 📄 File : {file_path}")
        print(f"    📍 Baris: {start_line} - {end_line}{score_info}")
        print("    " + "-" * 76)

        formatted_code = "\n".join(f"    {line}" for line in doc.strip().splitlines())
        print(formatted_code)
        print(f"{separator}")


def interactive_mode(project_name: Optional[str] = None, db_dir: Path = DEFAULT_DB_DIR):
    """Mode interaktif di terminal untuk tanya-jawab terus-menerus."""
    resolved_proj, collection_name = resolve_target_project(project_name, db_dir)
    print(f"\n💬 Masuk ke Mode Interaktif Code Semantic Search")
    print(f"Project Target : {resolved_proj} (Collection: {collection_name})")
    print(f"Database       : {db_dir}")
    print("Ketik pertanyaan Anda (atau 'exit' / 'q' untuk keluar).\n")

    while True:
        try:
            query = input(f"❓ [{resolved_proj}] Pertanyaan > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("👋 Sampai jumpa!")
                break
            query_code(query, project_name=resolved_proj, top_k=3, db_dir=db_dir)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Sampai jumpa!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="CLI Code Semantic Search / Codebase RAG Multi-Project lokal menggunakan ChromaDB & HuggingFace Embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")

    # Command: index
    parser_index = subparsers.add_parser("index", help="Index codebase dari direktori target")
    parser_index.add_argument("repo_path", help="Path direktori repository yang ingin diindeks")
    parser_index.add_argument("-p", "--project", default=None, help="Nama kustom untuk project (default: nama folder repo)")
    parser_index.add_argument("--db", default=str(DEFAULT_DB_DIR), help="Path direktori penyimpanan ChromaDB")
    parser_index.add_argument("--reset", action="store_true", help="Hapus indeks project ini sebelum indexing ulang")

    # Command: query
    parser_query = subparsers.add_parser("query", help="Cari snippet kode berdasarkan pertanyaan")
    parser_query.add_argument("question", help="Pertanyaan atau deskripsi pencarian")
    parser_query.add_argument("-p", "--project", default=None, help="Nama project target (default: auto-detect dari folder/last indexed)")
    parser_query.add_argument("-k", "--top_k", type=int, default=3, help="Jumlah snippet yang ditampilkan (default: 3)")
    parser_query.add_argument("--db", default=str(DEFAULT_DB_DIR), help="Path direktori penyimpanan ChromaDB")

    # Command: chat / interactive
    parser_chat = subparsers.add_parser("chat", help="Buka sesi terminal interaktif untuk pencarian berulang")
    parser_chat.add_argument("-p", "--project", default=None, help="Nama project target")
    parser_chat.add_argument("--db", default=str(DEFAULT_DB_DIR), help="Path direktori penyimpanan ChromaDB")

    # Command: list
    parser_list = subparsers.add_parser("list", help="Lihat daftar project yang sudah pernah di-index")
    parser_list.add_argument("--db", default=str(DEFAULT_DB_DIR), help="Path direktori penyimpanan ChromaDB")

    args = parser.parse_args()

    if args.command == "index":
        index_repo(args.repo_path, project_name=args.project, db_dir=Path(args.db), reset=args.reset)
    elif args.command == "query":
        query_code(args.question, project_name=args.project, top_k=args.top_k, db_dir=Path(args.db))
    elif args.command == "chat":
        interactive_mode(project_name=args.project, db_dir=Path(args.db))
    elif args.command == "list":
        list_projects(db_dir=Path(args.db))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
