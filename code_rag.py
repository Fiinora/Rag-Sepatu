#!/usr/bin/env python3
"""
Codebase RAG / Code Semantic Search CLI
Menggunakan ChromaDB lokal dan HuggingFace Embedding (all-MiniLM-L6-v2)
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any

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

# Lokasi penyimpanan ChromaDB
DEFAULT_DB_DIR = Path("E:/utils/.code_rag_db")
HASH_METADATA_FILE = "file_hashes.json"
COLLECTION_NAME = "codebase_rag"
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


def get_embedding_function():
    """Menggunakan SentenceTransformerEmbeddingFunction lokal bawaan ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )


def get_chroma_collection(persist_dir: Path):
    """Inisialisasi klien ChromaDB persistent."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    embed_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


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
    current_line_idx = 1

    for line in lines:
        line_len = len(line)
        if current_chunk_len + line_len > chunk_size and current_chunk_lines:
            # Selesai 1 chunk
            chunk_text = "".join(current_chunk_lines)
            end_line_idx = start_line_idx + len(current_chunk_lines) - 1
            chunks.append({
                "text": chunk_text,
                "start_line": start_line_idx,
                "end_line": end_line_idx
            })

            # Buat overlap mundur berdasarkan karakter overlap
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
        current_line_idx += 1

    # Chunk terakhir jika ada sisa
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


def index_repo(repo_path: str, db_dir: Path = DEFAULT_DB_DIR, reset: bool = False):
    """
    Scan repo secara incremental:
    - Skip file yang hash MD5-nya tidak berubah.
    - Update/embed file baru atau yang berubah (hapus chunk lama terlebih dahulu).
    - Hapus chunk dari file yang telah dihapus di codebase.
    """
    root_path = Path(repo_path).resolve()
    if not root_path.exists() or not root_path.is_dir():
        print(f"[ERROR] Path repository tidak valid: {repo_path}")
        return

    db_dir.mkdir(parents=True, exist_ok=True)
    hash_file_path = db_dir / HASH_METADATA_FILE

    print(f"\n📂 Membuka database di: {db_dir}")
    client = chromadb.PersistentClient(path=str(db_dir))
    embed_fn = get_embedding_function()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("🗑️  Koleksi lama berhasil di-reset.")
        except Exception:
            pass
        if hash_file_path.exists():
            try:
                hash_file_path.unlink()
                print("🗑️  Metadata hash file berhasil di-reset.")
            except Exception:
                pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    stored_hashes = {} if reset else load_file_hashes(hash_file_path)

    print(f"🔍 Memindai file kode di: {root_path} ...")
    current_files_map = {}  # relative_path -> file_path

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

    # Proses penghapusan file yang sudah dihapus
    for rel_path in deleted_files:
        try:
            collection.delete(where={"file_path": rel_path})
        except Exception as e:
            # Jika tidak ada chunk atau error where clause
            pass
        new_hashes.pop(rel_path, None)

    print(f"✨ Total file ditemukan: {len(current_files_map)} (Perlu diproses: {len(files_to_index)}, Di-skip: {files_skipped}, Dihapus: {len(deleted_files)})")

    all_ids = []
    all_documents = []
    all_metadatas = []
    total_chunks_added = 0
    batch_size = 100

    for rel_path, file_path, file_hash, is_modified in files_to_index:
        # Jika file dimodifikasi, hapus chunk lama sebelum memasukkan yang baru
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

    # Simpan state hash terbaru
    save_file_hashes(hash_file_path, new_hashes)

    # Tampilkan summary ringkas
    print("\n" + "=" * 50)
    print("📊 RINGKASAN INDEXING:")
    print(f"  • File Di-skip (Tidak berubah) : {files_skipped}")
    print(f"  • File Di-update / Baru         : {len(files_to_index)}")
    print(f"  • File Dihapus                 : {len(deleted_files)}")
    print(f"  • Chunk baru/update diproses   : {total_chunks_added}")
    print(f"  • Total item di ChromaDB       : {collection.count()}")
    print("=" * 50)
    print("✅ Indexing selesai!")


def query_code(query_text: str, top_k: int = 5, db_dir: Path = DEFAULT_DB_DIR):
    """
    Mencari snippet kode yang paling relevan dengan pertanyaan/query.
    """
    collection = get_chroma_collection(db_dir)
    count = collection.count()
    if count == 0:
        print("\n⚠️ Database masih kosong. Jalankan perintah 'index' terlebih dahulu.")
        return

    print(f"\n🔎 Mencari: '{query_text}' (Top {top_k} hasil dari {count} chunks)...")
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
        
        # Cetak baris kode dengan indentasi rapi
        formatted_code = "\n".join(f"    {line}" for line in doc.strip().splitlines())
        print(formatted_code)
        print(f"{separator}")


def interactive_mode(db_dir: Path = DEFAULT_DB_DIR):
    """Mode interaktif di terminal untuk tanya-jawab terus-menerus."""
    print(f"\n💬 Masuk ke Mode Interaktif Code Semantic Search")
    print(f"Database: {db_dir}")
    print("Ketik pertanyaan Anda (atau 'exit' / 'q' untuk keluar).\n")

    while True:
        try:
            query = input("❓ Pertanyaan > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("👋 Sampai jumpa!")
                break
            query_code(query, top_k=3, db_dir=db_dir)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Sampai jumpa!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="CLI Code Semantic Search / Codebase RAG lokal menggunakan ChromaDB & HuggingFace Embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")

    # Command: index
    parser_index = subparsers.add_parser("index", help="Index codebase dari direktori target")
    parser_index.add_argument("repo_path", help="Path direktori repository yang ingin diindeks")
    parser_index.add_argument("--db", default=str(DEFAULT_DB_DIR), help="Path direktori penyimpanan ChromaDB")
    parser_index.add_argument("--reset", action="store_true", help="Hapus indeks lama sebelum indexing ulang")

    # Command: query
    parser_query = subparsers.add_parser("query", help="Cari snippet kode berdasarkan pertanyaan")
    parser_query.add_argument("question", help="Pertanyaan atau deskripsi pencarian")
    parser_query.add_argument("-k", "--top_k", type=int, default=3, help="Jumlah snippet yang ditampilkan (default: 3)")
    parser_query.add_argument("--db", default=str(DEFAULT_DB_DIR), help="Path direktori penyimpanan ChromaDB")

    # Command: chat / interactive
    parser_chat = subparsers.add_parser("chat", help="Buka sesi terminal interaktif untuk pencarian berulang")
    parser_chat.add_argument("--db", default=str(DEFAULT_DB_DIR), help="Path direktori penyimpanan ChromaDB")

    args = parser.parse_args()

    if args.command == "index":
        index_repo(args.repo_path, db_dir=Path(args.db), reset=args.reset)
    elif args.command == "query":
        query_code(args.question, top_k=args.top_k, db_dir=Path(args.db))
    elif args.command == "chat":
        interactive_mode(db_dir=Path(args.db))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
