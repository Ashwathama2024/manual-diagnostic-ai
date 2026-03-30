from __future__ import annotations

import argparse

from common import ensure_command, ensure_ollama_models, ensure_python_package, load_config, resolve_path


BASE_REQUIRED_MODULES = [
    "yaml",
    "langchain_text_splitters",
    "ollama",
    "gradio",
]

VECTOR_REQUIRED_MODULES = [
    "lancedb",
    "langchain_ollama",
]


def print_check(name: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def run_check(name: str, func, failures: list[str]) -> None:
    try:
        detail = func()
        print_check(name, True, detail)
    except Exception as exc:
        print_check(name, False, str(exc))
        failures.append(f"{name}: {exc}")


def require_exists(path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def resolve_retrieval_mode(args_mode: str | None, cfg: dict) -> str:
    mode = args_mode or cfg.get("retrieval", {}).get("mode", "vector")
    allowed = {"vector", "vectorless", "hybrid"}
    if mode not in allowed:
        raise ValueError(f"Unsupported retrieval mode `{mode}`. Use one of: {', '.join(sorted(allowed))}.")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local offline RAG prerequisites.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--require-pdf", action="store_true", help="Fail if no source PDFs are present")
    parser.add_argument("--require-index", action="store_true", help="Fail if the LanceDB directory is empty")
    parser.add_argument(
        "--retrieval-mode",
        choices=["vector", "vectorless", "hybrid"],
        default=None,
        help="Validate prerequisites for a specific retrieval mode.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    retrieval_mode = resolve_retrieval_mode(args.retrieval_mode, cfg)

    # Resolve paths — support both new multi-manual keys and legacy single-file keys
    paths = cfg["paths"]
    if "raw_dir" in paths:
        raw_dir = resolve_path(paths["raw_dir"])
        processed_dir = resolve_path(paths["processed_dir"])
    else:
        raw_dir = resolve_path(paths["input_pdf"]).parent
        processed_dir = resolve_path(paths["markdown_output"]).parent

    chunks_output = resolve_path(paths["chunks_output"])
    lancedb_dir = resolve_path(paths["lancedb_dir"])

    failures: list[str] = []

    required_modules = list(BASE_REQUIRED_MODULES)
    if retrieval_mode in {"vector", "hybrid"} or args.require_index:
        required_modules.extend(VECTOR_REQUIRED_MODULES)

    for module_name in required_modules:
        run_check(
            f"Python module `{module_name}`",
            lambda module_name=module_name: (ensure_python_package(module_name), "available")[1],
            failures,
        )

    run_check(
        "Ollama CLI",
        lambda: (ensure_command("ollama", "Install Ollama locally and ensure it is on PATH."), "found on PATH")[1],
        failures,
    )

    required_models = [cfg["models"]["reasoning"]]
    if retrieval_mode in {"vector", "hybrid"}:
        required_models.append(cfg["models"]["embedding"])

    run_check(
        "Ollama models",
        lambda: (ensure_ollama_models(required_models), "required models are installed")[1],
        failures,
    )

    # Raw PDFs check
    if args.require_pdf:
        pdf_files = list(raw_dir.glob("*.pdf")) if raw_dir.exists() else []
        if not pdf_files:
            print_check("Source PDFs", False, f"No PDFs found in {raw_dir}")
            failures.append(f"Source PDFs: no PDFs in {raw_dir}")
        else:
            print_check("Source PDFs", True, f"{len(pdf_files)} file(s) in {raw_dir}")
    else:
        pdf_count = len(list(raw_dir.glob("*.pdf"))) if raw_dir.exists() else 0
        print_check("Source PDFs", pdf_count > 0, f"{pdf_count} file(s) in {raw_dir}")

    # Processed Markdown files
    md_files = list(processed_dir.glob("*.md")) if processed_dir.exists() else []
    print_check("Processed Markdown files", bool(md_files), f"{len(md_files)} file(s) in {processed_dir}")

    print_check("Chunks JSONL", chunks_output.exists(), str(chunks_output))

    index_present = lancedb_dir.exists() and any(lancedb_dir.iterdir())
    if retrieval_mode in {"vector", "hybrid"} or args.require_index:
        if args.require_index:
            run_check(
                "LanceDB directory",
                lambda: (
                    str(lancedb_dir) if index_present
                    else (_ for _ in ()).throw(FileNotFoundError(f"LanceDB directory is empty: {lancedb_dir}"))
                ),
                failures,
            )
        else:
            print_check("LanceDB directory", index_present, str(lancedb_dir))
    else:
        print_check("LanceDB directory", index_present,
                    f"optional for retrieval mode `{retrieval_mode}`: {lancedb_dir}")

    if failures:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(failures))
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
