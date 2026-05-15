from pathlib import Path

from .utils import SUPPORTED_EXTENSIONS, is_url, sanitize_project_name

MODELS = [
    ("turbo",    "Turbo     — fast, great accuracy (recommended)"),
    ("large-v3", "Large v3  — highest accuracy, slowest"),
    ("medium",   "Medium    — balanced speed / accuracy"),
    ("small",    "Small     — fast, lighter RAM"),
    ("tiny",     "Tiny      — fastest, lowest accuracy"),
]


def prompt_model_menu() -> str:
    """Return the selected Whisper model name."""
    print("\nWhich model would you like to use?")
    for i, (_, label) in enumerate(MODELS, start=1):
        print(f"  [{i}] {label}")
    while True:
        choice = input("Choice [1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            return MODELS[int(choice) - 1][0]
        print(f"Enter a number between 1 and {len(MODELS)}.\n")


def prompt_mode_menu() -> str:
    """Return 'single', 'urls', or 'batch'."""
    print("\nWhat would you like to do?")
    print("  [1] Transcribe a local audio/video file")
    print("  [2] Download and transcribe video URLs  (project mode)")
    print("  [3] Batch transcribe multiple local files")
    while True:
        choice = input("Choice [1]: ").strip() or "1"
        if choice == "1":
            return "single"
        if choice == "2":
            return "urls"
        if choice == "3":
            return "batch"
        print("Enter 1, 2, or 3.\n")


def prompt_for_urls() -> list[str]:
    """Interactively collect one or more video URLs."""
    print("\nEnter video URLs, one per line. Press Enter on a blank line when done.")
    urls: list[str] = []
    while True:
        line = input(f"  URL {len(urls) + 1}: ").strip()
        if not line:
            if not urls:
                print("At least one URL is required.\n")
                continue
            break
        if not is_url(line):
            print("  Not a recognised URL (must start with http:// or https://). Try again.")
            continue
        urls.append(line)
    return urls


def prompt_for_batch_files() -> list[str]:
    """Interactively collect one or more local file paths."""
    print("\nEnter file paths, one per line. Press Enter on a blank line when done.")
    paths: list[str] = []
    while True:
        line = input(f"  File {len(paths) + 1}: ").strip().strip('"').strip("'")
        if not line:
            if not paths:
                print("At least one file is required.\n")
                continue
            break
        p = Path(line).expanduser()
        if not p.exists():
            print(f"  File not found: {p}")
            continue
        if not p.is_file():
            print(f"  Not a file: {p}")
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(f"  Unsupported extension '{p.suffix}'. Allowed: {allowed}")
            continue
        paths.append(str(p))
    return paths


def prompt_for_audio_file() -> Path:
    while True:
        raw = input("Enter full path to the audio file: ").strip().strip('"').strip("'")
        if not raw:
            print("No path entered. Try again.\n")
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"File not found: {path}\n")
            continue
        if not path.is_file():
            print(f"Not a file: {path}\n")
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(f"Unsupported extension: {path.suffix}\nAllowed: {allowed}\n")
            continue
        return path


def prompt_project_name(project_root: Path) -> tuple[str, Path]:
    """Prompt for a project name and return (sanitized_name, created_folder_path)."""
    project_root.mkdir(parents=True, exist_ok=True)
    while True:
        raw = input("Project name (a folder will be created): ")
        safe = sanitize_project_name(raw)
        if safe is None:
            print("Name produced no valid characters. Try again.\n")
            continue
        project_dir = project_root / safe
        if project_dir.exists():
            confirm = input(f"Folder '{safe}' already exists. Use it? [y/N] ").strip().lower()
            if confirm != "y":
                continue
        project_dir.mkdir(parents=True, exist_ok=True)
        return safe, project_dir
