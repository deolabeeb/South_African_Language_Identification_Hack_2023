"""Local CLI. Use .venv/Scripts/python.exe -m sa_lid --help on Windows."""
import argparse
import json
import sys

from .workflow import evaluate, predict, prepare, select


def main():
    parser = argparse.ArgumentParser(description="Reproducible South African language identification")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "select", "evaluate", "predict"]:
        command = sub.add_parser(name)
        command.add_argument("--run", required=True, help="Local run directory; ignored by Git")
        if name == "prepare":
            command.add_argument("--data", required=True)
        if name == "predict":
            command.add_argument("--text", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.data, args.run)
        elif args.command == "select":
            result = select(args.run)
        elif args.command == "evaluate":
            result = evaluate(args.run)
        else:
            result = predict(args.run, args.text)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
