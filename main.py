"""Convenient no-subcommand entry point matching the README examples."""

from __future__ import annotations

import sys

from cli import app, expand_color_arguments


if __name__ == "__main__":
    arguments = sys.argv[1:]
    known_commands = {
        "run",
        "clothing",
        "wallpaper",
        "validate",
        "preprocess",
        "analyze",
        "config-check",
    }
    if not arguments or arguments[0] not in known_commands:
        arguments = ["run", *arguments]
    app(args=expand_color_arguments(arguments), prog_name="python main.py")
