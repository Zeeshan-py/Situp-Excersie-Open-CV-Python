"""Launcher for the production sit-up monitoring applications."""

from __future__ import annotations

import logging
import subprocess
import sys

APP_INFO = {
    "name": "Sit-Up Monitor",
    "version": "2.0.0",
    "tagline": "Production MediaPipe Pose rewrite",
}

QUICK_TIPS = [
    "Use a strict side view for the most reliable hip-angle tracking.",
    "Keep your full body visible, especially shoulders, hips, knees, and ankles.",
    "Place the camera near hip height with even front lighting.",
    "The GUI is best for review sessions, and the simple app is best for quick checks.",
]

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure launcher logging."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def print_menu() -> None:
    """Render the launcher menu with version information and setup tips."""

    print("=" * 68)
    print(f"{APP_INFO['name']}  |  v{APP_INFO['version']}")
    print(APP_INFO["tagline"])
    print("=" * 68)
    print("\nQuick Tips:")
    for tip in QUICK_TIPS:
        print(f"  - {tip}")
    print("\nChoose an application:")
    print("  1. GUI Version")
    print("  2. Simple Version")
    print("  3. Exit")
    print()


def run_script(script_name: str) -> int:
    """Launch one of the application entry points."""

    LOGGER.info("Launching %s", script_name)
    return subprocess.call([sys.executable, script_name])


def main() -> int:
    """Display the launcher menu and run the selected application."""

    configure_logging()
    print_menu()
    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        return run_script("SitUpGUI.py")
    if choice == "2":
        return run_script("SitUpCounter_Simple.py")
    if choice == "3":
        LOGGER.info("Exiting launcher.")
        return 0

    LOGGER.error("Invalid choice: %s", choice)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
