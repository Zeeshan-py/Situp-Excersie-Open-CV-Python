"""
Main Launcher for Sit-Up Exercise Monitor
Human Activity Monitoring System - PBL-CP-II

This launcher allows you to choose between:
1. GUI Version - Full featured with video controls
2. Simple Version - Standalone with webcam/video support
"""

import sys
import subprocess


def main():
    """Main launcher menu"""
    print("=" * 60)
    print("SIT-UP EXERCISE MONITOR")
    print("Human Activity Monitoring System")
    print("=" * 60)
    print("\nSelect application version:")
    print("  1. GUI Version (Recommended)")
    print("     - Full featured interface")
    print("     - Video file support")
    print("     - Statistics dashboard")
    print()
    print("  2. Simple Version")
    print("     - Standalone application")
    print("     - Webcam or video file")
    print("     - Lightweight")
    print()
    print("  3. Exit")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nLaunching GUI Version...")
        subprocess.run([sys.executable, "SitUpGUI.py"])
    elif choice == "2":
        print("\nLaunching Simple Version...")
        subprocess.run([sys.executable, "SitUpCounter_Simple.py"])
    elif choice == "3":
        print("\nExiting...")
    else:
        print("\nInvalid choice. Exiting...")


if __name__ == "__main__":
    main()
