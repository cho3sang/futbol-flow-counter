from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox


def main() -> None:
    try:
        from ui import FutbolFlowApp
    except ModuleNotFoundError as exc:
        root = tk.Tk()
        root.withdraw()
        missing = exc.name or "required dependency"
        messagebox.showerror(
            "Missing dependency",
            f"Python is missing '{missing}'. Install the requirements with:\n\npython3 -m pip install -r requirements.txt",
        )
        root.destroy()
        sys.exit(1)

    root = tk.Tk()
    FutbolFlowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
