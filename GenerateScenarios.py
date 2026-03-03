import re


# Helpers for file name


def _safe_float_str(x: float) -> str:
    """Format court et sûr pour noms de fichiers (pas de . ni d'espaces)."""
    s = f"{float(x):.6g}"
    s = s.replace(".", "p").replace("-", "m")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    return s
