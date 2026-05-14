"""Build py3Dmol views for Streamlit."""

from __future__ import annotations

import html
from typing import Any


ELEMENT_COLORS = {
    "O": "#d62728",
    "Zn": "#7f7f7f",
    "Fe": "#e67e22",
    "K": "#8e44ad",
    "Ti": "#1f77b4",
    "Pt": "#b0b0b0",
    "Pd": "#9edae5",
    "Cu": "#bc6c25",
    "Ni": "#2ca02c",
    "Co": "#17becf",
    "Ru": "#9467bd",
    "Rh": "#aec7e8",
    "Ag": "#c7c7c7",
    "Au": "#f1c40f",
}


def element_color(symbol: str) -> str:
    return ELEMENT_COLORS.get(symbol, "#4d4d4d")


def structure_dict_to_cif(struct_dict: dict[str, Any], supercell: tuple[int, int, int] = (2, 2, 1)) -> str:
    from pymatgen.core import Structure  # type: ignore

    s = Structure.from_dict(struct_dict)
    if supercell != (1, 1, 1):
        s.make_supercell(supercell)
    return s.to(fmt="cif")


def py3dmol_view_from_cif(
    cif: str,
    *,
    width: int = 900,
    height: int = 560,
    elements: list[str] | None = None,
) -> Any:
    import py3Dmol  # type: ignore

    v = py3Dmol.view(width=width, height=height)
    v.addModel(cif, "cif")
    v.setStyle({"sphere": {"scale": 0.36}})
    for symbol in elements or []:
        v.setStyle({"elem": symbol}, {"sphere": {"scale": 0.42, "color": element_color(symbol)}})
    v.addUnitCell()
    v.setBackgroundColor("white")
    v.zoomTo()
    return v


def py3dmol_html(view: Any) -> str:
    """Return standalone HTML for a py3Dmol view."""
    make_html = getattr(view, "_make_html", None)
    if callable(make_html):
        return str(make_html())
    repr_html = getattr(view, "_repr_html_", None)
    if callable(repr_html):
        return str(repr_html())
    raise RuntimeError("py3Dmol view cannot be converted to HTML.")


def py3dmol_html_with_legend(view: Any, legend_items: list[dict[str, Any]]) -> str:
    """Return py3Dmol HTML with a compact bottom-right species legend."""
    rows = []
    for item in legend_items:
        symbol = html.escape(str(item.get("element", "")))
        count = html.escape(str(item.get("atoms", "")))
        role = html.escape(str(item.get("role", "")))
        color = html.escape(str(item.get("color", element_color(symbol))))
        rows.append(
            "<div style='display:flex;align-items:center;gap:7px;margin:3px 0;white-space:nowrap;'>"
            f"<span style='width:10px;height:10px;border-radius:999px;background:{color};"
            "display:inline-block;border:1px solid rgba(0,0,0,.22);'></span>"
            f"<strong style='min-width:22px;'>{symbol}</strong>"
            f"<span style='color:#4b5563;'>x{count}</span>"
            f"<span style='color:#6b7280;'>- {role}</span>"
            "</div>"
        )
    legend = "".join(rows)
    return (
        "<div style='position:relative;width:100%;height:590px;'>"
        f"{py3dmol_html(view)}"
        "<div style='position:absolute;right:14px;bottom:14px;z-index:5;"
        "max-width:min(430px,calc(100% - 28px));padding:10px 12px;"
        "background:rgba(255,255,255,.92);border:1px solid rgba(17,24,39,.12);"
        "border-radius:8px;box-shadow:0 8px 24px rgba(15,23,42,.12);"
        "font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,sans-serif;"
        "font-size:12px;line-height:1.25;color:#111827;'>"
        "<div style='font-weight:700;margin-bottom:5px;'>Species legend</div>"
        f"{legend}"
        "</div></div>"
    )
