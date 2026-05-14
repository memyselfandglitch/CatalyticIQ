"""Build py3Dmol views for Streamlit."""

from __future__ import annotations

from typing import Any


def structure_dict_to_cif(struct_dict: dict[str, Any], supercell: tuple[int, int, int] = (2, 2, 1)) -> str:
    from pymatgen.core import Structure  # type: ignore

    s = Structure.from_dict(struct_dict)
    if supercell != (1, 1, 1):
        s.make_supercell(supercell)
    return s.to(fmt="cif")


def py3dmol_view_from_cif(cif: str, *, width: int = 900, height: int = 560) -> Any:
    import py3Dmol  # type: ignore

    v = py3Dmol.view(width=width, height=height)
    v.addModel(cif, "cif")
    v.setStyle({"sphere": {"scale": 0.42}})
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
