from __future__ import annotations

from typing import Iterable, Tuple
import pandas as pd


def _split_fen_fields(fen: pd.Series) -> pd.DataFrame:
    """Split FEN into fields: board, active, castle, ep, half, full.
    Expects a Series of FEN strings. Returns a DataFrame with columns.
    """
    parts = fen.astype("string").str.split(" ", n=5, expand=True)
    parts.columns = ["board", "active", "castle", "ep", "half", "full"]
    return parts


def extract_active_turn(fen: pd.Series, *, as_label: bool = True) -> pd.Series:
    """Return side to move from FEN ('w'/'b' or 'white'/'black')."""
    active = _split_fen_fields(fen)["active"].astype("string")
    if not as_label:
        return active
    return active.map({"w": "white", "b": "black"}).fillna("unknown")


def extract_castling_rights(fen: pd.Series) -> pd.DataFrame:
    """Return castling rights as booleans and a compact categorical state.

    Columns:
    - w_K, w_Q, b_k, b_q: per-side, per-rook-side castling flags
    - white_can_castle, black_can_castle: any castling right per side
    - castling_state: one of {none, white_only, black_only, both}
    - ep_available: True if en-passant target square is set (not '-')
    """
    fields = _split_fen_fields(fen)
    castle = fields["castle"].fillna("-").astype("string")
    out = pd.DataFrame(index=castle.index)
    out["w_K"] = castle.str.contains("K")
    out["w_Q"] = castle.str.contains("Q")
    out["b_k"] = castle.str.contains("k")
    out["b_q"] = castle.str.contains("q")
    out["white_can_castle"] = out[["w_K", "w_Q"]].any(axis=1)
    out["black_can_castle"] = out[["b_k", "b_q"]].any(axis=1)

    def _state(row) -> str:
        w = bool(row["white_can_castle"])  # type: ignore[index]
        b = bool(row["black_can_castle"])  # type: ignore[index]
        if w and b:
            return "both"
        if w:
            return "white_only"
        if b:
            return "black_only"
        return "none"

    out["castling_state"] = out.apply(_state, axis=1).astype("category")
    out["ep_available"] = fields["ep"].astype("string").ne("-")
    return out


def extract_queen_presence(fen: pd.Series) -> pd.DataFrame:
    """Return queen counts and presence flags for both sides.

    Columns: w_queens, b_queens, has_w_queen, has_b_queen, has_any_queen
    """
    board = _split_fen_fields(fen)["board"].astype("string")
    out = pd.DataFrame(index=board.index)
    out["w_queens"] = board.str.count("Q")
    out["b_queens"] = board.str.count("q")
    out["has_w_queen"] = out["w_queens"] > 0
    out["has_b_queen"] = out["b_queens"] > 0
    out["has_any_queen"] = out[["has_w_queen", "has_b_queen"]].any(axis=1)
    return out


def _bishop_colors_from_board(board: str) -> Tuple[set[str], set[str]]:
    """Return sets of colors {'light','dark'} where bishops reside for white/black.
    Board is the first FEN field. Assumes 'a1' is dark; color = 'dark' if (file+rank)%2==0.
    """
    white, black = set(), set()
    ranks = board.split("/")  # rank 8 .. 1
    for r_idx, rank in enumerate(ranks):
        file_idx = 0
        for ch in rank:
            if ch.isdigit():
                file_idx += int(ch)
                continue
            # piece occupies current square
            color = "dark" if (file_idx + r_idx) % 2 == 0 else "light"
            if ch == "B":
                white.add(color)
            elif ch == "b":
                black.add(color)
            file_idx += 1
    return white, black


def extract_bishop_parity(fen: pd.Series, *, show_progress: bool = False) -> pd.DataFrame:
    """Compute bishop color-sets and a parity indicator.

    Columns:
    - w_bishops: count
    - b_bishops: count
    - w_bishops_colors, b_bishops_colors: categorical in {none, light, dark, both}
    - bishops_parity: 'opposite' if exactly one bishop per side on opposite colors,
                      'same' if exactly one per side on same color,
                      'na' otherwise.
    """
    board = _split_fen_fields(fen)["board"].astype("string")

    def _summ(board_str: str):
        wset, bset = _bishop_colors_from_board(board_str)
        w_count, b_count = len(wset), len(bset)
        def set_label(s: set[str]) -> str:
            if not s:
                return "none"
            if len(s) == 2:
                return "both"
            return next(iter(s))

        w_label = set_label(wset)
        b_label = set_label(bset)
        if (w_count == 1) and (b_count == 1):
            parity = "opposite" if list(wset)[0] != list(bset)[0] else "same"
        else:
            parity = "na"
        return w_count, b_count, w_label, b_label, parity

    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
            tqdm.pandas(desc="Bishop parity")
            res = board.progress_apply(_summ)
        except Exception:
            # Fallback sin progreso si tqdm no está disponible
            res = board.apply(_summ)
    else:
        res = board.apply(_summ)
    out = pd.DataFrame(res.tolist(), index=board.index, columns=[
        "w_bishops", "b_bishops", "w_bishops_colors", "b_bishops_colors", "bishops_parity"
    ])
    out["w_bishops_colors"] = out["w_bishops_colors"].astype("category")
    out["b_bishops_colors"] = out["b_bishops_colors"].astype("category")
    out["bishops_parity"] = out["bishops_parity"].astype("category")
    return out
