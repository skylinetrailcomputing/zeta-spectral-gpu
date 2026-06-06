"""Invariants for zero ingestion: the Odlyzko loader, parser, and caching.

All offline — no network. The parser is checked against an inline fixture of the
first ten ordinates and cross-checked against mpmath (proving the file column is
tau_n); caching/loading is exercised through a temporary table so no real
download or 100k file is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from zeta_spectral_gpu import zeros

# First ten lines of Odlyzko's `zeros1`, verbatim (fixed-width leading spaces).
ODLYZKO_HEAD = """\
     14.134725142
     21.022039639
     25.010857580
     30.424876126
     32.935061588
     37.586178159
     40.918719012
     43.327073281
     48.005150881
     49.773832478
"""

EXPECTED_HEAD = np.array(
    [
        14.134725142,
        21.022039639,
        25.010857580,
        30.424876126,
        32.935061588,
        37.586178159,
        40.918719012,
        43.327073281,
        48.005150881,
        49.773832478,
    ]
)


def test_parse_odlyzko_known_values():
    taus = zeros.parse_odlyzko(ODLYZKO_HEAD)
    assert taus.dtype == np.float64
    assert taus.size == 10
    np.testing.assert_allclose(taus, EXPECTED_HEAD)
    assert np.all(np.diff(taus) > 0)  # ascending


def test_parse_odlyzko_accepts_bytes():
    np.testing.assert_array_equal(
        zeros.parse_odlyzko(ODLYZKO_HEAD.encode("ascii")),
        zeros.parse_odlyzko(ODLYZKO_HEAD),
    )


def test_parsed_column_is_tau_matches_mpmath():
    """The table column is the imaginary part of each zero (forward check)."""
    import mpmath

    taus = zeros.parse_odlyzko(ODLYZKO_HEAD)
    for k in range(1, 6):  # first five; mpmath is quick at this size
        assert taus[k - 1] == pytest.approx(float(mpmath.zetazero(k).imag), abs=1e-6)


def _seed_tiny_table(tmp_path, monkeypatch, *, lines: str) -> zeros.OdlyzkoTable:
    """Register a 10-ordinate stand-in table backed by a local fixture file."""
    table = zeros.OdlyzkoTable(
        name="tiny", url="https://example.invalid/none", count=10, accuracy=1e-9
    )
    monkeypatch.setitem(zeros.ODLYZKO_TABLES, "tiny", table)
    (tmp_path / "odlyzko_tiny.txt").write_text(lines, encoding="ascii")
    return table


def test_load_from_cache_without_download(tmp_path, monkeypatch):
    _seed_tiny_table(tmp_path, monkeypatch, lines=ODLYZKO_HEAD)
    taus = zeros.load_odlyzko_ordinates(
        5, name="tiny", cache_dir=tmp_path, download=False
    )
    np.testing.assert_allclose(taus, EXPECTED_HEAD[:5])


def test_load_all_when_n_is_none(tmp_path, monkeypatch):
    _seed_tiny_table(tmp_path, monkeypatch, lines=ODLYZKO_HEAD)
    taus = zeros.load_odlyzko_ordinates(name="tiny", cache_dir=tmp_path, download=False)
    assert taus.size == 10


def test_missing_cache_without_download_raises(tmp_path, monkeypatch):
    _seed_tiny_table(tmp_path, monkeypatch, lines=ODLYZKO_HEAD)
    (tmp_path / "odlyzko_tiny.txt").unlink()
    with pytest.raises(FileNotFoundError):
        zeros.load_odlyzko_ordinates(name="tiny", cache_dir=tmp_path, download=False)


def test_request_more_than_available_raises(tmp_path, monkeypatch):
    _seed_tiny_table(tmp_path, monkeypatch, lines=ODLYZKO_HEAD)
    with pytest.raises(ValueError, match="requested 11"):
        zeros.load_odlyzko_ordinates(
            11, name="tiny", cache_dir=tmp_path, download=False
        )


def test_partial_download_warns(tmp_path, monkeypatch):
    half = "\n".join(ODLYZKO_HEAD.split()[:5]) + "\n"  # 5 of the expected 10
    _seed_tiny_table(tmp_path, monkeypatch, lines=half)
    with pytest.warns(UserWarning, match="possible partial download"):
        zeros.load_odlyzko_ordinates(name="tiny", cache_dir=tmp_path, download=False)


def test_load_ordinates_dispatch(tmp_path, monkeypatch):
    _seed_tiny_table(tmp_path, monkeypatch, lines=ODLYZKO_HEAD)
    np.testing.assert_allclose(
        zeros.load_ordinates(3, source="tiny", cache_dir=tmp_path),
        EXPECTED_HEAD[:3],
    )
    # mpmath branch (small n is fast) and unknown source.
    np.testing.assert_allclose(
        zeros.load_ordinates(3, source="mpmath", cache_dir=tmp_path),
        EXPECTED_HEAD[:3],
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="unknown source"):
        zeros.load_ordinates(3, source="nope", cache_dir=tmp_path)


def test_loaded_ordinates_unfold_unchanged():
    """unfold() consumes the loaded array exactly as before (interface check)."""
    taus = zeros.parse_odlyzko(ODLYZKO_HEAD)
    unfolded = zeros.unfold(taus)
    assert unfolded.shape == taus.shape
    assert np.all(np.diff(unfolded) > 0)
