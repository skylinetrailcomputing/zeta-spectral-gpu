"""Intentional CI failure used to verify the skyline-oss-default merge gate.

Issue #2 asks us to confirm a red `test` check actually blocks the PR merge
button. This test MUST fail on purpose; its branch and PR are torn down once the
gate is verified. Nothing in this file should ever land on main.
"""

from __future__ import annotations


def test_ci_gate_blocks_merge_when_red():
    assert False, "intentional failure to verify the merge gate"
