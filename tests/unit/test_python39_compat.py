"""
Guard for the SDK's declared Python floor (``requires-python = ">=3.9"``).

``str | None`` in a function signature or variable annotation is evaluated when
the ``def``/``class`` body runs, and raises ``TypeError`` on Python 3.9 unless the
module has ``from __future__ import annotations``. The SDK shipped 1.3.0
candidates that could not even be imported on 3.9 this way, and the monorepo's CI
only runs one newer interpreter, so nothing noticed. This check is static so it
catches the mistake on any interpreter.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

# find_spec locates the package without importing it: importing is the very thing
# that fails on 3.9 when the guarded mistake is made.
_spec = importlib.util.find_spec("whiteboxxai")
SDK_PACKAGE = Path(_spec.origin).parent


def _annotation_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in (
                args.posonlyargs
                + args.args
                + args.kwonlyargs
                + [a for a in (args.vararg, args.kwarg) if a is not None]
            ):
                if arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _uses_pep604_union(annotation: ast.AST) -> bool:
    return any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        for node in ast.walk(annotation)
    )


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


@pytest.mark.parametrize(
    "path",
    sorted(SDK_PACKAGE.rglob("*.py")),
    ids=lambda p: str(p.relative_to(SDK_PACKAGE)).replace("\\", "/"),
)
def test_pep604_annotations_need_the_future_import(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    offending = sorted({a.lineno for a in _annotation_nodes(tree) if _uses_pep604_union(a)})

    if offending:
        assert _has_future_annotations(tree), (
            f"{path.name} uses `X | Y` annotations (lines {offending[:5]}...) without "
            "`from __future__ import annotations`; that raises TypeError at import on "
            "Python 3.9, which the SDK still supports."
        )


@pytest.mark.skipif(
    sys.version_info >= (3, 10),
    reason="only meaningful on Python 3.9, where `X | Y` is not evaluable",
)
def test_package_imports_on_python_39():
    import importlib

    import whiteboxxai

    importlib.reload(whiteboxxai)
