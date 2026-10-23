#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2026/10/23 10:30:13
@Author  :   George Trenins
@Desc    :   Numerical helpers. Submodules are imported lazily, on first attribute access (PEP 562).
'''

from __future__ import print_function, division, absolute_import
import importlib
from typing import TYPE_CHECKING

__all__ = [
    "arrays",
    "calculus",
    "fitaux",
    "kde",
    "kernel_transforms",
    "nmtrans",
    "ou",
    "rfft",
    "special",
    "tcfs",
]

if TYPE_CHECKING:
    # Visible to static analysers and IDEs only; never executed.
    from . import (arrays, calculus, fitaux, kde, kernel_transforms,
                   nmtrans, ou, rfft, special, tcfs)


def __getattr__(name):
    """Import the submodule `name` on first access. The import system then
    binds it as an attribute of this package, so later lookups bypass this hook.
    """
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
