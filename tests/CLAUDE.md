# CLAUDE.md — `tests`

## Running

Run from the repository root, with the `tools` environment active and `env.sh` sourced:

```bash
mamba activate tools && source env.sh

pytest tests                                                  # whole suite, ~10 s
pytest tests/utils                                            # one directory
pytest tests/utils/test_tcfs.py                               # one file
pytest tests/utils/test_tcfs.py::test_correlate_equal_size    # one test
pytest tests -k correlate                                     # tests whose names match
```

- `rpmdgle` is not installed. `env.sh` puts the repository root on `PYTHONPATH`, and without it collection fails with `ModuleNotFoundError: No module named 'rpmdgle'`. `python -m pytest` also works without `env.sh`, but only from the repository root, because it adds the current directory to `sys.path`.
- Almost all of the runtime is in `tests/propagators`. The other directories finish in about a second.

## Layout and collection

- The directories mirror the package: `tests/utils` tests `rpmdgle/utils`, `tests/sysbath/spectral` tests `rpmdgle/sysbath/spectral`, and so on. Name a new file `test_<module>.py`.
- There is no pytest configuration, no `conftest.py` and no `__init__.py`. pytest therefore imports each test file under its bare file name, so **test file names must be unique across all of `tests/`**. A duplicate name stops collection with "import file mismatch".
- With no `conftest.py`, fixtures and helpers are defined in the test file that uses them.
