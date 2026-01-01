Reference `README.md` for project overview and status.

## Before Committing

Run all tests and ensure they pass before committing changes:

```bash
cd build && make -j4 && ctest --output-on-failure
```