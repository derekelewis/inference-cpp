Reference `README.md` for project overview and status.

## Before Committing

Run fast tests (excludes slow forward pass tests):

```bash
cd build && make -j4 && ctest -E forward --output-on-failure
```

## Running All Tests

To run all tests including slow forward pass tests:

```bash
ctest --output-on-failure
```

To run only the slow forward pass tests:

```bash
ctest -L slow --output-on-failure
# OR
./qwen3_forward_tests
```