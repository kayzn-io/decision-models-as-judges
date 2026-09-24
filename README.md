# Decision models as judges

Typed decision models such as Jev and Laya are used as evaluation judges at each step of an agent evaluation loop on tau-bench. Runs are compared with and without these judges against deterministic ground truth, and every result is reproducible from committed caches.

## Results

<!-- results:start -->
Results are published here as evaluation gates complete.
<!-- results:end -->

## Development

```
uv sync
make check
```
