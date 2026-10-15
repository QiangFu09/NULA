# Contributing

For a bug report, include the command, Python and dependency versions, seed,
and the smallest configuration that reproduces the problem. Include the
traceback or relevant result file when available.

Changes to numerical updates, random-number ordering, stopping rules, or loss
aggregation should state their effect on the published experiments. Keep such
changes separate from documentation and code organization changes.

Run the checks before proposing a change:

```bash
python -m unittest discover -s tests -v
```
