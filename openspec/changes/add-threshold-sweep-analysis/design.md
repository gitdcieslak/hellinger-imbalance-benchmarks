# Design

## Threshold Metrics

Threshold sweeps use positive-class scores from the existing classifier scoring helper. For each threshold, scores greater than or equal to the threshold are classified as positive.

The initial threshold ladder is:

- 0.01
- 0.05
- 0.10
- 0.25
- 0.50

Metrics per threshold are precision, recall, F1, and balanced accuracy.

## Runner Output

Threshold sweep runs emit one JSONL record per experiment, dataset, model, seed, split, skew ratio, and threshold. Records include the threshold value, threshold metrics, package versions, and synthetic dataset metadata.

## Reporting

Threshold summaries aggregate by skew ratio, model id, and threshold. CSV output is flat and machine-readable. Markdown output groups simple tables by skew ratio.

Generated threshold outputs remain under `results/` and `reports/`, which are gitignored.
