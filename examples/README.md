# Run grid-search

## Prepare dataset

Please refer to `environments/Dockerfile` for installing requirements.

```shell
$cd ../data

$uv run python pixelprose_preprocess.py
```

## Run

```shell
$uv run python grid_search_eval.py
```
