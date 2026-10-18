# V3 quickstart

Run commands from `D:\WIFI Based CSI Project`.

## Inspect existing work

```powershell
Get-Content V3/manifests/models.json
Get-Content V3/manifests/experiment_index.json
Get-Content V3/results/metrics/v3_metrics.csv
```

## Check saved transformer inference

```powershell
py -3.12 V3/src/show_transformer.py --task presence
py -3.12 V3/src/show_transformer.py --task motion
```

This uses cached pipeline windows and does not access an ESP32.

For V3 inference, use only these classifiers:

```text
V3/models/presence_lgbm.txt
V3/models/presence_cnn.pt
V3/models/presence_tf.pt
V3/models/motion_lgbm.txt
V3/models/motion_cnn.pt
V3/models/motion_tf.pt
```

Do not use `V3/models/transformer_pretrained.pt` directly; it is a
self-supervised initialization backbone. Do not mix in `V2/models` or
`V4/models` when reproducing V3.

## Regenerate figures and registries

```powershell
py -3.12 V3/src/plots.py
py -3.12 V3/src/dashboard_plots.py
py -3.12 V3/src/build_registry.py
```

## Compile and run lightweight tests

```powershell
Get-ChildItem V3/src -Filter *.py | ForEach-Object { py -3.12 -m py_compile $_.FullName }
py -3.12 -m unittest discover -s V4/tests -v
```

## Load a checkpoint

The `.pt` files are state dictionaries. Reconstruct the matching `deep.py`
architecture before calling `load_state_dict()`. Presence uses two classes;
motion uses three. The existing smoke script is offline only. A complete live
serial-to-model application is not currently implemented or validated.