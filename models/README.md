# Models

Pre-trained RadWet-L classifiers from the original repository
([gro5-AberUni/ALOS-LBand](https://github.com/gro5-AberUni/ALOS-LBand),
`Scripts/Model/`, GPL-3.0), by Gregory Oakes, Aberystwyth University.

| File | Origin | Change |
|------|--------|--------|
| `XGBoost_Water.ubj` / `.json` | `Trained_XGBoostModel_02_Amazon_Water.model` | Converted from the legacy XGBoost binary format |
| `XGBoost_Flood.ubj` / `.json` | `Trained_XGBoostModel_02_Amazon_Flood.model` | Converted from the legacy XGBoost binary format |
| `Scaler_Water.pkl` | `Scaler_Water.pkl` | None (byte-identical copy) |
| `Scaler_Flood.pkl` | `Scaler_Flood.pkl` | None (byte-identical copy) |

The models were not retrained. The legacy `.model` format can only be read by
XGBoost < 3.1, so the boosters were re-saved in the current UBJ format (and as
JSON for inspection). The pipeline loads the `.ubj` files.

The conversion is reproducible. With XGBoost 3.0.x, the snippet below produces
files byte-identical to the ones in this folder, and their predictions match
the legacy models exactly:

```python
import xgboost as xgb

for kind in ("Water", "Flood"):
    booster = xgb.Booster()
    booster.load_model(f"Scripts/Model/Trained_XGBoostModel_02_Amazon_{kind}.model")
    booster.save_model(f"XGBoost_{kind}.ubj")
    booster.save_model(f"XGBoost_{kind}.json")
```

## Inputs each model expects

The original apply script lists 21 (water) and 9 (flood) features, but the
shipped scalers and boosters expect 8 and 4. The feature lists come from the
original training scripts (`trainXGBoostGPU_Water.py:48`,
`trainXGBoostGPU_Flood.py:56`):

- **Water (8):** `HHMean, HVMean, NDPIMean, NDPIstdDev, IncstdDev, SlopeMean, SlopestdDev, HandstdDev`
- **Flood (4):** `HHMean, HVMean, NDPIMean, IncMean`

Both are per-segment statistics computed on the 50 m working grid, with HH/HV
in dB. The flood model outputs probabilities of exactly 0 or 1, so its 0.99
threshold has no effect.
