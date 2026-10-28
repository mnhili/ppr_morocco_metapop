# TARP calibration diagnostics

The scripts use the model data in `../data`, the retained posterior in
`../outputs/inference`, and save TARP results in `../outputs/tarp`.

```powershell
python 6_metapopulation_model/TARP/scripts/run_tarp_diagnostic.py --workers 12
```

Replot saved TARP results:

```powershell
python 6_metapopulation_model/TARP/scripts/replot_tarp.py
```

Results are saved in `6_metapopulation_model/outputs/tarp`. The figure is
saved in `6_metapopulation_model/visualization`.
