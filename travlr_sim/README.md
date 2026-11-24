# travlr-sim

Python simulator for the tissue and expression tests that scMultiSim does not cover. The tissue loop owns positions, motility, division, and time. The expression engine owns the telegraph kinetics, time-varying coefficients, and distance-weighted ligand exposure.

The architecture rule and the numeric gates are in `PRECOMMIT_TRAVLR_SIM.md`. The scMultiSim reading that forced a reimplementation is in `SCMULTISIM_AUDIT.md`.

```python
from travlr_sim.scenes import build_scene
from travlr_sim.simulate import simulate

scene, rng = build_scene("zonation")
cells, history = simulate(scene, rng)
```

`travlr_sim.io.write_scene` writes `params.json`, an AnnData observation file, and a ground-truth AnnData file whose `uns` holds positions, spliced and unspliced levels, velocity, coefficients, exposure, impact, division rate, and lineage.
