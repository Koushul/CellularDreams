"""Spatial layouts and the transcriptomics dataset writer."""

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from travlr_sim.layouts import place
from travlr_sim.spatial import generate, neighbor_exposure


class LayoutTests(unittest.TestCase):
    def test_layers_are_separated_along_y(self):
        spatial = place("layers", 36, 3, np.random.default_rng(0))
        y = spatial["positions"][:, 1]
        means = [y[spatial["cell_type"] == kind].mean() for kind in range(3)]
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])

    def test_islands_are_tighter_than_the_mixture(self):
        spatial = place("islands", 45, 3, np.random.default_rng(1))
        positions = spatial["positions"]
        within = []
        between = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                if spatial["cell_type"][i] == spatial["cell_type"][j]:
                    within.append(distance)
                else:
                    between.append(distance)
        self.assertLess(np.mean(within), np.mean(between))

    def test_enhanced_neighbors_share_type(self):
        spatial = place("enhanced", 60, 3, np.random.default_rng(2))
        edges = spatial["edges"]
        same = 0
        for i, j in edges:
            same += int(spatial["cell_type"][i] == spatial["cell_type"][j])
        self.assertGreater(same / len(edges), 0.45)

    def test_grid_neighbors_are_orthogonal(self):
        spatial = place("basic", 20, 3, np.random.default_rng(3))
        pitch = spatial["pitch_um"]
        for i, j in spatial["edges"]:
            delta = np.abs(spatial["positions"][i] - spatial["positions"][j])
            self.assertAlmostEqual(delta.sum(), pitch, places=6)
            self.assertAlmostEqual(delta.min(), 0.0, places=6)


class DatasetTests(unittest.TestCase):
    def test_neighbor_exposure_matches_a_hand_sum(self):
        neighbors = [np.array([1, 2]), np.array([0]), np.array([0])]
        exposure = neighbor_exposure(np.array([4.0, 1.0, 7.0]), neighbors)
        np.testing.assert_allclose(exposure, [8.0, 4.0, 4.0])

    def test_program_dataset_records_cci_and_platforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = generate(path, layout="layers", expression="program", n_cells=36, seed=4)
            self.assertEqual(data.n_obs, 36)
            self.assertIn("spatial", data.obsm)
            np.testing.assert_allclose(data.obs["cci_impact"], 0.15 * data.obs["ligand_exposure"])
            ligand = data.layers["before_cci"][:, 0]
            touched = data.obs["ligand_exposure"].to_numpy() > 0
            self.assertTrue(touched.any())
            self.assertGreater(data.X[touched, 2].mean(), data.X[~touched, 2].mean())
            again = ad.read_h5ad(path / "observed.h5ad")
            self.assertEqual(again.n_obs, 36)
            for name in ("identity", "panel", "visium", "slideseq", "spillover", "batch"):
                platform = ad.read_h5ad(path / "platforms" / f"{name}.h5ad")
                self.assertIn("spatial", platform.obsm)
            params = (path / "params.json").read_text()
            self.assertIn("spatial-transcriptomics", params)

    def test_rust_file_keeps_atac_grn_and_kinetics(self):
        binary = Path("target/release/scmultisim-export")
        if not binary.is_file():
            self.skipTest("scmultisim-export is not built")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = generate(
                path,
                layout="islands",
                expression="rust",
                n_cells=16,
                seed=6,
                velocity=True,
                binary=binary,
            )
            self.assertEqual(data.layers["kon"].shape, data.X.shape)
            self.assertEqual(data.layers["koff"].shape, data.X.shape)
            self.assertEqual(data.layers["s"].shape, data.X.shape)
            self.assertEqual(data.layers["unspliced"].shape, data.X.shape)
            self.assertEqual(data.obsm["atac"].shape[0], data.n_obs)
            self.assertEqual(data.obsm["atac_clean"].shape, data.obsm["atac"].shape)
            self.assertEqual(data.uns["region_to_gene"].shape[1], data.n_vars)
            self.assertEqual(data.varm["giv_s"].shape[0], data.n_vars)
            self.assertEqual(data.obsm["cif_s"].shape[0], data.n_obs)
            self.assertGreater(len(data.uns["grn_effect"]), 0)
            self.assertEqual(len(data.uns["grn_target"]), len(data.uns["grn_regulator"]))
            self.assertEqual(data.uns["geff"].shape[0], data.n_vars)
            self.assertEqual(data.var["beta"].shape[0], data.n_vars)
            self.assertEqual(data.obsm["spatial"].shape, (16, 2))
            expected = data.layers["s"] * data.layers["kon"] / (data.layers["kon"] + data.layers["koff"])
            corr = np.corrcoef(np.asarray(data.X).ravel(), expected.ravel())[0, 1]
            self.assertGreater(corr, 0.4)
            again = ad.read_h5ad(path / "observed.h5ad")
            self.assertEqual(again.uns["grn_effect"].shape, data.uns["grn_effect"].shape)
            self.assertEqual(again.layers["s"].shape, data.layers["s"].shape)

    def test_travlr_scene_keeps_its_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = generate(Path(tmp), layout="zonation", expression="travlr", seed=5)
            y = data.obsm["spatial"][:, 1]
            self.assertTrue(np.allclose(y, y[0]))
            self.assertEqual(list(data.var_names), ["ligand", "decoy", "null", "receptor", "receptor_b", "target", "tf", "cycle"])
