"""Feature tests for travlr-sim, against the gates in PRECOMMIT_TRAVLR_SIM.md."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import spearmanr

from travlr_sim.calibrate import (
    displacement_to_noise,
    in_range,
    log10_library_cv,
    median_abs_correlation,
    morans_i,
    zero_fraction,
)
from travlr_sim.exposure import pairwise_exposure, receptor_gate
from travlr_sim.kinetics import beta_poisson_draw, beta_poisson_mean, piecewise_linear, telegraph_step
from travlr_sim.observe import slideseq_beads, spillover, visium_spots
from travlr_sim.scenes import build_scene
from travlr_sim.simulate import Cell, Edge, Scene, simulate


def _blank_cell(index, pos, receptor=(1.0, 1.0), ligand=5.0, moves=False, speed=0.0, chemotaxis=False):
    spliced = np.array([ligand, 1.0, 1.0, receptor[0], receptor[1], 1.0, 1.0, 1.0])
    return Cell(
        id=index,
        parent=None,
        birth=0.0,
        death=None,
        pos=np.array(pos, dtype=float),
        heading=0.0,
        speed=speed,
        unspliced=spliced.copy(),
        spliced=spliced.copy(),
        promoter_on=np.ones(8, dtype=bool),
        receptor_subunits=np.array(receptor, dtype=float),
        division_rate=0.0,
        moves=moves,
        chemotaxis=chemotaxis,
    )


class EquivalenceTests(unittest.TestCase):
    def test_beta_poisson_mean_within_five_percent(self):
        rng = np.random.default_rng(0)
        kon, koff, s = 2.0, 3.0, 12.0
        draws = beta_poisson_draw(rng, kon, koff, s, 2000)
        analytic = beta_poisson_mean(kon, koff, s)
        self.assertLess(abs(draws.mean() - analytic) / analytic, 0.05)

    def test_gene_means_track_analytic_target(self):
        rng = np.random.default_rng(1)
        kons = np.array([0.5, 1.0, 2.0, 4.0])
        koffs = np.array([1.0, 1.0, 1.0, 2.0])
        ss = np.array([8.0, 10.0, 6.0, 20.0])
        ok = 0
        for kon, koff, s in zip(kons, koffs, ss):
            draws = beta_poisson_draw(rng, kon, koff, s, 200)
            analytic = beta_poisson_mean(kon, koff, s)
            if abs(draws.mean() - analytic) / analytic < 0.15:
                ok += 1
        self.assertGreaterEqual(ok / len(kons), 0.9)

    def test_promoter_on_equilibrium(self):
        u, y = 0.0, 0.0
        s_rate, beta, degradation, dt = 4.0, 0.4, 1.0, 0.01
        for _ in range(8000):
            u, y = telegraph_step(u, y, True, s_rate, beta, degradation, dt)
        self.assertLess(abs(u - s_rate / beta) / (s_rate / beta), 0.10)
        self.assertLess(abs(y - s_rate / degradation) / (s_rate / degradation), 0.10)

    def test_no_ligands_ignore_position(self):
        def run(positions):
            cells = [_blank_cell(i, pos) for i, pos in enumerate(positions)]
            scene = Scene("static", 0, ["ligand", "decoy", "null", "receptor", "receptor_b", "target", "tf", "cycle"], [], cells, stationary=True, duration=2.0, dt=0.5)
            _, history = simulate(scene, np.random.default_rng(0))
            return history.spliced[-1]

        a = run([[10.0, 10.0], [80.0, 80.0]])
        b = run([[150.0, 20.0], [30.0, 170.0]])
        np.testing.assert_allclose(a, b)


class ExposureTests(unittest.TestCase):
    def test_three_cell_kernel(self):
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        levels = np.array([1.0, 0.0, 0.0])
        got = pairwise_exposure(positions, levels, 10.0)
        expected = np.array([0.0, np.exp(-1.0), np.exp(-1.0)])
        np.testing.assert_allclose(got, expected)

    def test_missing_subunit_closes_gate(self):
        gate = receptor_gate(np.array([[2.0, 2.0], [2.0, 0.0], [0.0, 4.0]]))
        self.assertGreater(gate[0], 0.0)
        self.assertEqual(gate[1], 0.0)
        self.assertEqual(gate[2], 0.0)

    def test_moving_cell_updates_exposure_on_that_step(self):
        cells = [
            _blank_cell(0, [0.0, 100.0], ligand=20.0),
            _blank_cell(1, [80.0, 100.0], ligand=0.0, moves=True, speed=40.0),
        ]
        cells[1].heading = np.pi
        edge = Edge(0, 5, [(0.0, 1.0), (4.0, 1.0)], "ligand", 40.0)
        scene = Scene(
            "move",
            0,
            ["ligand", "decoy", "null", "receptor", "receptor_b", "target", "tf", "cycle"],
            [edge],
            cells,
            duration=1.0,
            dt=0.5,
            receptor_genes=[3, 4],
            target_gene=5,
            ligand_gene=0,
        )
        _, history = simulate(scene, np.random.default_rng(0))
        self.assertGreater(history.exposure[2][1, 0], history.exposure[0][1, 0])


class RegulationAndLineageTests(unittest.TestCase):
    def test_piecewise_doubles(self):
        self.assertAlmostEqual(piecewise_linear([(0.0, 1.0), (6.0, 1.0), (6.0, 2.0), (12.0, 2.0)], 8.0), 2.0)

    def test_doubling_raises_target(self):
        def run(knots):
            cells = [
                _blank_cell(0, [50.0, 50.0], ligand=15.0),
                _blank_cell(1, [55.0, 50.0], ligand=0.0),
            ]
            scene = Scene(
                "coef",
                0,
                ["ligand", "decoy", "null", "receptor", "receptor_b", "target", "tf", "cycle"],
                [Edge(0, 5, knots, "ligand", 40.0)],
                cells,
                duration=8.0,
                dt=0.5,
                stationary=True,
                receptor_genes=[3, 4],
                target_gene=5,
            )
            _, history = simulate(scene, np.random.default_rng(2))
            return history.spliced[-1][1, 5]

        low = run([(0.0, 0.2), (8.0, 0.2)])
        high = run([(0.0, 0.2), (4.0, 0.2), (4.0, 2.0), (8.0, 2.0)])
        self.assertGreater(high, low)

    def test_destructive_sampling_and_lineage(self):
        scene, rng = build_scene("time_varying")
        cells, _ = simulate(scene, rng)
        seen = []
        for cell in cells:
            seen.extend(cell.samples)
            self.assertEqual(len(cell.samples), len(set(cell.samples)))
        ids = [cell.id for cell in cells for _ in cell.samples]
        self.assertEqual(len(ids), len(set(ids)))
        sampled_ids = {cell.id for cell in cells if cell.samples}
        children = [cell for cell in cells if cell.parent in sampled_ids]
        self.assertTrue(all(cell.parent is not None for cell in children) or children == [])
        by_id = {cell.id: cell for cell in cells}
        for cell in cells:
            if cell.parent is None:
                continue
            self.assertIn(cell.parent, by_id)

    def test_receptor_cv(self):
        rng = np.random.default_rng(5)
        mean, cv = 4.0, 0.5
        sigma = np.sqrt(np.log(1.0 + cv**2))
        levels = mean * np.exp(rng.normal(0.0, sigma, size=4000))
        realized = levels.std() / levels.mean()
        self.assertLess(abs(realized - cv), 0.05)

    def test_knockout_removes_only_that_cells_impact(self):
        cells = [
            _blank_cell(0, [40.0, 50.0], ligand=20.0, receptor=(4.0, 4.0)),
            _blank_cell(1, [48.0, 50.0], ligand=0.0, receptor=(4.0, 4.0)),
            _blank_cell(2, [70.0, 50.0], ligand=0.0, receptor=(4.0, 4.0)),
        ]
        scene = Scene(
            "ko",
            0,
            ["ligand", "decoy", "null", "receptor", "receptor_b", "target", "tf", "cycle"],
            [Edge(0, 5, [(0.0, 1.0), (8.0, 1.0)], "ligand", 30.0)],
            cells,
            duration=8.0,
            dt=0.5,
            stationary=True,
            receptor_genes=[3, 4],
            target_gene=5,
            ligand_gene=0,
            knockout_time=4.0,
            knockout_gene=3,
            knockout_cells=[1],
        )
        _, history = simulate(scene, np.random.default_rng(3))
        late = history.impact[-1]
        self.assertEqual(late[1, 5], 0.0)
        self.assertGreater(late[2, 5], 0.0)


class ObservationTests(unittest.TestCase):
    def test_spillover_conserves_and_matches_fraction(self):
        counts = np.array([[10.0, 0.0], [0.0, 8.0]])
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = spillover(counts, positions, 0.1)
        np.testing.assert_allclose(out.sum(), counts.sum())
        np.testing.assert_allclose(out[0], [9.0, 0.8])
        np.testing.assert_allclose(out[1], [1.0, 7.2])

    def test_spot_and_bead_sums_match_captured_cells(self):
        rng = np.random.default_rng(4)
        positions = rng.uniform(0, 80, size=(30, 2))
        counts = rng.poisson(3, size=(30, 4)).astype(float)
        vis, vcent, _ = visium_spots(counts, positions, 80.0)
        beads, bcent, _ = slideseq_beads(counts, positions, 80.0)
        for agg, centers, radius in ((vis, vcent, 55.0), (beads, bcent, 10.0)):
            distance = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)
            captured = distance.min(axis=1) <= radius
            np.testing.assert_allclose(agg.sum(), counts[captured].sum())


class ScenarioTests(unittest.TestCase):
    def test_chemotaxis_bias(self):
        scene, rng = build_scene("chemotaxis")
        cells, history = simulate(scene, rng)
        start = {int(i): pos for i, pos in zip(history.cell_ids[0], history.positions[0])}
        end_ids = history.cell_ids[-1]
        end_pos = history.positions[-1]
        by_id = {cell.id: cell for cell in cells}
        pos_shift = []
        neg_shift = []
        for index, cell_id in enumerate(end_ids):
            cell = by_id[int(cell_id)]
            if cell_id not in start or not cell.moves:
                continue
            shift = end_pos[index, 0] - start[int(cell_id)][0]
            if cell.receptor_subunits.min() > 0:
                pos_shift.append(shift)
            else:
                neg_shift.append(shift)
        self.assertGreater(np.mean(pos_shift), np.mean(neg_shift))

    def test_zonation_is_stationary_and_ordered(self):
        scene, rng = build_scene("zonation")
        _, history = simulate(scene, rng)
        np.testing.assert_allclose(history.positions[0], history.positions[-1])
        order = np.argsort(history.positions[-1][:, 0])
        target = history.spliced[-1][order, scene.target_gene]
        self.assertGreater(spearmanr(np.arange(len(target)), target).statistic, 0.8)

    def test_decoy_edge_is_zero(self):
        scene, _ = build_scene("decoy")
        decoy = [edge for edge in scene.edges if edge.source == 1][0]
        self.assertTrue(all(value == 0.0 for _, value in decoy.knots))

    def test_cycle_score_tracks_division(self):
        scene, rng = build_scene("coupled")
        _, history = simulate(scene, rng)
        score = np.concatenate(history.cycle_score)
        rate = np.concatenate(history.division_rate)
        self.assertGreater(spearmanr(score, rate).statistic, 0.5)

    def test_library_scenes_exist(self):
        for name in (
            "coupled",
            "fork",
            "crossing",
            "radial",
            "rim",
            "zonation",
            "signal_in_place",
            "decoy",
            "il21_like",
            "time_varying",
            "chemotaxis",
        ):
            scene, rng = build_scene(name)
            self.assertEqual(scene.name, name)
            cells, history = simulate(scene, rng)
            self.assertGreater(len(history.time), 0)
            self.assertGreater(len(cells), 0)


class CalibrationTests(unittest.TestCase):
    def test_reported_statistics_are_finite(self):
        scene, rng = build_scene("zonation")
        _, history = simulate(scene, rng)
        counts = rng.poisson(np.clip(history.spliced[-1], 0, None) * 0.05)
        values = {
            "log10_library_cv": log10_library_cv(counts),
            "zero_fraction": zero_fraction(counts),
            "median_abs_correlation": median_abs_correlation(counts.astype(float)),
            "morans_i": morans_i(history.spliced[-1][:, scene.target_gene], history.positions[-1]),
        }
        move, move_rng = build_scene("coupled")
        _, move_hist = simulate(move, move_rng)
        ids0 = {int(i): k for k, i in enumerate(move_hist.cell_ids[0])}
        start, end, expr = [], [], []
        for index, cell_id in enumerate(move_hist.cell_ids[-1]):
            if int(cell_id) not in ids0:
                continue
            start.append(move_hist.positions[0][ids0[int(cell_id)]])
            end.append(move_hist.positions[-1][index])
            expr.append(move_hist.spliced[-1][index])
        values["displacement_to_noise"] = displacement_to_noise(np.stack(start), np.stack(end), np.stack(expr))
        for key, value in values.items():
            self.assertTrue(np.isfinite(value), key)
        # The acceptance decision is recorded, including failures.
        self.results = {key: (value, in_range(key, value)) for key, value in values.items()}


if __name__ == "__main__":
    unittest.main()
