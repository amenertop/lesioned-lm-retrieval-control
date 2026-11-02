"""General degradation model (lesion.degradation) and the experiment 3 statistics built on it (lesion.e3_analysis)."""
import numpy as np

from lesion import degradation as dg, e3_analysis as ea


def data(n=40, k=5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 6, (n, k))
    z[:, 0] += 15                                     # the target leads, as in the intact model
    prior = dg.frequency_prior(rng.uniform(2, 6, (n, k)))
    return z, prior


def test_prediction_limits():
    z, prior = data()
    assert np.allclose(dg.log_predict(z, prior, 1.0, 0.0), dg.log_softmax(z))
    assert np.allclose(dg.log_predict(z, prior, 3.0, 1.0), np.log(prior))
    p = np.exp(dg.log_predict(z, prior, 2.0, 0.3))
    assert np.allclose(p.sum(axis=1), 1)


def test_fit_recovers_parameters_on_the_grid():
    z, prior = data()
    T, lam = dg.T_GRID[23], 0.27
    p_obs = np.exp(dg.log_predict(z, prior, T, lam))
    f = dg.fit(p_obs, z, prior)
    assert abs(f["T"] - T) < 1e-12 and f["lam"] == lam
    assert f["near_optimal"]["n_points"] >= 1 and len(f["profile_T"]) == len(dg.T_GRID)


def test_bootstrap_fit_with_every_item_once_is_the_point_fit():
    z, prior = data()
    p_obs = np.exp(dg.log_predict(z, prior, dg.T_GRID[10], 0.1))
    ce = dg.item_cross_entropy_grid(p_obs, z, prior)
    idx = dg.bootstrap_fits(ce, np.ones((3, len(z))))
    assert (idx == [10, 10]).all()


def test_statistics():
    logp = np.log(np.array([[0.5, 0.1, 0.2, 0.15, 0.05]]))    # target, coordinate, superordinate, associate, unrelated
    assert abs(ea.s_stat(logp)[0] - (0.15 - 0.1 - 0.2)) < 1e-12
    lp = np.zeros((1, 3, 2))
    lp[0, :, 0] = [-1.0, -0.5, -3.0]                          # target under none, cue, miscue
    lp[0, :, 1] = [-4.0, -6.0, -1.0]                          # associate
    assert ea.r_stat(lp)[0] == (-0.5 + 6.0) - (-3.0 + 1.0)


def test_a_lesion_that_is_pure_degradation_shows_no_excess():
    """K follows the degradation model exactly, C changes nothing: D_obs is all degradation, so excess is 0."""
    zc, pc = data(40, seed=1)
    zt, pt = data(60, seed=2)
    T, lam = dg.T_GRID[30], 0.2
    k_cal, k_test = dg.log_predict(zc, pc, T, lam), dg.log_predict(zt, pt, T, lam)
    ce = {"K": dg.item_cross_entropy_grid(np.exp(k_cal), zc, pc),
          "C": dg.item_cross_entropy_grid(np.exp(dg.log_softmax(zc)), zc, pc)}
    obs = {"intact": ea.s_stat(zt), "K": ea.s_stat(k_test), "C": ea.s_stat(zt)}
    pred = {L: ea.h3a_predicted_grid(zt, pt) for L in ("C", "K")}
    out = ea.excess_test(obs, pred, ce, n_boot=200)
    assert out["fits"]["K"]["T"] == T and out["fits"]["K"]["lam"] == lam
    assert out["fits"]["C"]["T"] == 1.0 and out["fits"]["C"]["lam"] == 0.0
    assert abs(out["excess"]) < 1e-9 and abs(out["D_obs"] - out["D_pred"]) < 1e-9
    assert not out["supported"]


def test_h3b_grid_matches_direct_prediction():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 3, (5, 3, 2))
    prior = dg.frequency_prior(rng.uniform(2, 6, (5, 2)))
    grid = ea.h3b_predicted_grid(z, prior)
    i, j = 7, 40
    direct = ea.r_stat(dg.log_predict(z.reshape(15, 2), np.repeat(prior, 3, axis=0), dg.T_GRID[i], dg.LAMBDA_GRID[j]).reshape(5, 3, 2))
    assert np.allclose(grid[:, i * len(dg.LAMBDA_GRID) + j], direct)
    ce = ea.h3b_item_ce_grid(z, z, prior)                   # lesioned == intact: best fit is T = 1, lam = 0
    assert dg.best(ce.sum(axis=0))[:2] == (1.0, 0.0)
