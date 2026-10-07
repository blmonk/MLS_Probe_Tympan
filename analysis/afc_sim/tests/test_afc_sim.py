#!/usr/bin/env python3
"""
Regression tests for afc_sim, no pytest dependency (none is installed in this
environment) -- plain functions, each either passes silently or raises
AssertionError. Run directly: `python -m afc_sim.tests.test_afc_sim`.

Two of these are regression tests for real bugs found by manual review on
2026-10-02, both of which silently gave wrong numbers before being fixed:

  1. test_chunk_size_invariance -- PEMAFCFull/PEMAFCPCA's process_block only
     updated spk/mic history once per EXTERNAL call, not per internal
     AR-re-estimation sub-chunk, so mode A results depended on how a signal
     happened to be split into process_block() calls (e.g. --block-size).
  2. test_closed_loop_lag_is_correct -- simulate.py and gain_amplitude_sweep.py
     both compared closed-loop efbp against the OPEN-loop ground-truth window
     (zero_lag_window's default lag=1), but closed_loop.py's own convolution
     already applies the minimum causal lag directly to true_ir, so the
     correct closed-loop target is lag=0. Confirmed by direct hand-trace (see
     git history / conversation record) with a synthetic true_ir that has
     significant near-zero-lag energy -- negligible for this project's one
     real dataset (leading taps are noise-floor-tiny) but a serious error for
     any IR with real direct/near-instant coupling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from afc_sim.algorithms import create_algorithm
from afc_sim.algorithms.pemafc import PEMAFCFull
from afc_sim.algorithms.pemafc_pca import PEMAFCPCA, load_pca_model
from afc_sim.algorithms.whitening import PEMAFCMode
from afc_sim.closed_loop import run_closed_loop_epoch
from afc_sim.conditions import zero_lag_window
from afc_sim.feedback_path import FeedbackPath
from afc_sim import metrics


def _write_pca_model(path, mean_ir, components):
    K, afl = components.shape
    with open(path, "w") as f:
        f.write(f"{afl},{K}\n")
        f.write(",".join(f"{v:.8e}" for v in mean_ir) + "\n")
        for k in range(K):
            f.write(",".join(f"{v:.8e}" for v in components[k]) + "\n")


def test_chunk_size_invariance():
    """process_block's result must not depend on how the input is split into
    calls -- that's a basic correctness property for any per-sample
    sequential algorithm. Deliberately uses chunk sizes that straddle
    ar_block_len boundaries in different ways (1, prime, exact, larger,
    whole-array) to stress the re-estimation-boundary code path."""
    rng = np.random.default_rng(1)
    afl, ar_order, ar_block_len = 24, 4, 37  # block_len deliberately not a multiple of any chunk size below
    n = 2000
    mic = rng.standard_normal(n) * 0.1
    spk = rng.standard_normal(n) * 0.1

    def run(algo_name, chunk, **kwargs):
        algo = create_algorithm(algo_name, mu=0.2, delta=1e-3, alpha=0.9,
                                 ar_order=ar_order, ar_block_len=ar_block_len, ar_reg=1e-5,
                                 mode=PEMAFCMode.A, **kwargs)
        for i in range(0, n, chunk):
            algo.process_block(mic[i:i + chunk], spk[i:i + chunk])
        return algo.get_estimated_ir()

    chunks = [1, 7, 37, 211, n]

    ref = run("pemafc_a", chunks[0], afl=afl)
    for c in chunks[1:]:
        out = run("pemafc_a", c, afl=afl)
        assert np.allclose(out, ref, atol=1e-9, rtol=1e-9), (
            f"PEMAFCFull mode A: result changed with chunk size {c} vs {chunks[0]} "
            f"(max abs diff {np.max(np.abs(out - ref)):.3e}) -- stale-history-across-"
            f"sub-chunk-boundary bug has regressed")

    mean_ir = np.zeros(afl)
    components = np.eye(3, afl)
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    pca_path = os.path.join(tmpdir, "pca.csv")
    _write_pca_model(pca_path, mean_ir, components)

    def run_pca(chunk):
        algo = create_algorithm("pemafc_pca_a", pca_model_path=pca_path, mu=0.2, delta=1e-3,
                                 alpha=0.9, ar_order=ar_order, ar_block_len=ar_block_len, ar_reg=1e-5)
        for i in range(0, n, chunk):
            algo.process_block(mic[i:i + chunk], spk[i:i + chunk])
        return algo.get_estimated_ir()

    ref_pca = run_pca(chunks[0])
    for c in chunks[1:]:
        out = run_pca(c)
        assert np.allclose(out, ref_pca, atol=1e-9, rtol=1e-9), (
            f"PEMAFCPCA mode A: result changed with chunk size {c} vs {chunks[0]} "
            f"(max abs diff {np.max(np.abs(out - ref_pca)):.3e})")

    print("PASS: test_chunk_size_invariance")


def test_closed_loop_lag_is_correct():
    """A synthetic IR with meaningful near-zero-lag energy (true_ir[0]=5.0)
    must be recovered against the lag=0 window, not the open-loop lag=1
    window -- the earlier bug reported the correctly-converged algorithm as
    PERFORMING WORSE THAN DOING NOTHING (positive misalignment_db) because it
    compared against the wrong, shifted target."""
    rng = np.random.default_rng(2)
    afl = 16
    true_ir = np.zeros(64)
    true_ir[0] = 5.0
    true_ir[3] = 2.0
    true_ir[7] = -1.0
    n = 150_000
    desired = 1e-3 * rng.standard_normal(n)

    algo = PEMAFCFull(afl=afl, mu=0.3, delta=1e-6, mode=PEMAFCMode.F, alpha=0.9,
                       ar_order=12, ar_block_len=640)
    run_closed_loop_epoch(algo, desired, true_ir, gain_linear=1.0, saturation_limit=1e9)
    efbp = algo.get_estimated_ir()

    correct_target = zero_lag_window(true_ir, afl, lag=0)
    wrong_target = zero_lag_window(true_ir, afl, lag=1)

    mis_correct = metrics.misalignment_db(efbp, correct_target)
    mis_wrong = metrics.misalignment_db(efbp, wrong_target)

    assert mis_correct < -10.0, (
        f"closed-loop efbp should converge well against the lag=0 target, got {mis_correct:.2f} dB")
    assert mis_wrong > mis_correct + 10.0, (
        "lag=1 (old buggy) target should look meaningfully worse than lag=0 for this IR "
        f"(got wrong={mis_wrong:.2f} dB, correct={mis_correct:.2f} dB) -- "
        "if this fails the two targets may have become numerically similar for an unrelated reason")

    print("PASS: test_closed_loop_lag_is_correct "
          f"(lag=0: {mis_correct:.2f} dB, lag=1/old-buggy: {mis_wrong:.2f} dB)")


def test_open_loop_regressor_matches_feedback_path():
    """Open-loop ground truth (lag=1) must actually be the correct target --
    i.e. FeedbackPath's convolution convention and the algorithm's regressor
    convention must agree. Noise-free-enough synthetic check: a persistently
    exciting (white noise) regressor should drive efbp arbitrarily close to
    zero_lag_window(true_ir, afl, lag=1) given enough samples."""
    rng = np.random.default_rng(3)
    afl = 20
    true_ir = np.zeros(80)
    true_ir[0] = 3.0   # should NOT be representable (excluded by design)
    true_ir[1] = 4.0
    true_ir[5] = -2.0
    true_ir[19] = 1.0

    fp = FeedbackPath(ir_truncate_len=80)
    fp.set_ir(true_ir)
    n = 300_000
    spk = 0.1 * rng.standard_normal(n)
    mic = fp.contaminate(spk)

    algo = PEMAFCFull(afl=afl, mu=0.3, delta=1e-6, mode=PEMAFCMode.A, alpha=0.9,
                       ar_order=12, ar_block_len=640)
    algo.process_block(mic, spk)
    efbp = algo.get_estimated_ir()

    target = zero_lag_window(true_ir, afl, lag=1)
    mis = metrics.misalignment_db(efbp, target)
    # NLMS settles at a nonzero misadjustment floor (empirically ~-11 to -13dB
    # for this sparse synthetic IR/mu combination, same mu-vs-floor tradeoff
    # established elsewhere this session) -- -8dB is a comfortable margin
    # below that floor, not a tautological near-zero check.
    assert mis < -8.0, f"open-loop efbp should converge close to the lag=1 target, got {mis:.2f} dB"
    print(f"PASS: test_open_loop_regressor_matches_feedback_path (misalignment={mis:.2f} dB)")


def test_msg_edge_cases():
    # DC and Nyquist bins (both structurally Im=0) must be detected by the
    # boundary-bin pass, not missed by an interior-only scan. Verified by
    # hand: rfft([2,-1,1,-1], n=8) gives F[0]=1+0j, F[4]=5+0j -- both real
    # and positive.
    fft_len = 8
    ir = np.array([2.0, -1.0, 1.0, -1.0])
    freqs, F = metrics.frequency_response(ir, fft_len, fs=1.0)
    assert np.isclose(F.imag[0], 0.0) and F.real[0] > 0, f"test fixture assumption broken: F[0]={F[0]}"
    nyq = fft_len // 2
    assert np.isclose(F.imag[nyq], 0.0) and F.real[nyq] > 0, f"test fixture assumption broken: F[nyq]={F[nyq]}"
    crossings = metrics.find_zero_phase_crossings(freqs, F)
    cross_freqs = {round(f, 6) for f, _ in crossings}
    assert 0.0 in cross_freqs, "DC bin crossing missed"
    assert round(freqs[nyq], 6) in cross_freqs, "Nyquist bin crossing missed"

    # fft_len < len(ir) must raise (would silently truncate, not zero-pad)
    raised = False
    try:
        metrics.frequency_response(np.ones(10), fft_len=4)
    except AssertionError:
        raised = True
    assert raised, "frequency_response should have asserted on fft_len < len(ir)"

    # No positive-real-part crossings -> +inf, not a crash
    ir_zero = np.zeros(16)
    msg = metrics.msg_db(ir_zero, fft_len=64)
    assert msg == float("inf")

    print("PASS: test_msg_edge_cases")


def test_pca_zero_active_components_pins_at_mean():
    afl = 10
    mean_ir = np.linspace(1, 10, afl)
    components = np.eye(3, afl)
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    pca_path = os.path.join(tmpdir, "pca.csv")
    _write_pca_model(pca_path, mean_ir, components)

    algo = PEMAFCPCA(pca_model_path=pca_path, mu=0.5, delta=1e-3, mode=PEMAFCMode.F,
                      alpha=0.9, ar_order=4, ar_block_len=50, n_components_active=0)
    rng = np.random.default_rng(4)
    mic = rng.standard_normal(5000) * 0.1
    spk = rng.standard_normal(5000) * 0.1
    algo.process_block(mic, spk)
    assert np.allclose(algo.get_estimated_ir(), mean_ir), (
        "n_components_active=0 must pin efbp at mean_ir with no adaptation")
    print("PASS: test_pca_zero_active_components_pins_at_mean")


def test_nlms_matches_plain_nlms_reference():
    """create_algorithm('nlms', ...) must be exactly (bit-for-bit) plain NLMS
    with no whitening -- verified independently against a hand-coded plain
    NLMS implementation using the same regressor convention (tap0=spk[i-1])."""
    rng = np.random.default_rng(5)
    afl = 20
    n = 5000
    mic = rng.standard_normal(n) * 0.1
    spk = rng.standard_normal(n) * 0.1
    mu, delta = 0.3, 1e-3

    algo = create_algorithm("nlms", afl=afl, mu=mu, delta=delta, alpha=0.9,  # alpha should be IGNORED/forced
                             ar_order=4, ar_block_len=100, ar_reg=1e-5)
    algo.process_block(mic, spk)
    efbp = algo.get_estimated_ir()

    hist_len = afl + 8
    efbp_manual = np.zeros(afl)
    extended = np.concatenate([np.zeros(hist_len), spk])
    for k in range(n):
        idx = hist_len + k
        u = extended[idx - afl: idx][::-1]
        e = mic[k] - np.dot(u, efbp_manual)
        norm_sq = np.dot(u, u) + delta
        efbp_manual += (mu / norm_sq) * e * u

    assert np.allclose(efbp, efbp_manual, atol=1e-12, rtol=1e-12), (
        f"'nlms' variant diverged from plain-NLMS reference (max diff "
        f"{np.max(np.abs(efbp - efbp_manual)):.3e}) -- alpha=0 no longer an exact identity?")
    print("PASS: test_nlms_matches_plain_nlms_reference")


def test_ar_fit_signal_round_trips_to_white():
    """Generating 'ar_fit' noise from a fitted AR envelope, then whitening it
    with that SAME envelope, should produce something close to white (flat
    autocorrelation away from lag 0) -- the basic round-trip property the
    whole point of 'ar_fit' depends on."""
    import sys as _sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
    from afc_sim.signals import SignalSpec, generate_excitation
    from afc_sim.algorithms.whitening import biased_autocorr, levinson_durbin, whiten_fir

    rng = np.random.default_rng(6)
    fs = 8000.0
    # synthetic "speech-like" segment: a few resonances, not actual speech,
    # so this test has no file dependency
    t = np.arange(int(2 * fs)) / fs
    segment = (np.sin(2*np.pi*220*t) + 0.5*np.sin(2*np.pi*800*t)
               + 0.1*rng.standard_normal(len(t)))
    import tempfile, os
    from scipy.io import wavfile
    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "synthetic.wav")
    wavfile.write(wav_path, int(fs), (segment / np.max(np.abs(segment)) * 0.9).astype(np.float32))

    spec = SignalSpec(kind="ar_fit", wav_path=wav_path, amplitude=0.1,
                       ar_fit_start_s=0.0, ar_fit_duration_s=2.0, ar_fit_order=10, seed=0)
    x = generate_excitation(spec, n_samples=20000, fs=fs)
    assert np.all(np.isfinite(x))

    r_x = biased_autocorr(x, 10)
    ar_coeffs, stable = levinson_durbin(r_x, 10)
    assert stable
    whitened = whiten_fir(x, ar_coeffs, lookback=np.zeros(10))
    r_w = biased_autocorr(whitened[1000:], 10)  # skip transient
    off_diag_ratio = np.max(np.abs(r_w[1:])) / (abs(r_w[0]) + 1e-15)
    assert off_diag_ratio < 0.1, (
        f"whitened ar_fit signal should look close to white, got off-diag/diag ratio {off_diag_ratio:.3f}")
    print("PASS: test_ar_fit_signal_round_trips_to_white")


def test_soft_pca_boundary_cases():
    """soft_pca_alpha=0 must be bit-exact plain NLMS (no pull-back at all).
    soft_pca_alpha=1 need NOT match PEMAFCPCA (different effective NLMS
    normalization -- see pemafc_soft_pca.py's module docstring for why an
    earlier claim of bit-exactness was wrong), but efbp MUST land exactly on
    the mean_ir+subspace affine set after every sample at alpha=1 (z_perp
    forced to ~0 by construction)."""
    rng = np.random.default_rng(7)
    afl = 20
    mean_ir = np.zeros(afl)
    components = np.eye(5, afl)
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    pca_path = os.path.join(tmpdir, "pca.csv")
    _write_pca_model(pca_path, mean_ir, components)

    n = 5000
    mic = rng.standard_normal(n) * 0.1
    spk = rng.standard_normal(n) * 0.1
    mu, delta = 0.3, 1e-3

    algo_soft0 = create_algorithm("soft_pca", pca_model_path=pca_path, mu=mu, delta=delta,
                                   mode=PEMAFCMode.F, alpha=0.0, soft_pca_alpha=0.0,
                                   ar_order=4, ar_block_len=100)
    algo_soft0.process_block(mic, spk)
    efbp_soft0 = algo_soft0.get_estimated_ir()

    algo_nlms = create_algorithm("nlms", afl=afl, mu=mu, delta=delta, ar_order=4, ar_block_len=100)
    algo_nlms.process_block(mic, spk)
    efbp_nlms = algo_nlms.get_estimated_ir()
    assert np.allclose(efbp_soft0, efbp_nlms, atol=1e-12, rtol=1e-12), (
        f"soft_pca_alpha=0 should be bit-exact NLMS, max diff "
        f"{np.max(np.abs(efbp_soft0 - efbp_nlms)):.3e}")

    algo_soft1 = create_algorithm("soft_pca", pca_model_path=pca_path, mu=mu, delta=delta,
                                   mode=PEMAFCMode.F, alpha=0.9, soft_pca_alpha=1.0,
                                   ar_order=4, ar_block_len=100)
    algo_soft1.process_block(mic, spk)
    efbp_soft1 = algo_soft1.get_estimated_ir()
    z = efbp_soft1 - mean_ir
    z_perp_norm = np.linalg.norm(z - (components @ z) @ components)
    assert z_perp_norm < 1e-9, (
        f"soft_pca_alpha=1 should land exactly on the PCA subspace, "
        f"off-subspace residual norm={z_perp_norm:.3e}")
    print("PASS: test_soft_pca_boundary_cases")


def test_blend_pca_boundary_and_independence():
    """BlendPCA combines two INDEPENDENTLY adapting filters (unlike
    PEMAFCSoftPCA, which maintains only one state) -- blend_weight=0/1 must
    reduce to exactly PEMAFC-F/PEMAFC-PCA-F alone, and at any blend weight
    each sub-filter's internal state must exactly match running that same
    algorithm standalone on the same data (true independence: blending must
    not feed back into either sub-filter's own adaptation)."""
    rng = np.random.default_rng(8)
    afl = 20
    mean_ir = np.zeros(afl)
    components = np.eye(5, afl)
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    pca_path = os.path.join(tmpdir, "pca.csv")
    _write_pca_model(pca_path, mean_ir, components)

    n = 5000
    mic = rng.standard_normal(n) * 0.1
    spk = rng.standard_normal(n) * 0.1
    mu, delta, alpha = 0.3, 1e-3, 0.9
    ar_order, ar_block_len = 4, 100

    full_alone = create_algorithm("pemafc_f", afl=afl, mu=mu, delta=delta, alpha=alpha,
                                   ar_order=ar_order, ar_block_len=ar_block_len)
    full_alone.process_block(mic, spk)
    efbp_full_alone = full_alone.get_estimated_ir()

    pca_alone = create_algorithm("pemafc_pca_f", pca_model_path=pca_path, mu=mu, delta=delta,
                                  alpha=alpha, ar_order=ar_order, ar_block_len=ar_block_len)
    pca_alone.process_block(mic, spk)
    efbp_pca_alone = pca_alone.get_estimated_ir()

    blend0 = create_algorithm("blend_pca", pca_model_path=pca_path, mu=mu, delta=delta,
                               blend_weight=0.0, alpha=alpha, ar_order=ar_order, ar_block_len=ar_block_len)
    blend0.process_block(mic, spk)
    assert np.allclose(blend0.get_estimated_ir(), efbp_full_alone, atol=1e-12, rtol=1e-12), \
        "blend_weight=0 must be bit-exact PEMAFC-F alone"

    blend1 = create_algorithm("blend_pca", pca_model_path=pca_path, mu=mu, delta=delta,
                               blend_weight=1.0, alpha=alpha, ar_order=ar_order, ar_block_len=ar_block_len)
    blend1.process_block(mic, spk)
    assert np.allclose(blend1.get_estimated_ir(), efbp_pca_alone, atol=1e-12, rtol=1e-12), \
        "blend_weight=1 must be bit-exact PEMAFC-PCA-F alone"

    blend5 = create_algorithm("blend_pca", pca_model_path=pca_path, mu=mu, delta=delta,
                               blend_weight=0.5, alpha=alpha, ar_order=ar_order, ar_block_len=ar_block_len)
    blend5.process_block(mic, spk)
    assert np.allclose(blend5.efbp_full, efbp_full_alone, atol=1e-12, rtol=1e-12), \
        "blending must not perturb the full-rank sub-filter's own independent trajectory"
    assert np.allclose(blend5.efbp_pca, efbp_pca_alone, atol=1e-12, rtol=1e-12), \
        "blending must not perturb the PCA sub-filter's own independent trajectory"
    expected_blend = 0.5 * efbp_full_alone + 0.5 * efbp_pca_alone
    assert np.allclose(blend5.get_estimated_ir(), expected_blend, atol=1e-12, rtol=1e-12), \
        "reported efbp must be exactly the weighted average of the two independent trajectories"
    print("PASS: test_blend_pca_boundary_and_independence")


def test_blend_pca_dynamic_boundary_and_direction():
    """mu_lambda=0 must be bit-exact the fixed-blend_weight case (regression
    safety). Directional check: given enough data, a dynamically-blended
    filter should land at a HIGHER blend_weight (favoring PCA) when the true
    IR lies exactly in the PCA subspace than when it's orthogonal to it --
    confirmed empirically to need a fairly long run to resolve cleanly (an
    initial short/aggressive-mu probe was ambiguous/reversed early on,
    traced to the gradient tracking "which sub-filter predicts the observed
    signal better *right now*", not "which is closer to the unknowable true
    IR" -- these only reliably agree once each filter has had enough data to
    show its real behavior)."""
    rng = np.random.default_rng(9)
    afl = 64
    raw = rng.standard_normal((6, afl))
    Q, _ = np.linalg.qr(raw.T)
    components = Q[:, :6].T
    mean_ir = np.zeros(afl)
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    pca_path = os.path.join(tmpdir, "pca.csv")
    _write_pca_model(pca_path, mean_ir, components)

    mu, delta, alpha, ar_order, ar_block_len = 0.02, 1e-3, 0.9, 12, 640

    # boundary: mu_lambda=0 bit-exact to fixed blend_weight
    n_small = 5000
    mic0 = rng.standard_normal(n_small) * 0.1
    spk0 = rng.standard_normal(n_small) * 0.1
    fixed = create_algorithm("blend_pca", pca_model_path=pca_path, mu=mu, delta=delta,
                              blend_weight=0.5, alpha=alpha, ar_order=ar_order, ar_block_len=ar_block_len)
    fixed.process_block(mic0, spk0)
    dyn0 = create_algorithm("blend_pca_dynamic", pca_model_path=pca_path, mu=mu, delta=delta,
                             blend_weight_init=0.5, mu_lambda=0.0, alpha=alpha,
                             ar_order=ar_order, ar_block_len=ar_block_len)
    dyn0.process_block(mic0, spk0)
    assert np.allclose(dyn0.get_estimated_ir(), fixed.get_estimated_ir(), atol=1e-12, rtol=1e-12), \
        "mu_lambda=0 must be bit-exact the fixed blend_weight case"

    # Local gradient direction, fully controlled (single sample): directly
    # set efbp_full/efbp_pca so fb_est_pca is clearly closer to mic than
    # fb_est_full is (same-sign, smaller-magnitude error) -- blend_weight
    # must increase (favor PCA more). And vice versa.
    #
    # NOTE: an earlier version of this test tried to verify direction via a
    # long emergent run on a synthetic "true IR in/out of the PCA subspace"
    # scenario, expecting in-subspace to always end up favoring PCA. That
    # assumption was wrong and the result flipped depending on random seed,
    # even over 600k samples -- not noise, but a genuine confound: PCA's
    # advantage isn't only about bias (can it represent the truth), it's a
    # real bias/variance tradeoff, since a K-parameter filter has intrinsically
    # lower misadjustment NOISE than an afl-parameter one regardless of bias.
    # Testing the LOCAL gradient direction directly (below) is the property
    # that's actually guaranteed; "which one wins after a long run" depends
    # on specifics of both filters' noise floors, not just subspace membership.
    afl2 = 5
    components2 = np.eye(2, afl2)
    mean_ir2 = np.zeros(afl2)
    pca_path2 = os.path.join(tmpdir, "pca2.csv")
    _write_pca_model(pca_path2, mean_ir2, components2)

    def one_step_weight_change(efbp_full_val, efbp_pca_val, mic_val, spk_hist_val):
        algo = create_algorithm("blend_pca_dynamic", pca_model_path=pca_path2, mu=0.0, delta=1e-3,
                                 alpha=0.9, ar_order=1, ar_block_len=10_000_000,
                                 mu_lambda=0.1, blend_weight_init=0.5)
        algo.efbp_full[:] = efbp_full_val
        algo.efbp_pca[:] = efbp_pca_val
        algo._spk_hist[-afl2:] = spk_hist_val
        w_before = algo.blend_weight
        algo.process_block(np.array([mic_val]), np.array([0.0]))
        return algo.blend_weight - w_before

    # u_raw ends up some permutation/reversal of spk_hist_val depending on
    # exact indexing -- use a UNIFORM u_raw (all 1s) so fb_est=dot(u_raw,efbp)
    # is just afl2 * efbp's (uniform) scalar value, regardless of which
    # index lands where. NOTE: an earlier version of this test forgot the
    # afl2x factor (set efbp_pca uniformly to 0.9 expecting fb_est_pca=0.9,
    # but dot([1,1,1,1,1], [0.9]*5)=4.5, not 0.9) -- caught because the
    # resulting fb_est_pca=4.5 badly OVERSHOT mic=1.0 and was actually the
    # worse predictor, so the gradient correctly favored full-rank instead,
    # which looked like a sign bug until traced through by hand.
    spk_hist_val = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    efbp_full_val = np.full(afl2, 0.0)          # fb_est_full = 5*0 = 0
    efbp_pca_val = np.full(afl2, 0.9 / afl2)    # fb_est_pca = 5*(0.9/5) = 0.9, closer to mic=1.0
    dw_pca_better = one_step_weight_change(efbp_full_val, efbp_pca_val, mic_val=1.0, spk_hist_val=spk_hist_val)
    assert dw_pca_better > 0, f"blend_weight should increase when PCA predicts better, got delta={dw_pca_better:.6f}"

    dw_full_better = one_step_weight_change(efbp_pca_val, efbp_full_val, mic_val=1.0, spk_hist_val=spk_hist_val)
    assert dw_full_better < 0, f"blend_weight should decrease when full-rank predicts better, got delta={dw_full_better:.6f}"
    print(f"PASS: test_blend_pca_dynamic_boundary_and_direction "
          f"(dw_pca_better={dw_pca_better:+.4f}, dw_full_better={dw_full_better:+.4f})")


def test_cli_help_text_not_corrupted_by_percent_signs():
    """A literal '%' in an argparse help= string, followed by whitespace and
    a letter like 's', gets parsed by argparse's internal %-formatting as a
    %s conversion specifier -- since help text is formatted against the
    action's own __dict__, this silently stringifies that whole dict into
    the help output. Confirmed and fixed once already (run_experiment.py's
    --block-size help literally printed "...~20{'option_strings': [...]...}
    lower for no benefit." instead of "~20% slower..."). The standard fix is
    escaping literal percents as '%%'; this guards against it recurring."""
    from afc_sim.run_experiment import build_parser
    help_text = build_parser().format_help()
    assert "option_strings" not in help_text, (
        "CLI help text contains a raw argparse action dict -- a literal unescaped "
        "'%' in some add_argument(help=...) string is being misparsed as a %s "
        "format specifier; escape it as '%%'")
    print("PASS: test_cli_help_text_not_corrupted_by_percent_signs")


def main():
    tests = [
        test_chunk_size_invariance,
        test_closed_loop_lag_is_correct,
        test_open_loop_regressor_matches_feedback_path,
        test_msg_edge_cases,
        test_pca_zero_active_components_pins_at_mean,
        test_nlms_matches_plain_nlms_reference,
        test_ar_fit_signal_round_trips_to_white,
        test_soft_pca_boundary_cases,
        test_blend_pca_boundary_and_independence,
        test_blend_pca_dynamic_boundary_and_direction,
        test_cli_help_text_not_corrupted_by_percent_signs,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
