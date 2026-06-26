from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from config import APP_VERSION, DEFAULT_COLUMNS, DEFAULT_CSV, RNG_SEED
from calibration import calibrated_randomness_fingerprint
from data import load_draws
from data_quality import validate_draw_history
from lotteries import LOTTERY
from quantum_profiles import LOCAL_QC25_QUBITS, resolve_quantum_profile
from simulator import run_profiled_sampling
from math_model import (
    backtest_summary,
    hit_distribution,
    number_scores,
    optimize_tickets,
    optimize_tickets_with_metadata,
    ticket_set_metrics,
)
from randomness import audit_pool_randomness, score_vector, walk_forward_models
from validation import nested_ticket_backtest
from qcbm_config import DEFAULT_QCBM_MODEL, load_config as load_qcbm_config
from qcbm_train import blend_weights, qcbm_probability_vector, run_training_pipeline as run_qcbm_training
from copula_config import DEFAULT_COPULA_MODEL, load_config as load_copula_config
from copula_train import load_model as load_copula_model, run_training_pipeline as run_copula_training


def prompt(value: str | None, label: str) -> str:
    if value:
        return value
    answer = input(f"{label}: ").strip()
    if not answer:
        raise SystemExit(f"{label} je obavezno.")
    return answer


def _resolve_seed_weights(
    classical_scores: list[float],
    use_qcbm: bool,
) -> tuple[list[float], dict | None]:
    if not use_qcbm:
        return classical_scores, None
    qcbm_probs = qcbm_probability_vector()
    if qcbm_probs is None:
        return classical_scores, None
    cfg = load_qcbm_config()
    alpha = float(cfg.get("blend", {}).get("qcbm_weight", 0.5))
    blended = blend_weights(np.asarray(classical_scores), qcbm_probs, qcbm_weight=alpha)
    from qcbm_combo import DEFAULT_QCBM_COMBO_MODEL, load_combo_model

    combo_loaded = load_combo_model() is not None
    return blended, {
        "qcbm_model": str(DEFAULT_QCBM_COMBO_MODEL if combo_loaded else DEFAULT_QCBM_MODEL),
        "qcbm_type": "combo_qc25" if combo_loaded else "marginal_pytorch",
        "qcbm_blend": alpha,
        "qcbm_loaded": True,
    }


def _resolve_copula(use_copula: bool):
    if not use_copula:
        return None, None
    model = load_copula_model()
    if model is None:
        return None, None
    cfg = load_copula_config()
    weight = float(cfg.get("blend", {}).get("objective_weight", 2.0))
    meta = {
        "copula_model": str(DEFAULT_COPULA_MODEL),
        "copula_weight": weight,
        "copula_loaded": True,
        "unique_combos": len(model.combo_counts),
        "n_draws": model.n_draws,
    }
    return model, meta


def _print_next_step(*, after: str) -> None:
    from qcbm_combo import DEFAULT_QCBM_COMBO_MODEL

    has_qcbm = DEFAULT_QCBM_COMBO_MODEL.is_file()
    has_copula = DEFAULT_COPULA_MODEL.is_file()

    if after in ("qcbm", "copula", "polazna"):
        if has_qcbm and has_copula:
            print("Sledeći korak: python cli.py audit --date GGGG-MM-DD")
            return
        if not has_copula:
            print("Sledeći korak: python cli.py train-copula --seed 39")
            return
        print("Sledeći korak: python cli.py train-qcbm --seed 39")


def _run_quantum_layer(
    args: argparse.Namespace,
    weights: list[float],
    output_path: str | None,
    qcbm_meta: dict | None = None,
) -> tuple[list[int], dict]:
    output_counts = Path(output_path).with_suffix(".counts.json") if output_path else None
    profile = resolve_quantum_profile(args.quantum_profile, LOCAL_QC25_QUBITS)
    seed_bits, quantum_job = run_profiled_sampling(
        qubits=args.qubits or profile["qubits"],
        layers=args.layers or profile["layers"],
        batch_circuits=args.batch_circuits or profile["batch_circuits"],
        shots=args.shots or profile["shots"],
        seed_weights=weights,
        output_counts=output_counts,
        repeat_jobs=args.repeat_jobs or profile["repeat_jobs"],
        profile=args.quantum_profile,
        csv_path=args.csv,
        seed=args.seed,
    )
    if qcbm_meta:
        quantum_job = {**quantum_job, "qcbm": qcbm_meta}
    return seed_bits, quantum_job


def cmd_train_qcbm(args: argparse.Namespace) -> None:
    spec = LOTTERY
    draws = load_draws(spec, args.csv)
    if len(draws) < 30:
        raise SystemExit(f"Potrebno je bar 30 izvlačenja. Učitano: {len(draws)}.")

    print(f"\n=== QCBM trening (pozicijski combo) — {spec.name} ({APP_VERSION}) ===")
    print(f"Izvlačenja: {len(draws)} | CSV: {args.csv} | seed: {args.seed}")

    result = run_qcbm_training(
        draws,
        spec.main,
        seed=args.seed,
        model_path=Path(args.model) if args.model else None,
        config_path=Path(args.config) if args.config else None,
        csv_path=args.csv,
    )

    print("\n=== Završeno ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"\nModel: {result['model_path']}")
    _print_next_step(after="qcbm")


def cmd_train_copula(args: argparse.Namespace) -> None:
    spec = LOTTERY
    draws = load_draws(spec, args.csv)
    if len(draws) < 30:
        raise SystemExit(f"Potrebno je bar 30 izvlačenja. Učitano: {len(draws)}.")

    print(f"\n=== Copula trening — {spec.name} ({APP_VERSION}) ===")
    print(f"Izvlačenja: {len(draws)} | CSV: {args.csv} | seed: {args.seed}")

    result = run_copula_training(
        draws,
        seed=args.seed,
        model_path=Path(args.model) if args.model else None,
        config_path=Path(args.config) if args.config else None,
    )

    print("\n=== Završeno ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"\nModel: {result['model_path']}")
    _print_next_step(after="copula")


def cmd_train_polazna(args: argparse.Namespace) -> None:
    """Oba polazna koraka: QCBM + copula (ceo CSV)."""
    spec = LOTTERY
    draws = load_draws(spec, args.csv)
    if len(draws) < 30:
        raise SystemExit(f"Potrebno je bar 30 izvlačenja. Učitano: {len(draws)}.")

    print(f"\n=== Polazni trening (QCBM + Copula) — {spec.name} ({APP_VERSION}) ===")
    print(f"Izvlačenja: {len(draws)} | CSV: {args.csv} | seed: {args.seed}\n")

    qcbm = run_qcbm_training(draws, spec.main, seed=args.seed, csv_path=args.csv)
    print()
    copula = run_copula_training(draws, seed=args.seed)

    print("\n=== Oba koraka završena ===")
    print(f"  QCBM: {qcbm['model_path']}")
    print(f"  Copula: {copula['model_path']}")
    _print_next_step(after="polazna")


def cmd_predict(args: argparse.Namespace) -> None:
    spec = LOTTERY
    target_date = prompt(args.date, "Datum izvlačenja (GGGG-MM-DD)")
    try:
        date.fromisoformat(target_date)
    except ValueError as exc:
        raise SystemExit("--date mora biti GGGG-MM-DD") from exc

    draws = load_draws(spec, args.csv)
    if len(draws) < 30:
        raise SystemExit(f"Potrebno je bar 30 izvlačenja. Učitano: {len(draws)}.")

    main_scores = number_scores(draws, spec.main)
    seed_bits: list[int] | None = None
    quantum_job = None
    qcbm_meta = None
    copula_model = None
    copula_meta = None
    if not args.no_quantum:
        weights, qcbm_meta = _resolve_seed_weights(
            main_scores.tolist(),
            use_qcbm=not args.no_qcbm,
        )
        seed_bits, quantum_job = _run_quantum_layer(args, weights, args.output, qcbm_meta)

    copula_model, copula_meta = _resolve_copula(use_copula=not args.no_copula)
    copula_weight = float((copula_meta or {}).get("copula_weight", 0.0))

    tickets = optimize_tickets(
        spec,
        draws,
        columns=args.columns,
        seed_bits=seed_bits,
        seed=args.seed,
        copula=copula_model,
        copula_weight=copula_weight,
    )
    history = backtest_summary(tickets, draws[-min(len(draws), args.backtest_draws) :])
    set_metrics = ticket_set_metrics(tickets, spec.main)
    baseline = hit_distribution(spec.main.maximum - spec.main.minimum + 1, spec.main.pick, args.columns)

    payload = {
        "warning": "Samo istraživanje / zabava. Izvlačenja su nasumična; nema garancije.",
        "lottery": spec.name,
        "draw_date": target_date,
        "columns": args.columns,
        "tickets": [ticket.as_dict() for ticket in tickets],
        "ticket_set_metrics": set_metrics,
        "history": history,
        "baseline": {
            "random_any_2_plus": baseline["any_2_plus"],
            "random_any_3_plus": baseline["any_3_plus"],
        },
        "source": {
            "draws_loaded": len(draws),
            "first_draw": str(draws[0].date),
            "last_draw": str(draws[-1].date),
            "note": spec.source_note,
        },
        "quantum": quantum_job,
        "qcbm": qcbm_meta,
        "copula": copula_meta,
    }

    text = human_report(payload)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"\nSačuvano: {out}")
        print(f"Sačuvano: {out.with_suffix('.md')}")


def cmd_audit(args: argparse.Namespace) -> None:
    spec = LOTTERY
    target_date = prompt(args.date, "Datum izvlačenja (GGGG-MM-DD)")
    try:
        date.fromisoformat(target_date)
    except ValueError as exc:
        raise SystemExit("--date mora biti GGGG-MM-DD") from exc

    draws = load_draws(spec, args.csv)
    if len(draws) < 30:
        raise SystemExit(f"Potrebno je bar 30 izvlačenja. Učitano: {len(draws)}.")

    quality = validate_draw_history(draws, spec)
    columns = args.columns
    null_trials = args.null_trials if args.null_trials is not None else (2000 if args.deep_calibration else 500)
    randomness = audit_pool_randomness(draws, spec.main, "main")
    fingerprint = calibrated_randomness_fingerprint(
        draws,
        spec.main,
        "main",
        null_trials=null_trials,
        seed=args.seed,
    )
    walk = walk_forward_models(draws, spec, field="main", train_min=args.train_min)
    baseline = hit_distribution(spec.main.maximum - spec.main.minimum + 1, spec.main.pick, columns)

    best_model_scores = score_vector(draws, spec, "main", walk["best_model"])
    seed_bits = None
    quantum_job = None
    qcbm_meta = None
    copula_model = None
    copula_meta = None
    if not args.no_quantum:
        weights, qcbm_meta = _resolve_seed_weights(
            best_model_scores.tolist(),
            use_qcbm=not args.no_qcbm,
        )
        seed_bits, quantum_job = _run_quantum_layer(args, weights, args.output, qcbm_meta)

    copula_model, copula_meta = _resolve_copula(use_copula=not args.no_copula)
    copula_weight = float((copula_meta or {}).get("copula_weight", 0.0))

    tickets, search_report = optimize_tickets_with_metadata(
        spec,
        draws,
        columns=columns,
        seed_bits=seed_bits,
        seed=args.seed,
        score_override=best_model_scores,
        candidate_mode=args.candidate_mode,
        exact_top_k=args.exact_top_k,
        max_exact_combinations=args.max_exact_combinations,
        copula=copula_model,
        copula_weight=copula_weight,
    )
    ticket_backtest = backtest_summary(tickets, draws[-min(len(draws), args.backtest_draws) :])
    nested_backtest = nested_ticket_backtest(
        spec,
        draws,
        columns=columns,
        train_min=args.train_min,
        seed=args.seed,
        candidate_pool=args.nested_candidate_pool,
        max_test_draws=args.nested_test_draws,
    )
    set_metrics = ticket_set_metrics(tickets, spec.main)
    payload = {
        "warning": "Samo istraživanje / zabava. Izvlačenja su nasumična; nema garancije.",
        "right_question": "Da li postoji merljiva nestruktura, i da li preživljava out-of-sample validaciju?",
        "lottery": spec.name,
        "draw_date": target_date,
        "columns": columns,
        "randomness_audit": randomness,
        "randomness_fingerprint": fingerprint,
        "calibration": fingerprint.get("calibration"),
        "data_quality": quality,
        "null_trials": null_trials,
        "walk_forward": walk,
        "selected_generation_model": walk["best_model"],
        "candidate_search": search_report,
        "tickets": [ticket.as_dict() for ticket in tickets],
        "ticket_set_metrics": set_metrics,
        "ticket_backtest": ticket_backtest,
        "nested_ticket_backtest": nested_backtest,
        "baseline": {
            "random_any_2_plus": baseline["any_2_plus"],
            "random_any_3_plus": baseline["any_3_plus"],
        },
        "source": {
            "draws_loaded": len(draws),
            "first_draw": str(draws[0].date),
            "last_draw": str(draws[-1].date),
            "note": spec.source_note,
        },
        "quantum": quantum_job,
        "qcbm": qcbm_meta,
        "copula": copula_meta,
    }
    text = audit_report(payload)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"\nSačuvano: {out}")
        print(f"Sačuvano: {out.with_suffix('.md')}")


def quantum_report_rows(payload: dict) -> list[str]:
    q = payload.get("quantum")
    if not q:
        return []
    rows = [
        "",
        "## Lokalni qc25 simulator",
        "",
        f"- Profil: `{q.get('profile', 'custom')}`",
        f"- Backend: `{q.get('backend', 'aer_simulator')}`",
        f"- Kubiti/slojevi/batch/šutevi: `{q.get('qubits')}` / `{q.get('layers')}` / `{q.get('batch_circuits')}` / `{q.get('shots_per_circuit')}`",
        f"- Ponavljanja posla: `{q.get('repeat_jobs', 1)}`",
        f"- Ukupno šuteva: `{q.get('total_requested_shots', q.get('total_shots', ''))}`",
    ]
    qcbm = q.get("qcbm") or payload.get("qcbm")
    if qcbm and qcbm.get("qcbm_loaded"):
        rows.append(f"- QCBM mešanje: `{qcbm.get('qcbm_blend', 0.5)}` (model: `{qcbm.get('qcbm_model', '')}`)")
    elif qcbm is None and not (payload.get("qcbm") or {}).get("qcbm_loaded"):
        rows.append("- QCBM: nije učitan (samo klasični skorovi za qc25)")
    copula = payload.get("copula")
    if copula and copula.get("copula_loaded"):
        rows.append(
            f"- Copula sedmerke: učitan, težina `{copula.get('copula_weight', 2.0)}` "
            f"({copula.get('unique_combos', '?')} jedinstvenih iz CSV)"
        )
    else:
        rows.append("- Copula: nije učitan")
    if q.get("encode_values"):
        rows.append(f"- Encode vrednosti (poz. 1–5 u kolu, 6–7 izvedene): `{q['encode_values']}`")
    top = q.get("top_combos") or []
    if top:
        rows.append(f"- Top kvantna kombinacija (7): `{top[0]['combo']}` (p={top[0]['probability']:.4f})")
    rows.append(
        "Lokalni Aer simulator (qc25, 5×5 kubita). Kolo ne „razume” loto — audit/backtest je sloj razumevanja."
    )
    return rows


def audit_report(payload: dict) -> str:
    audit = payload["randomness_audit"]
    fingerprint = payload["randomness_fingerprint"]
    walk = payload["walk_forward"]
    best = walk["models"][walk["best_model"]]
    uniform = walk["models"]["uniform"]
    calibration = fingerprint.get("calibration") or {}
    candidate_search = payload.get("candidate_search") or {}
    nested = payload.get("nested_ticket_backtest") or {}
    quality = payload.get("data_quality") or {}
    rows = [
        f"# {payload['lottery']} — audit nasumičnosti",
        "",
        f"Datum izvlačenja: `{payload['draw_date']}`",
        f"Pitanje: {payload['right_question']}",
        "",
        "## Kratak odgovor",
        "",
        f"- Jačina izmerene strukture: `{audit['verdict']['signal_strength']}`.",
        f"- Otisak nasumičnosti: `{', '.join(fingerprint['randomness_type']['dominant_types'])}`.",
        f"- Tumačenje testa nasumičnosti: {audit['verdict']['plain']}",
        f"- Out-of-sample presuda: {walk['verdict']['plain']}",
        f"- Najbolji walk-forward model: `{walk['best_model']}`.",
        "",
        "## Dokazi",
        "",
        f"- Učitana izvlačenja: `{payload['source']['draws_loaded']}` od `{payload['source']['first_draw']}` do `{payload['source']['last_draw']}`.",
        f"- Kvalitet podataka (upotrebljivo): `{quality.get('usable', True)}`.",
        f"- Duplikati datuma: `{len(quality.get('duplicate_dates', []))}`.",
        f"- Greške opseg/veličina/dupli broj: `{len(quality.get('range_errors', []))}` / `{len(quality.get('size_errors', []))}` / `{len(quality.get('duplicate_number_errors', []))}`.",
        f"- Frekvencija max |z|: `{audit['frequency']['max_abs_z']:.2f}`.",
        f"- Normalizovana entropija: `{fingerprint['entropy']['normalized_entropy']:.4f}`.",
        f"- Top par lift: `{audit['pair_lift']['max_lift']:.2f}`.",
        f"- Top trojka lift: `{fingerprint['triple_lift']['max_lift']:.2f}`.",
        f"- Serijski lag max delta: `{fingerprint['serial_dependence']['max_abs_lift_delta']:.2f}`.",
        f"- Drift raspodele (JS): `{fingerprint['drift']['js_divergence']:.4f}`.",
        f"- Najbolji model — prosečan pogodak: `{best['mean_hits']:.3f}` naspram uniform `{uniform['mean_hits']:.3f}`.",
        f"- Najbolji model — stopa 2+: `{best['any_2_plus'] * 100:.2f}%` naspram uniform `{uniform['any_2_plus'] * 100:.2f}%`.",
        f"- Najbolji model — stopa 3+: `{best['any_3_plus'] * 100:.2f}%` naspram uniform `{uniform['any_3_plus'] * 100:.2f}%`.",
    ]
    if calibration:
        rows.extend(
            [
                f"- Kalibracija — null simulacije: `{payload['null_trials']}`.",
                f"- Frekvencija chi-kvadrat — kalibrisani p: `{calibration['frequency_chi_square']['empirical_p']:.4f}`.",
                f"- Par max-lift — kalibrisani p: `{calibration['pair_max_lift']['empirical_p']:.4f}`.",
                f"- Trojka max-lift — kalibrisani p: `{calibration['triple_max_lift']['empirical_p']:.4f}`.",
                f"- Temporalni lag — kalibrisani p: `{calibration['lag_max_delta']['empirical_p']:.4f}`.",
                f"- Drift JS — kalibrisani p: `{calibration['drift_js']['empirical_p']:.4f}`.",
                f"- Runs max-z — kalibrisani p: `{calibration['runs_max_abs_z']['empirical_p']:.4f}`.",
                f"- Gap anomalija — kalibrisani p: `{calibration['gap_max_abs_lift']['empirical_p']:.4f}`.",
                f"- Kalendar efekat — kalibrisani p: `{calibration['calendar_max_js']['empirical_p']:.4f}`.",
            ]
        )
    if candidate_search:
        rows.extend(
            [
                f"- Režim kandidata: `{candidate_search['candidate_mode']}`.",
                f"- Tačna pretraga: `{candidate_search['exact_used']}`.",
                f"- Ukupan prostor kombinacija: `{candidate_search['total_combinations']}`.",
                f"- Evaluisano kombinacija: `{candidate_search['evaluated_combinations']}`.",
                f"- Broj kandidata u optimizatoru: `{candidate_search['candidate_count']}`.",
            ]
        )
    rows.extend(
        [
            "",
            "## Šta to znači",
            "",
            "Ako signal postoji pre kalibracije, ali padne na kalibrisanim null testovima — verovatno je šum.",
            "Ako kalibrisani signal postoji, ali padne na walk-forward validaciji — verovatno nije upotrebljiv.",
            "Ako ipak poboljša out-of-sample metrike, sistem ga može koristiti kao slab signal težina. I dalje nema garancije.",
            f"Rezime otiska: {fingerprint['plain_language']['summary']}",
            "",
            "## Generisani tiketi",
            "",
        ]
    )
    for idx, ticket in enumerate(payload["tickets"], start=1):
        rows.append(f"{idx:02d}. {ticket['main']}")
    rows.extend(
        [
            "",
            "## Istorijski fit skupa tiketa",
            "",
            f"- Pokrivenost unije: `{payload['ticket_set_metrics']['union_size']}/{payload['ticket_set_metrics']['pool_size']}`.",
            f"- Max preklapanje parova: `{payload['ticket_set_metrics']['max_pairwise_overlap']}`.",
            f"- Max ponavljanje broja: `{payload['ticket_set_metrics']['max_number_reuse']}`.",
            f"- Pokrivenost par/trojka: `{payload['ticket_set_metrics']['pair_coverage_count']}` / `{payload['ticket_set_metrics']['triple_coverage_count']}`.",
            f"- Prosek najboljeg glavnog: `{payload['ticket_backtest']['best_main_mean']:.2f}`.",
            f"- Stopa 2+: `{payload['ticket_backtest']['any_2_plus'] * 100:.2f}%`.",
            f"- Stopa 3+: `{payload['ticket_backtest']['any_3_plus'] * 100:.2f}%`.",
        ]
    )
    if nested:
        rows.extend(
            [
                "",
                "## Ugnježdena prediktivna validacija",
                "",
                f"- Zaštita od curenja: `{nested['leakage_guard']}`.",
                f"- Test izvlačenja: `{nested['test_draws']}`.",
                f"- Prosek najboljeg glavnog: `{nested.get('best_main_mean', 0.0):.2f}`.",
                f"- Stopa 2+: `{nested.get('any_2_plus', 0.0) * 100:.2f}%`.",
                f"- Stopa 3+: `{nested.get('any_3_plus', 0.0) * 100:.2f}%`.",
                f"- Izabrani modeli: `{nested.get('selected_models', {})}`.",
            ]
        )
    rows.extend(quantum_report_rows(payload))
    rows.extend(
        [
            "",
            "## Upozorenje",
            "",
            "Statistički audit i generator tiketa optimizovan po riziku. Ne dokazuje predvidljivost izvlačenja.",
        ]
    )
    return "\n".join(rows) + "\n"


def human_report(payload: dict) -> str:
    rows = [
        f"# {payload['lottery']} — Quantum Loto Lab ({APP_VERSION})",
        "",
        f"Datum izvlačenja: `{payload['draw_date']}`",
        f"Kolone: `{payload['columns']}`",
        "",
        "## Očekivanje (čitljivo)",
        "",
        f"- Nasumična baza za bar jedan 2+ pogodak: `{payload['baseline']['random_any_2_plus'] * 100:.2f}%`.",
        f"- Nasumična baza za bar jedan 3+ pogodak: `{payload['baseline']['random_any_3_plus'] * 100:.2f}%`.",
    ]
    history = payload.get("history") or {}
    set_metrics = payload.get("ticket_set_metrics") or {}
    if history:
        rows.extend(
            [
                f"- Istorijski prosek najboljeg glavnog (backtest prozor): `{history['best_main_mean']:.2f}`.",
                f"- Istorijska stopa 2+ na generisanom skupu: `{history['any_2_plus'] * 100:.2f}%`.",
                f"- Istorijska stopa 3+ na generisanom skupu: `{history['any_3_plus'] * 100:.2f}%`.",
            ]
        )
    if set_metrics:
        rows.extend(
            [
                f"- Pokrivenost unije: `{set_metrics['union_size']}/{set_metrics['pool_size']}`.",
                f"- Max preklapanje parova: `{set_metrics['max_pairwise_overlap']}`.",
            ]
        )
    rows.extend(quantum_report_rows(payload))
    rows.extend(["", "## Tiketi", ""])
    for idx, ticket in enumerate(payload["tickets"], start=1):
        rows.append(f"{idx:02d}. {ticket['main']}")
    rows.extend(
        [
            "",
            "## Upozorenje",
            "",
            "Istraživački / zabavni alat. Nema garancije dobitka.",
        ]
    )
    return "\n".join(rows) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantum-loto-lab-v3", description=f"Loto 7/39 — {APP_VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_qcbm = sub.add_parser(
        "train-qcbm",
        help="Treniraj pozicijski QCBM combo (7×5q, empirijska sedmerka po kolonama).",
    )
    train_qcbm.add_argument("--csv", default=DEFAULT_CSV, help="Istorijski CSV izvlačenja.")
    train_qcbm.add_argument("--seed", type=int, default=RNG_SEED)
    train_qcbm.add_argument("--model", help="Put za loto739_qcbm_combo.json (podrazumevano u folderu projekta).")
    train_qcbm.add_argument("--config", help="Put za loto739_qcbm_config.json.")
    train_qcbm.set_defaults(func=cmd_train_qcbm)

    train_copula = sub.add_parser("train-copula", help="Treniraj copula — zajednička raspodela 7 brojeva.")
    train_copula.add_argument("--csv", default=DEFAULT_CSV, help="Istorijski CSV izvlačenja.")
    train_copula.add_argument("--seed", type=int, default=RNG_SEED)
    train_copula.add_argument("--model", help="Put za loto739_copula.json.")
    train_copula.add_argument("--config", help="Put za loto739_copula_config.json.")
    train_copula.set_defaults(func=cmd_train_copula)

    train_polazna = sub.add_parser(
        "train-polazna",
        help="Prvi put: QCBM + copula odjednom (inače koristi train-qcbm / train-copula posebno).",
    )
    train_polazna.add_argument("--csv", default=DEFAULT_CSV, help="Istorijski CSV izvlačenja.")
    train_polazna.add_argument("--seed", type=int, default=RNG_SEED)
    train_polazna.set_defaults(func=cmd_train_polazna)

    audit = sub.add_parser("audit", help="Loto Serbia 7/39 — audit, walk-forward, tiket.")
    audit.add_argument("--date", help="Datum izvlačenja GGGG-MM-DD.")
    audit.add_argument("--csv", default=DEFAULT_CSV, help="Istorijski CSV izvlačenja.")
    audit.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    audit.add_argument("--train-min", type=int, default=80)
    audit.add_argument("--backtest-draws", type=int, default=157)
    audit.add_argument("--nested-test-draws", type=int, default=52)
    audit.add_argument("--nested-candidate-pool", type=int, default=800)
    audit.add_argument(
        "--deep-calibration", action="store_true", help="Više null simulacija za kalibraciju nasumičnosti."
    )
    audit.add_argument("--null-trials", type=int, default=None, help="Broj null simulacija (prepisuje podrazumevano).")
    audit.add_argument("--candidate-mode", choices=["sampled", "exact"], default="sampled")
    audit.add_argument("--exact-top-k", type=int, default=10000)
    audit.add_argument("--max-exact-combinations", type=int, default=60000000)
    audit.add_argument("--seed", type=int, default=RNG_SEED)
    audit.add_argument(
        "--no-quantum",
        action="store_true",
        help="Isključi lokalni qc25 Aer simulator (podrazumevano uključen).",
    )
    audit.add_argument(
        "--no-qcbm",
        action="store_true",
        help="Ne mešaj trenirani QCBM u seed_weights (samo klasični skorovi).",
    )
    audit.add_argument(
        "--no-copula",
        action="store_true",
        help="Ne koristi treniranu copula za skor sedmerke u optimizatoru.",
    )
    audit.add_argument("--quantum-profile", choices=["standard", "long", "deep", "extreme"], default="long")
    audit.add_argument("--repeat-jobs", type=int, default=None)
    audit.add_argument("--qubits", type=int, default=None)
    audit.add_argument("--layers", type=int, default=None)
    audit.add_argument("--batch-circuits", type=int, default=None)
    audit.add_argument("--shots", type=int, default=None)
    audit.add_argument("--output", default="audit.json")
    audit.set_defaults(func=cmd_audit)

    predict = sub.add_parser("predict", help="Loto Serbia 7/39 — generiši tiket.")
    predict.add_argument("--date", help="Datum izvlačenja GGGG-MM-DD.")
    predict.add_argument("--csv", default=DEFAULT_CSV, help="Istorijski CSV izvlačenja.")
    predict.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    predict.add_argument("--backtest-draws", type=int, default=157)
    predict.add_argument("--seed", type=int, default=RNG_SEED)
    predict.add_argument(
        "--no-quantum",
        action="store_true",
        help="Isključi lokalni qc25 Aer simulator (podrazumevano uključen).",
    )
    predict.add_argument(
        "--no-qcbm",
        action="store_true",
        help="Ne mešaj trenirani QCBM u seed_weights (samo klasični skorovi).",
    )
    predict.add_argument(
        "--no-copula",
        action="store_true",
        help="Ne koristi treniranu copula za skor sedmerke u optimizatoru.",
    )
    predict.add_argument("--quantum-profile", choices=["standard", "long", "deep", "extreme"], default="long")
    predict.add_argument("--repeat-jobs", type=int, default=None)
    predict.add_argument("--qubits", type=int, default=None)
    predict.add_argument("--layers", type=int, default=None)
    predict.add_argument("--batch-circuits", type=int, default=None)
    predict.add_argument("--shots", type=int, default=None)
    predict.add_argument("--output", default="prediction.json")
    predict.set_defaults(func=cmd_predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


"""
cd 

python cli.py train-polazna --seed 39

python cli.py audit --date 2026-06-24
"""


"""
Analiza — quantum-loto-lab-v3 (3.1-loto739-combo)
Srbija sistem za Loto 7/39 na 4638 izvlačenja, seed 39, 1 tiket po auditu.

Tri sloja:
QCBM combo — 7 odvojenih kola (5 kubita, COBYLA, KL). Uči empirijsku raspodelu po kolonama Num1–Num7 (redosled iz CSV-a). Daje 39 težina za blend i parametre za qc25.

Copula — Gaussian 7D na sortiranoj sedmerki. Skoruje celu kombinaciju u optimizatoru (4637 jedinstvenih sedmerki iz istorije).

Audit — kalibrisani test nasumičnosti → walk-forward (15 modela, bira najbolji) → qc25 Aer (25 kubita, profil long) → optimizator tiketa → nested backtest na 52 kola.

Tok predikcije: klasični skor (walk-forward) + QCBM blend → seed_weights → qc25 simulator → kandidati + copula skor → finalni tiket.

Jake strane:
pozicijski QCBM hvata obrazac po mestu u sedmerki, ne samo frekvenciju brojeva
copula pokriva zajedničku raspodelu sedmerke
audit ima punu validaciju (null testovi, walk-forward, nested)
qc25 koristi trenirane QCBM parametre u kolu

Slabe strane:
QCBM mapira 33 vrednosti na 32 stanja (modulo) — gubitak preciznosti na užim opsezima
copula radi na sortiranoj sedmerki, QCBM na pozicijskoj — dva različita pogleda na iste podatke
audit traje dugo (walk-forward + nested + kvant)
nema garancije predvidljivosti — statistički signal je slab ili šum

Zaključak: najkompletnija verzija u ovom projektu — dva trenirana modela + pun statistički audit + kvantni sloj. 
"""




"""

"""