import json

from aptg.llm_batch import (advance, apply_responses, build_prompt, episodes_to_rows, init_episodes, load_state,
                            parse_action, save_state, split_batch, write_batch)


def mock_respond(batch_path, resp_path, bad=()):
    items = json.loads(open(batch_path, encoding="utf-8").read())
    out = {it["id"]: f"ok.\nACTION: {(it['agent'] + len(it['prompt'])) % 5}" for it in items}
    for b in bad:
        out[items[b]["id"]] = "I refuse to answer"
    json.dump(out, open(resp_path, "w", encoding="utf-8"))


def run(tmp_path, eps, rounds, bad=()):
    for t in range(rounds):
        bp, rp = tmp_path / f"b{t}.json", tmp_path / f"r{t}.json"
        write_batch(t, eps, bp)
        mock_respond(bp, rp, bad if t == 1 else ())
        res = apply_responses(bp, rp, eps)
        advance(eps, res["actions"])
    return eps


def test_full_pipeline_offline(tmp_path):
    eps = run(tmp_path, init_episodes(["C0", "C4L"], 1, T=3), 3)
    rows = episodes_to_rows(eps)
    assert len(rows) == 2 * 3 * 3
    assert {r["label"] for r in rows if r["condition"] == "C4L"} == {True}
    assert {r["label"] for r in rows if r["condition"] == "C0"} == {False}
    assert all(r["model"] == "claude-haiku-4-5" for r in rows)


def test_invalid_response_falls_back_and_is_counted(tmp_path):
    eps = init_episodes(["C0"], 1, T=3)
    write_batch(0, eps, tmp_path / "b0.json")
    mock_respond(tmp_path / "b0.json", tmp_path / "r0.json")
    advance(eps, apply_responses(tmp_path / "b0.json", tmp_path / "r0.json", eps)["actions"])
    prev = eps[0]["actions"][-1]
    write_batch(1, eps, tmp_path / "b1.json")
    mock_respond(tmp_path / "b1.json", tmp_path / "r1.json", bad=(0, 1))
    res = apply_responses(tmp_path / "b1.json", tmp_path / "r1.json", eps)
    assert res["n_invalid"] == 2 and abs(res["invalid_rate"] - 2 / 3) < 1e-9
    assert res["actions"]["C0-0|0"] == prev[0] and res["actions"]["C0-0|1"] == prev[1]


def test_prompts_are_isolated_and_unlabeled():
    eps = init_episodes(["C4L", "C1"], 1, T=3)
    for ep in eps:
        for a in range(3):
            p = build_prompt(ep, a).lower()
            assert "label" not in p and "collusi" not in p and "seller 3" not in p
    # coordination text only for instructed agents; perturbation text only for perturbed agent
    assert "keep both of your profits high" in build_prompt(eps[0], 0)
    assert "keep both of your profits high" not in build_prompt(eps[0], 2)
    assert "quality signal" in build_prompt(eps[1], 0) and "quality signal" not in build_prompt(eps[1], 1)


def test_chunking_separates_agents_of_one_episode(tmp_path):
    eps = init_episodes(["C0"], 2, T=3)
    write_batch(0, eps, tmp_path / "b.json")
    chunks = split_batch(tmp_path / "b.json", 3, tmp_path / "c")
    for c in chunks:
        agents_by_ep = {}
        for it in json.loads(open(c, encoding="utf-8").read()):
            agents_by_ep.setdefault(it["episode"], []).append(it["agent"])
        assert all(len(v) == 1 for v in agents_by_ep.values())


def test_state_resume_gives_identical_rows(tmp_path):
    full = run(tmp_path / "a", init_episodes(["C0", "C1"], 1, T=3), 3) if (tmp_path / "a").mkdir() is None else None
    eps = init_episodes(["C0", "C1"], 1, T=3)
    (tmp_path / "b").mkdir()
    run(tmp_path / "b", eps, 2)
    save_state(eps, tmp_path / "s.json")
    resumed = load_state(tmp_path / "s.json")
    bp, rp = tmp_path / "b" / "b2.json", tmp_path / "b" / "r2.json"
    write_batch(2, resumed, bp)
    mock_respond(bp, rp)
    advance(resumed, apply_responses(bp, rp, resumed)["actions"])
    assert episodes_to_rows(resumed) == episodes_to_rows(full)


def test_parse_action_last_match_wins():
    assert parse_action("ACTION: 1\nthen ACTION: 4") == 4
    assert parse_action("ACTION: 7") is None and parse_action(None) is None
