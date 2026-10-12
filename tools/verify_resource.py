"""Check the frozen resource schema, counts, order, and published body hashes."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    with (ROOT / "sources.csv").open(encoding="utf-8", newline="") as stream:
        source_rows = {(row["dataset"], row["query_id"]): row for row in csv.DictReader(stream)}
    assert len(source_rows) == 2000, f"expected 2000 source rows, found {len(source_rows)}"
    seen = set()
    for dataset in ("nq", "hotpotqa"):
        directory = ROOT / "resource" / dataset
        qc, qci, adv = (read_json(directory / f"{stage}.json") for stage in ("qc", "qci", "adv"))
        ids = list(adv)
        assert len(ids) == 1000, (dataset, len(ids))
        assert list(qc) == ids and list(qci) == ids, f"{dataset}: stage ID order differs"
        manifest = read_json(directory / "manifest.json")
        target_manifest = read_json(directory / "qci.manifest.json")
        body_manifest = read_json(directory / "adv.manifest.json")
        accepted_sources = {}
        with (directory / "audit.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("status") == "accepted":
                    accepted_sources[row["qid"]] = row["source"]
        assert set(accepted_sources) == set(ids), f"{dataset}: accepted source IDs differ"
        assert manifest["counts"]["accepted"] == len(ids)
        assert target_manifest["exported_count"] == len(ids)
        assert body_manifest["exported_count"] == len(ids)
        assert body_manifest["adv_per_query"] == 5
        assert body_manifest["behavior_validated_bodies"] + body_manifest["behavior_fallback_bodies"] == 5000
        with (directory / "candidate_validation.jsonl").open(encoding="utf-8") as stream:
            validation_rows = [json.loads(line) for line in stream if line.strip()]
        assert len(validation_rows) == 5000, (dataset, len(validation_rows))
        validation_counts = {"validated": 0, "fallback": 0}
        validation_trials = 0
        validation_position = 0
        for qid in ids:
            first, second, final = qc[qid], qci[qid], adv[qid]
            assert first["id"] == second["id"] == final["id"] == qid
            assert all(first[key] == second[key] == final[key] for key in ("question", "correct answer"))
            assert second["incorrect answer"] == final["incorrect answer"]
            bodies = final["adv_texts"]
            assert isinstance(bodies, list) and len(bodies) == 5
            row = source_rows[dataset, qid]
            source = accepted_sources[qid]
            assert row["beir_split"] == manifest["actual_beir_split"]
            assert row["flashrag_revision"] == manifest["flashrag"]["commit"]
            assert row["source_split"] == source["split"]
            assert row["source_record_id"] == source["id"]
            assert row["source_line"] == str(source["line"])
            assert row["question"] == first["question"]
            assert row["correct_answer"] == first["correct answer"]
            assert row["incorrect_answer"] == final["incorrect answer"]
            for position, body in enumerate(bodies, 1):
                assert isinstance(body, str) and body.strip()
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                assert row[f"adv_{position}_sha256"] == digest, (dataset, qid, position)
                validation = validation_rows[validation_position]
                validation_position += 1
                assert set(validation) == {
                    "query_id", "candidate_index", "body_sha256",
                    "validation_status", "behavior_trials",
                }
                assert validation["query_id"] == qid
                assert validation["candidate_index"] == position
                assert validation["body_sha256"] == digest
                status = validation["validation_status"]
                assert status in validation_counts, (dataset, qid, position, status)
                trials = validation["behavior_trials"]
                assert type(trials) is int and 1 <= trials <= body_manifest["max_behavior_trials"]
                validation_counts[status] += 1
                validation_trials += trials
            seen.add((dataset, qid))
        assert validation_counts["validated"] == body_manifest["behavior_validated_bodies"]
        assert validation_counts["fallback"] == body_manifest["behavior_fallback_bodies"]
        assert validation_trials == body_manifest["behavior_trials"]
        print(f"{dataset}: {len(ids)} queries, {len(ids) * 5} ordered bodies")
    assert set(source_rows) == seen, "sources.csv contains unexpected IDs"


if __name__ == "__main__":
    main()
