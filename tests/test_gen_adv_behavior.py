import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gen_adv import MAX_BEHAVIOR_TRIALS, run_adv_generation


def response(text):
    return {"text": text, "finish_reason": "stop", "usage": {}}


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def query_with_metadata(self, _prompt, _system_prompt=None):
        self.calls += 1
        return next(self.responses)


class BehaviorValidationTests(unittest.TestCase):
    def prepare(self, root):
        input_path = root / "qci.json"
        config_path = root / "config.json"
        output_path = root / "adv.json"
        input_path.write_text(json.dumps({
            "q1": {
                "id": "q1", "question": "Who won?",
                "correct answer": "Alice", "incorrect answer": "Bob",
            }
        }), encoding="utf-8")
        config_path.write_text(json.dumps({
            "model_info": {"provider": "gpt", "name": "mock"},
            "params": {"temperature": 0, "seed": 1, "gpus": []},
        }), encoding="utf-8")
        return input_path, config_path, output_path

    def test_regenerates_until_proxy_returns_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, config_path, output_path = self.prepare(root)
            model = SequenceModel([
                response("First candidate."), response("Alice"),
                response("Second candidate."), response("Bob"),
            ])
            code = run_adv_generation(
                input_path, output_path, config_path, 1, 0, False, model=model)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output_path.read_text("utf-8"))["q1"]["adv_texts"],
                             ["Second candidate."])
            manifest = json.loads(output_path.with_suffix(".manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["behavior_validated_bodies"], 1)
            self.assertEqual(manifest["behavior_trials"], 2)
            self.assertEqual(model.calls, 4)

    def test_keeps_last_candidate_after_five_failed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, config_path, output_path = self.prepare(root)
            responses = []
            for index in range(MAX_BEHAVIOR_TRIALS):
                responses.extend([response(f"Candidate {index}."), response("Alice")])
            model = SequenceModel(responses)
            code = run_adv_generation(
                input_path, output_path, config_path, 1, 0, False, model=model)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output_path.read_text("utf-8"))["q1"]["adv_texts"],
                             [f"Candidate {MAX_BEHAVIOR_TRIALS - 1}."])
            manifest = json.loads(output_path.with_suffix(".manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["behavior_validated_bodies"], 0)
            self.assertEqual(manifest["behavior_fallback_bodies"], 1)
            self.assertEqual(manifest["behavior_trials"], MAX_BEHAVIOR_TRIALS)


if __name__ == "__main__":
    unittest.main()
