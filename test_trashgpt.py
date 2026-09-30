"""Integration checks for the local lab and the preserved experiment evidence."""
import hashlib
import http.client
import json
import threading
import unittest

import torch
import trashgpt as lab
from run_evals import generate_reply, model_hash


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.server = lab.ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def test_evidence_matches_all_four_complete_results(self):
        status, body, _ = self.request("GET", "/api/experiments")
        self.assertEqual(status, 200)
        data = json.loads(body)
        for name, experiment in data["experiments"].items():
            self.assertEqual(len(experiment["inspection"]["embedding_before"]), 64)
            self.assertEqual(len(experiment["inspection"]["embedding_after"]), 64)
            for stage, result in experiment["evaluations"].items():
                cases, summary = result["cases"], result["summary"]["overall"]
                self.assertEqual(len(cases), 48)
                self.assertEqual(len({c["id"] for c in cases}), 48)
                self.assertEqual(sum(c["score"] for c in cases), summary["correct"])
                self.assertEqual(sum(c["status"] in ("scored", "tied") for c in cases), summary["scorable"])
                self.assertEqual(summary["correct"], 9 if stage == "untrained" else 20 if name == "starter" else 28)
        self.assertEqual(data["experiments"]["starter"]["inspection"]["token_id"], 28)
        self.assertEqual(data["experiments"]["expanded"]["inspection"]["token_id"], 105)

    def test_all_models_match_direct_inference_without_updates(self):
        for experiment in lab.RUNS:
            for stage in ("untrained", "final"):
                for temperature in (.3, .8, 1.2):
                    settings = dict(experiment=experiment, stage=stage, temperature=temperature,
                                    seed=2029, prompt="the cup is not full . it is")
                    status, body, _ = self.request("POST", "/api/generate", json.dumps(settings),
                                                 {"Content-Type": "application/json"})
                    self.assertEqual(status, 200, body)
                    actual = json.loads(body)
                    model, vocabulary, saved, identity = lab.MODELS[(experiment, stage)]
                    direct = generate_reply(model, vocabulary, settings["prompt"], seed=2029, temperature=temperature)
                    self.assertEqual(actual["response"], direct["response"])
                    self.assertEqual(model_hash(model), identity)
                    self.assertEqual(actual["model_sha256"], identity)
                    self.assertTrue(actual["fresh_context_per_prompt"])
                    if experiment == "expanded" and stage == "final" and temperature == .8:
                        self.assertEqual(actual["response"], "a goose .")

    def test_unknown_empty_and_long_outputs_remain_visible_in_data(self):
        response = lab.infer(dict(experiment="starter", prompt="the opposite of hot is", seed=2029))
        self.assertEqual(response["response"], "")
        self.assertEqual(response["unknown_prompt_words"], ["hot", "is", "opposite"])
        long = lab.infer(dict(prompt="the customer " * 60))
        self.assertTrue(long["prompt_truncated"])

    def test_validation_and_local_only_boundaries(self):
        for value in [[], {}, {"prompt":" "}, {"prompt":"x", "temperature":float('nan')},
                      {"prompt":"x", "seed":True}, {"prompt":"x", "seed":-1},
                      {"prompt":"x", "experiment":"../../"}, {"prompt":"x"*4001}]:
            status, _, _ = self.request("POST", "/api/generate", json.dumps(value), {"Content-Type":"application/json"})
            self.assertEqual(status, 400)
        self.assertEqual(self.request("POST", "/api/generate", "{}", {"Content-Type":"text/plain"})[0],415)
        self.assertEqual(self.request("POST", "/api/generate", "{}", {"Content-Type":"application/json", "Origin":"https://example.com"})[0],403)
        self.assertEqual(self.request("GET", "/", headers={"Host":"example.com"})[0],403)
        for path in ["/../.git/config", "/.git/config", "/evidence/../trashgpt.py", "/llm_runs/", "/missing"]:
            self.assertEqual(self.request("GET", path)[0],404)

    def test_approved_assets_and_evidence_links_exist(self):
        for path in [*lab.STATIC, *lab.EVIDENCE]:
            status, _, headers = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        for file, digest in {
            "run_evals.py":"da87f28d128344807512e2bac1cfc662b37ac2c7e4a32b84c09f1950e92d67a0",
            "chat.py":"6152c8b7780f3b46fef5de38461adfc4b1a55df70ed106ca73ec5e9aded86d25",
            "evals/language_evals.json":"e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7",
        }.items():
            self.assertEqual(hashlib.sha256((lab.ROOT/file).read_bytes()).hexdigest(),digest)


if __name__ == "__main__":
    unittest.main()
