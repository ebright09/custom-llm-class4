"""Local, inference-only TrashGPT lab. Run: python trashgpt.py --open"""
import argparse
from datetime import datetime, timezone
import errno
import json
import mimetypes
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser

import torch
from run_evals import generate_reply, load_model, model_hash

ROOT = Path(__file__).resolve().parent
RUNS = {"starter": "20260922T224924_640441Z", "expanded": "20260922T225358_991487Z"}
V1 = "20260922T225202_918677Z"
STATIC = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/raccoon.svg": "raccoon.svg"}
MODELS = {}
INFERENCE_LOCK = threading.Lock()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_files():
    """Explicitly expose experiment evidence, never the project directory."""
    paths = [ROOT / p for p in ("README.md", "EXPERIMENT_REPORT.md", "custom_llm_starter.ipynb",
             "custom_llm_expanded.ipynb", "evals/language_evals.json", "results/eval_comparison.md")]
    names = ["config.json", "history.json", "inspection.json", "tokenization.json", "training.csv",
             "training_curves.svg", "training_summary.json", "corpus_manifest.json", "vocabulary_report.json",
             "eval_separation.json", "temperature_comparison.json", "chat_transcript.json"]
    for run in [*RUNS.values(), V1]:
        base = ROOT / "llm_runs" / run
        paths.extend(base / name for name in names)
        for stage in ("untrained", "final"):
            paths.extend(base / "language_evals" / stage / name for name in
                         ("eval_summary.json", "eval_results.json", "eval_results.csv"))
        paths.extend(base / "samples" / f"step_{step:04d}.txt" for step in (0, 1500, 3000))
    return {"/evidence/" + p.relative_to(ROOT).as_posix(): p for p in paths}


EVIDENCE = evidence_files()


def experiment_data():
    result = {}
    for name, run in RUNS.items():
        base = ROOT / "llm_runs" / run
        tokenization = read_json(base / "tokenization.json")
        inspection = read_json(base / "inspection.json")
        vocabulary = tokenization["vocabulary"]
        probabilities = {}
        for stage in ("before", "after"):
            probabilities[stage] = sorted(zip(vocabulary, inspection["probabilities_" + stage]),
                                          key=lambda row: row[1], reverse=True)[:5]
        split = read_json(base / "split.json")
        result[name] = {"run": run, "config": read_json(base / "config.json"),
            "training_summary": read_json(base / "training_summary.json"),
            "manifest": read_json(base / "corpus_manifest.json"),
            "inspection": inspection, "probabilities": probabilities,
            "history": read_json(base / "history.json"),
            "corpus_examples": split["train"][:4],
            "temperature_samples": read_json(base / "temperature_comparison.json"),
            "samples": {str(s): (base / "samples" / f"step_{s:04d}.txt").read_text().splitlines()
                        for s in (0, 1500, 3000)},
            "evaluations": {s: {"summary": read_json(base / "language_evals" / s / "eval_summary.json"),
                                "cases": read_json(base / "language_evals" / s / "eval_results.json")}
                            for s in ("untrained", "final")}}
    saved_chat = read_json(ROOT / "llm_runs" / RUNS["expanded"] / "chat_transcript.json")["turns"]
    terminal_chat = read_json(ROOT / "results/chat/expanded_terminal_chat.json")["turns"]
    selected = {"goose": saved_chat[3], "customer": saved_chat[0],
                "spatial": terminal_chat[1], "opposite": terminal_chat[2], "homework": terminal_chat[6]}
    curated = {key: {"prompt": turn["prompt"], "seed": turn["seed"], "expected": turn["response"]}
               for key, turn in selected.items()}
    return {"experiments": result, "earlier_run": V1, "curated_examples": curated}


def infer(payload):
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object.")
    experiment, stage = payload.get("experiment", "expanded"), payload.get("stage", "final")
    if experiment not in RUNS or stage not in ("untrained", "final"):
        raise ValueError("Choose a listed experiment and training stage.")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Enter a sentence beginning between 1 and 4,000 characters.")
    temperature, seed = payload.get("temperature", .8), payload.get("seed", 2029)
    if isinstance(temperature, bool) or temperature not in (.3, .8, 1.2):
        raise ValueError("Temperature must be 0.3, 0.8, or 1.2.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise ValueError("Seed must be a whole number between 0 and 4294967295.")
    with INFERENCE_LOCK:
        key = (experiment, stage)
        if key not in MODELS:
            filename = "model.pt" if stage == "final" else "model_untrained.pt"
            model, vocabulary, saved = load_model(ROOT / "llm_runs" / RUNS[experiment] / filename)
            identity = model_hash(model)
            expected = read_json(ROOT / "llm_runs" / RUNS[experiment] / "language_evals" / stage / "eval_summary.json")
            if identity != expected["model_sha256"]:
                raise RuntimeError("The saved model does not match its experiment evidence.")
            MODELS[key] = model, vocabulary, saved, identity
        model, vocabulary, saved, identity = MODELS[key]
        reply = generate_reply(model, vocabulary, prompt, seed=seed, temperature=temperature, max_tokens=24)
        if model_hash(model) != identity:
            raise RuntimeError("Inference changed model weights unexpectedly.")
    return {"experiment": experiment, "stage": stage, "run": RUNS[experiment],
            "model_sha256": identity, "completed_steps": saved["completed_steps"],
            "prompt": prompt, "seed": seed, "temperature": temperature, "max_tokens": 24,
            "fresh_context_per_prompt": True, "context_tokens": 48,
            "timestamp": datetime.now(timezone.utc).isoformat(), **reply}


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, mime="application/json; charset=utf-8"):
        data = content if isinstance(content, bytes) else json.dumps(content, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def do_GET(self):
        if not self.valid_host():
            return self.send(403, {"error": "Open the local 127.0.0.1 address printed by the launcher."})
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/experiments":
                return self.send(200, experiment_data())
            file = ROOT / "trashgpt_web" / STATIC[path] if path in STATIC else EVIDENCE.get(path)
            if file is None or not file.is_file():
                return self.send(404, {"error": "That page or evidence file is unavailable."})
            self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        except (OSError, KeyError, ValueError):
            self.send(500, {"error": "An experiment evidence file is missing or unreadable. See the launch instructions."})

    def do_POST(self):
        origin = self.headers.get("Origin")
        if not self.valid_host() or (origin and origin != f"http://127.0.0.1:{self.server.server_port}"):
            return self.send(403, {"error": "Use the local TrashGPT page to generate replies."})
        if self.path != "/api/generate":
            return self.send(404, {"error": "Unknown endpoint."})
        if self.headers.get_content_type() != "application/json":
            return self.send(415, {"error": "Send application/json."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 20000:
                return self.send(413, {"error": "Request is too large or empty."})
            payload = json.loads(self.rfile.read(length))
            self.send(200, infer(payload))
        except (ValueError, TypeError, KeyError):
            self.send(400, {"error": "Check your prompt, experiment, stage, temperature, and seed."})
        except (OSError, RuntimeError):
            self.send(500, {"error": "The saved model could not be loaded or verified. Check the launch instructions."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open your browser")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"TrashGPT is open for questionable business: {url}\nPress Control-C to close.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
