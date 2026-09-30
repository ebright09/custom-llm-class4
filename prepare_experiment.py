"""Create an experiment copy of the starter notebook with my prediction and chat prompts.

Only three things change relative to custom_llm.ipynb: the prediction text, the
experiment label, and the chat cell, which loops over several prompts instead of
one so the executed notebook records at least three real interactions. Model,
training, evaluation and saving code are untouched.

    python prepare_experiment.py starter   # -> custom_llm_starter.ipynb
    python prepare_experiment.py expanded  # -> custom_llm_expanded.ipynb (corpus revision 2)
    python prepare_experiment.py expanded_v1  # first corpus version, kept for the record
    python prepare_experiment.py setup     # 10-step setup check (TRAINING_STEPS = 10)
"""
import sys
import nbformat

PREDICTIONS = {
"setup": """### My prediction (setup check: 10 steps)
**Choices.** The classroom corpus with an empty `corpus/` folder and learning rate 0.001,
but only `TRAINING_STEPS = 10`. The assignment says to try 10 steps first to check that
the notebook runs from top to bottom before spending the real 3,000-step budget.

**Prediction.** Everything should run and save a results ZIP, but 10 updates out of a
planned 3,000 should barely move the model: the loss should still be close to
ln(136) = 4.91, the samples should still be random words, and the eval score should
stay near chance.
""",
"starter": """### My prediction (experiment 1: starter corpus)
**Choices.** `CORPUS = "classroom"` with an empty `corpus/` folder, so this is the
unmodified baseline. `TRAINING_STEPS = 3000` because the assignment recommends it as the
starting budget and the corpus is small and repetitive, so a few thousand batches of
32 passages should see every template many times. `LEARNING_RATE = 0.001`, the
recommended AdamW value for a model this small: much larger steps could make loss
jump around or diverge, much smaller ones would barely move the random weights in 3,000 steps.

**Prediction.** Before training the loss should be close to ln(vocabulary size),
and samples should be random word salad. After
training I expect both panel losses to fall well below 1.0 and stay close together,
because validation passages come from the same templates. Final samples should be
grammatical template sentences that mix nouns of one domain with their contexts.
I expect most of the 16 starter-pattern evals to pass, some of the 8 new-wording
evals, and **none** of the 24 extension evals, because their words (opposite, hot,
bird, inside, ...) are not in this vocabulary. For the word **customer** I expect
its nearest neighbors to become client/buyer/shopper-type words that share its contexts.
""",
"expanded_v1": """### My prediction (experiment 2, first version: expanded corpus v1)
**Choices.** Same settings as the starter run (`CORPUS = "classroom"`, 3,000 steps,
learning rate 0.001, same seed) so that the only change is the data. `corpus/`
now contains four Markdown files I wrote myself (`corpus/extension/*.md`) targeting
four extension categories: **opposites, negation, spatial relations and
categories/analogies**. They use different words, people and sentence shapes from
the eval prompts, and the notebook's leakage check rejects any exact test prefix.

**Prediction.** The vocabulary will grow (new words such as opposite, cold, inside,
below, fish, cat) and the unknown-word coverage of the 24 extension evals should
rise for the categories I taught. I expect a few of those cases to become correct,
mainly opposites and categories where the answer word follows a simple learned
pattern; negation and spatial reasoning may still fail because they require
copying or inverting a word from earlier in the context. The starter-pattern
scores should stay about the same, but the classroom corpus now shares the model
with new text, so losses are not directly comparable across the two runs.
Categories I did not teach (grammar, reference, sequence, everyday knowledge)
should stay mostly unscorable or wrong.
""",
"expanded": """### My prediction (experiment 2: expanded corpus, revision 2)
**Choices.** Same settings as the starter run (`CORPUS = "classroom"`, 3,000 steps,
learning rate 0.001, same seed) so that the only change is the data. `corpus/extension/`
holds four Markdown files I generated with `make_extension_corpus.py` for four
extension categories: **opposites, negation, spatial relations and categories/analogies**.
They use different objects, names and word pairs from the eval items, and the
notebook's leakage check rejects any exact test prefix.

**Why a revision.** My first expanded run (kept in `experiments/`) made only 5 of
the 24 extension cases scorable: a case needs its prompt *and all four answer
choices* in the vocabulary, and I had not taught distractor words such as heavy,
loud, missing, beside or north. Revision 2 adds those words only in neutral
sentences. Eval names such as ava are still never added.

**Prediction.** Vocabulary coverage of my four categories should rise from 5 to
11 of their 12 cases (only the ava case should stay unknown). I expect
categories and "inside/contains" to pass, because v1 already got them right, and
opposites to be the hardest: the tested pairs (hot/cold, empty/full, noisy/quiet)
never appear as opposites in my data, so the model can only guess from the pattern.
Starter-pattern scores should stay at 16/16. Categories I did not teach (grammar,
reference, sequence, everyday knowledge) should stay unscorable.
""",
}

CHAT_PROMPTS = {
"setup": ["the customer", "the doctor explained the", "the opposite of hot is"],
"starter": ["the customer", "the doctor explained the", "our school has a question about the",
            "the opposite of hot is", "what should i eat for lunch ?"],
"expanded_v1": ["the customer", "the doctor explained the", "the opposite of tall is",
             "the cup is not full . it is", "a sparrow is a", "what should i eat for lunch ?"],
"expanded": ["the customer", "the doctor explained the", "the opposite of tall is",
             "the cup is not full . it is", "a sparrow is a", "what should i eat for lunch ?"],
}

CHAT_CELL = '''CHAT_PROMPTS = {prompts!r}  # each prompt is one fresh message
chat_file = run_dir/"chat_transcript.json"
chat_record = json.loads(chat_file.read_text()) if chat_file.exists() else {{
    "model_sha256":model_hash(model), "completed_steps":completed_steps,
    "fresh_context_per_prompt":True, "temperature":0.8, "max_tokens":24, "turns":[]}}
if chat_record["model_sha256"] != model_hash(model):
    raise ValueError("The model changed. Start a new run instead of mixing chat evidence.")
for CHAT_PROMPT in CHAT_PROMPTS:
    chat_seed = 2026 + len(chat_record["turns"])
    reply = generate_reply(model, vocabulary, CHAT_PROMPT, seed=chat_seed)
    print("You:", CHAT_PROMPT, "\\nModel:", reply["response"] or "[empty response]")
    if reply["unknown_prompt_words"]:
        print("Unknown words:", reply["unknown_prompt_words"])
    if reply["prompt_truncated"]:
        print("Long prompt: only the most recent 48 tokens were used.")
    print()
    chat_record["turns"].append({{"prompt":CHAT_PROMPT, "seed":chat_seed, **reply}})
save_json("chat_transcript.json",chat_record)
archive = shutil.make_archive(str(run_dir),"zip",run_dir)
print("Saved chat and refreshed ZIP:", archive)
try:
    display(FileLink(archive))
except NameError:
    pass'''

experiment = sys.argv[1]
nb = nbformat.read("custom_llm.ipynb", as_version=4)
prediction = nb.cells[2]
assert prediction.source.startswith("### My prediction")
prediction.source = PREDICTIONS[experiment] + "\n## 2." + prediction.source.split("\n## 2.", 1)[1]
chat = nb.cells[23]
assert chat.source.startswith('CHAT_PROMPT = "the customer"')
chat.source = CHAT_CELL.format(prompts=CHAT_PROMPTS[experiment])
if experiment == "setup":
    settings = nb.cells[1]
    assert "TRAINING_STEPS = 3000" in settings.source
    settings.source = settings.source.replace("TRAINING_STEPS = 3000", "TRAINING_STEPS = 10  ")
nbformat.write(nb, f"custom_llm_{experiment}.ipynb")
print("Wrote", f"custom_llm_{experiment}.ipynb")
