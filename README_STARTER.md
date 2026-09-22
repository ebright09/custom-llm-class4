# Building a Custom LLM with nanoGPT

Class 4, Fall 26 · From Zero to AI Agents

Train Karpathy's actual **nanoGPT transformer** from scratch and inspect its learned
**word-token embeddings**. The classroom adaptation uses whole words and punctuation,
a small sentence corpus, and an explanatory notebook. nanoGPT itself supports different
tokenizers; changing the model name alone would not turn character tokens into words.

[Open in Colab](https://colab.research.google.com/github/pepealonso95/custom-llm/blob/main/custom_llm.ipynb)
· [Assignment Google Doc](https://docs.google.com/document/d/1MQ3YQl2ywWZF7W5_l_91FiIp7pTYPO_3viI2JVapRcc/edit)
· [Assignment text](ASSIGNMENT.md)
· [3D embedding viewer](embedding-viewer.html)

## Start here

1. Open the notebook in Colab and save your own copy. The default CPU runtime is enough.
2. Choose corpus, training steps and learning rate in section 1. Optionally add PDF, TXT or MD files to `corpus/` as explained below. Write your reasons and prediction.
3. Try 10 steps for setup, then start with 3,000 steps and a learning rate of 0.001.
4. Run All. Inspect the data, IDs, vectors, gradient, first weight update, probabilities, attention and samples.
5. Inspect the 48 language evals in sections 6b and 8b. Keep every result, including unknown-word cases. Use section 10 to chat with your trained model and save at least three real interactions.
6. Choose at least two extension categories, add different teaching examples to `corpus/`, and run a second experiment using the same tests.
7. Download each results ZIP and the executed notebook separately after the final cell.
8. Download embedding-viewer.html and open it locally. Use **Open your checkpoint** to load checkpoint.json from your extracted results ZIP.
9. Explain the actual evidence in your own README and submit your public repository URL through the [course portal](https://submissions-portal-eight.vercel.app).

Locally, install the dependencies in requirements.txt, then open custom_llm.ipynb
with that Python environment. You can also run custom_llm.py directly after editing
its settings. Colab generally already includes PyTorch. Setup installs pypdf if absent, creates `corpus/`, and downloads
the pinned nanoGPT source if needed and verifies its hash; it downloads no model weights.

## Fixed language evals and chat

The repository includes **48 synthetic language evals**: 16 reserved starter-pattern
cases, 8 new phrasings using starter vocabulary, and 24 corpus-extension challenges
covering grammar, opposites, negation, references, sequence, spatial relations,
everyday knowledge, and categories/analogies.

**[Read the eval guide and examples](evals/README.md)** ·
**[Inspect all 48 cases](evals/language_evals.json)**

Run All now saves complete untrained/final scores and free continuations, category
breakdowns, unknown-word coverage, and corpus-separation evidence in the results ZIP.
The answer key stays outside the model input and training data. Exact test prefixes
are excluded from generated training sentences; imported files containing them are
rejected. The original validation-loss panels remain a separate measurement.

Students first save a starter-corpus run, then add different teaching material for
at least two extension categories and compare a second run. Keep all tests fixed.
This is a public development benchmark, not an unseen generalization claim. A low
score is valid evidence; there is no required pass rate.

**How this affects the assignment grade:** the course's 10-point framework stays
deliverable quality **4**, testing & evaluation **3**, and working result **3**.
Complete, valid evals and a reasoned comparison of both corpus experiments are
required evidence for the 3-point evaluation category. Submit all four untrained/
trained result sets. Missing runs, leaked tests, or missing analysis reduce credit;
a low model score alone does not. The runner's score is not an automatic grade.
See [the full grading guidance](ASSIGNMENT.md#how-evals-affect-your-assignment-grade).

Notebook section 10 provides a working prompt/reply interface. Edit the prompt and
rerun its cell; each message starts fresh. Terminal alternatives after training:

```sh
python run_evals.py --model llm_runs/YOUR_RUN/model.pt --output results/my-evals
python chat.py --model llm_runs/YOUR_RUN/model.pt --transcript results/my-chat.json
```

Replace `YOUR_RUN` with your actual folder. `model.pt` contains the network;
`checkpoint.json` serves the embedding viewer. No external model API is used.
Keep `evals/`, results, and chat transcripts outside `corpus/`.

## What students should understand

| Idea | Evidence |
|---|---|
| Corpus and data | Imported text, short passages, deduplication and held-out passages |
| Tokens and IDs | Words/punctuation mapped to arbitrary integer IDs |
| Vectors and embeddings | One word's 64 numbers before/after, and the complete table |
| Neural networks | Weighted sums, GELU, attention blocks, residuals and parameters |
| Learning | Next-token loss, a real gradient and a parameter update |
| Context and prediction | Trained causal attention and next-token probabilities |
| Inference | Samples at three temperatures without weight updates |

The story is **examples → predictions → loss → gradients → updates → changed predictions**.
Character embeddings were already real embeddings in the original lab; the difference
here is that the units represent words rather than letters. A coordinate is not a
named concept, and a small language model is not a general chat assistant.

## Corpus and tokenizer

The default is a **synthetic classroom corpus**, generated visibly in the notebook.
It repeats sentence contexts around business, finance, food, transport, software,
health and education words. No category labels or coordinates are given to the model
or viewer. This deliberately controlled dataset makes distributional learning easy
to inspect; the resulting similarities are not evidence of broad semantic knowledge.

- Reserve fixed language-eval prompts before the split or vocabulary building.
- Split long text into passages of at most 47 word/punctuation tokens, normalize
  case and spacing, deduplicate, then split passages 90/10.
- Build the vocabulary only from training passages. Keep the 509 most frequent
  word/punctuation types, plus UNK, BOS and EOS (512 total at most).
- Reserve UNK for omitted/unknown tokens, BOS for passage start and EOS for passage end.
- Report unknown-token rates for both training and held-out text.
- Validation shares sentence templates with training. It tests new combinations
  within those templates, not generalization to unseen domains or writing styles.

## Expand your corpus with files

Put your files in **`corpus/` beside the notebook**, for example:

```text
custom_llm.ipynb
corpus/
  report.pdf
  notes.txt
  research/
    summary.md
```

1. **Locally:** add files to that folder. Subfolders and uppercase extensions work too.
2. **In Colab:** run sections 1 and 2 once to create `/content/corpus`. In the left
   Files sidebar, refresh and upload your files into that folder. Opening the notebook
   from GitHub does not copy the repository's folders or your local files into Colab.
3. Keep `CORPUS = "classroom"` to add the files to the teaching sentences. Use
   `CORPUS = "folder"` to train only on your files. Folder-only mode needs at least
   100 distinct extracted passages. `CORPUS_FOLDER` can point to another local folder.
4. **Run All from the top.** Section 3 reports the imported filenames, previews,
   passage counts and extraction warnings. Check that the expected text is present.
5. After training, download the new results ZIP and load its `checkpoint.json` in the
   embedding viewer. Adding files alone does not update the model or viewer: this is
   training from scratch, not a document search system.

PDFs must contain extractable text. Scans need OCR first; encrypted, unreadable and
entirely textless files stop the run with the filename and a useful error. PDFs with
some textless pages produce warnings. Inspect previews and `corpus.txt`, especially
for tables, columns or headers whose extraction order can be confusing. TXT and MD
must use UTF-8. Markdown is read as plain text; links and code are not fetched or run.

Long text is split automatically, not truncated, with sentence/line boundaries kept
when possible. There is no overlap between chunks. The 90/10 split is by deduplicated
**passage, not original file**: parts of one source file can appear in both sets.
This does not measure generalization to entirely unseen source documents.

Larger vocabularies no longer cause rejection. Tokens outside the 509 retained types
become UNK, as do token strings longer than 128 characters. Inspect
`vocabulary_report.json` and the printed coverage rates; a large, varied collection
can lose much of its detail in this deliberately small vocabulary. The notebook warns
above 5% unknown tokens. Start with focused, related text, not an entire library.

`corpus_manifest.json` records filenames, hashes, previews, warnings, added passages
and duplicate counts. Limits: 50 supported files, 25 MB each, 100 MB total, 200 pages
per PDF and 2 million extracted characters per file. Hidden files, symbolic links,
unsupported formats and the root `corpus/README.md` instructions are ignored.

**Sharing:** added source files in `corpus/` are Git-ignored, but the results ZIP,
executed notebook and trained model can still expose their content. The ZIP includes
extracted text, filenames/hashes and weights. Use material you have permission to use
and share; review every artifact before publishing. Colab uploads disappear when its
runtime storage is reset. See [the folder instructions](corpus/README.md).

## The actual nanoGPT model

[nanogpt_model.py](nanogpt_model.py) is an unchanged copy of Karpathy's
[model.py at commit 3adf61e](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py).
Its [MIT license](NANOGPT_LICENSE) is included.

The classroom configuration uses 2 blocks, 4 heads, 64-dimensional token/position
embeddings, a 48-token context, LayerNorm, GELU, residual connections and tied
input/output embeddings. PyTorch handles autograd; batched AdamW replaces the old
handwritten scalar training loop. Word tokenization and the teaching/evaluation/export
helpers are classroom additions, not claims about nanoGPT's default tokenizer.

The upstream repository now labels nanoGPT deprecated in favor of nanochat.
We deliberately pin nanoGPT here because this assignment is about its compact,
inspectable GPT implementation, not adopting a production training stack.

## Viewer

Open the single offline HTML file. It bundles the reference model's **actual recorded
initial and final token lookup embeddings**. Drag to rotate, scroll or use buttons to
zoom, and select a word from the menu or click a dot. Scroll the vector panel for all
64 coordinates. Selected words and their three closest neighbors are labeled.

Both states share a PCA center, basis and scale. The default projection retains
40.9% of pooled variance, so proximity in 3D can distort the full space. Neighbor
rankings use cosine similarity across all 64 coordinates. Movement lines connect
endpoints, not intermediate training trajectories. These are token lookup embeddings,
not position embeddings or context-dependent representations after attention.

Load your own checkpoint.json to see your actual run, including its saved initial
table. Files stay on your device. Legacy character checkpoints remain supported;
when no initial table is present, before/after comparison is disabled.

## Measured language-eval run

The revised notebook was executed with 3,000 steps: **16/16 starter-pattern cases**
and **4/8 new-wording cases** passed. All 24 extension cases lacked the required
vocabulary. Complete untrained/final outputs and an actual three-prompt terminal
transcript are in [the language-eval reference](examples/language-evals/README.md).
These are measured teaching examples, not a required student score.

## Historical reference run

The historical reference notebook below was executed with 3,000 steps and learning rate 0.001,
before the separate language-eval suite was added. Its numbers are not results for
the new suite or the new reserved-prompt corpus.
It learned 136 word/punctuation/special-token vectors. A separate 10-step setup run
also completed. These are fixed panels of 20 documents per split, not full-corpus loss.

| Step | Training panel loss | Validation panel loss |
|---|---:|---:|
| 0 | 4.9238 | 4.9247 |
| 1500 | 0.6929 | 0.7113 |
| 3000 | 0.6956 | 0.7057 |

Final examples include “our school has a question about the new educator and lesson .”
and “the consumer compared the merchandise after checking the price .”
The measured nearest neighbors of customer are client, buyer and subscriber.
Their similar designed contexts explain this result; it does not prove general understanding.

Inspect [the executed notebook](examples/custom_llm.executed.ipynb),
[complete reference evidence](examples/reference/), and
[results ZIP](examples/reference.zip). Use your own outputs in your submission.

## Results and longer training

Every run saves config.json, corpus.txt, corpus_manifest.json, vocabulary_report.json,
split.json, tokenization.json, inspection.json,
history.json, training.csv, training_summary.json, training_curves.svg, the sample
timeline, temperature_comparison.json, checkpoint.json and model.pt. It also saves
model_untrained.pt, eval_separation.json, language_eval_comparison.json, and complete
language_evals/untrained and language_evals/final folders. The chat cell adds
chat_transcript.json and refreshes the ZIP.

- **checkpoint.json:** token labels and initial/final embedding tables for the viewer.
- **model.pt:** all network weights and model settings for inference.
- Neither includes the complete optimizer/random state for exact training resume.
- For the required corpus-extension experiment, keep the original results and train
  a fresh model with your added teaching examples.
- To train longer, set TRAINING_STEPS to 5000 or 10000 and Run All from the top.
  Compare validation loss and samples. More steps can overfit and are not required.
- Interrupted training can be followed by the remaining save cells. Other failures
  require correcting the cause; do not assume a complete ZIP was saved.

Use [STUDENT_README.md](STUDENT_README.md) to organize your explanation.
The [historical microgpt lab](legacy/README.md) is preserved separately; its results
must not be presented as results of this word-token model.

For maintainers: run python3 build_embedding_viewer.py to refresh the bundled
reference vectors, then node test_embedding_viewer.cjs to verify PCA, similarities,
checkpoint consistency and import validation. Run python3 test_corpus.py to check
real PDF/TXT/MD extraction, long-text chunking, nested files and useful failure messages.
