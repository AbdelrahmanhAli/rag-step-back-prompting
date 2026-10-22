# Step-Back Prompting

Retrieve with a broader conceptual question as well as the literal one, so background context is not missed.

**Part of the [RAG Techniques](https://github.com/AbdelrahmanhAli/rag-techniques-collection) series — 15 self-contained LangChain v1 demos.**

---

## What it demonstrates

Specific questions often retrieve specific passages but miss the
background needed to interpret them. Step-back prompting asks the model to
generalise first:

```
"What does the 120-character overlap do in this index?"
        │
        ├── step back ──► "How does chunk overlap affect retrieval quality?"
        │
        ├── search(original) ──► precise passages
        └── search(step-back) ──► background passages
                                   │
                              dedupe + union
                                   ▼
                                 answer
```

The literal query keeps the answer on target; the broader query supplies the
conceptual frame. Both sets are deduplicated before generation, so overlapping
hits do not crowd out the prompt.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your OpenAI key
```

Then run:

```bash
python step_back.py "what does the 120-character overlap actually do?"
python step_back.py "why did my retrieval return four near-identical chunks?" -k 4
```

`python <script> --help` lists the available flags. The first run builds a
persistent index in `./.chroma`; later runs reuse it. Delete that folder (or use
`--rebuild` where supported) after changing anything in `data/`.

## Project layout

```
.
├── step_back.py
├── core.py
├── conftest.py
├── data/
│   └── sample.txt
├── tests/
│   └── test_core.py
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── LICENSE
```

## Tests

```bash
pip install pytest
pytest
```

The tests run entirely offline — no API key and no network. They cover the
shared helpers and the technique-specific logic, and use a deterministic fake
embedding model for the vector-store path.

## Notes

- One extra LLM call for the rewrite; retrieval cost roughly doubles.
- Helpful for 'why' and 'how' questions, less useful for factual lookups where a single passage is enough.
- The generated step-back question is shown before the answer so you can sanity-check the direction it took.
- Combines well with multi-query retrieval: expand *and* step back, then fuse.

## References

- [Take a Step Back: Evoking Reasoning via Abstraction (Zheng et al., 2023)](https://arxiv.org/abs/2310.06117)
- [LangChain v1 overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain integrations reference](https://reference.langchain.com/python/integrations/overview)
- [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)

## License

MIT — see [LICENSE](LICENSE).
