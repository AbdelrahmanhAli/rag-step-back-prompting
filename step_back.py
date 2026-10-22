"""Step-back prompting.

Retrieve with a broader, more conceptual version of the question *as well as*
the original. The step-back query pulls in background material that the specific
question would miss; the original query keeps the answer on target.
"""

from __future__ import annotations

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import core

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Rewrite the user's question as a broader conceptual question that "
                "would help retrieve useful background material. Return only the "
                "question, with no preamble."
            ),
        ),
        ("human", "{question}"),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*")
    parser.add_argument("-k", type=int, default=3, help="chunks per query")
    args = parser.parse_args()

    question = " ".join(args.question).strip() or core.question_from_args(["prog"])
    model = core.chat_model()

    step_back = (
        (STEP_BACK_PROMPT | model | StrOutputParser()).invoke({"question": question}).strip()
    )

    store = core.get_vectorstore()
    broad_hits = store.similarity_search(step_back, k=args.k)
    direct_hits = store.similarity_search(question, k=args.k)

    print(f"Original question : {question}")
    print(f"Step-back question: {step_back}")
    print(f"\nBroad hits : {len(broad_hits)}")
    print(f"Direct hits: {len(direct_hits)}")

    context = core.dedupe([*broad_hits, *direct_hits])
    print(f"Unique context: {len(context)} chunk(s)\n")
    print(core.answer(question, context, model=model, show_sources=True))


if __name__ == "__main__":
    main()
