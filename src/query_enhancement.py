"""
Query enhancement techniques for improved retrieval (use only one):
- HyDE (Hypothetical Document Embeddings): Generate hypothetical answer for better retrieval
- Query Enrichment: LLM-based query expansion
- Query Rewriting: Expand vague follow-up questions using conversation history
"""

import textwrap
from typing import Optional, List, Dict
from src.generator import ANSWER_END, ANSWER_START, run_llama_cpp, text_cleaning


def generate_hypothetical_document(
    query: str,
    model_path: str,
    max_tokens: int = 100,
    **llm_kwargs
) -> str:
    """
    HyDE: Generate a hypothetical answer to improve retrieval quality.
    Concept: Hypothetical answers are semantically closer to actual documents than queries.
    Ref: https://arxiv.org/abs/2212.10496
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a database systems expert. Generate a concise, technical answer using precise database terminology.
        Write in the formal academic style of Database System Concepts (Silberschatz, Korth, Sudarshan).
        Use specific terms for: relational model concepts (relations, tuples, attributes, keys, schemas), 
        SQL and query languages, transactions (ACID properties, concurrency control, recovery), 
        storage structures (indexes, B+ trees), normalization (functional dependencies, normal forms), 
        and database design (E-R model, decomposition).
        Focus on definitions, mechanisms, and technical accuracy rather than examples.
        <|im_end|>
        <|im_start|>user
        Question: {query}
        
        Generate a precise answer (2-4 sentences) using appropriate technical terminology. End with {ANSWER_END}.
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
        """)
    
    prompt = text_cleaning(prompt)
    hypothetical = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        **llm_kwargs
    )
    
    return hypothetical.strip()


def rewrite_query_with_history(
    query: str,
    conversation_history: List[Dict[str, str]],
    model_path: str,
    max_tokens: int = 50,
    **llm_kwargs
) -> str:
    """
    Rewrite a potentially vague follow-up query into a standalone query using conversation history.
    """
    if not conversation_history:
        return query

    history_text = ""
    for entry in conversation_history[-2:]:
        history_text += f"Q: {entry['question']}\nA: {entry['answer']}\n\n"

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a query rewriting assistant. Rewrite the follow-up question to be a standalone query that includes necessary context from the conversation history.
        Replace pronouns (it, that, they) with specific terms. Expand references (the first one, that property) with actual names.
        Keep the rewritten query concise but self-contained.
        <|im_end|>
        <|im_start|>user
        Conversation history:
        {history_text}
        Follow-up question: {query}

        Rewrite this as a standalone question with full context. Output only the rewritten question, nothing else. End with {ANSWER_END}.
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
        """)

    prompt = text_cleaning(prompt)
    rewritten = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.1,
        **llm_kwargs
    )

    return rewritten.strip() if rewritten.strip() else query
