import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ── LLM ───────────────────────────────────────────────────────────────────────
# Provider : Groq (free tier — sign up at console.groq.com)
# Model    : llama-3.3-70b-versatile
# Context  : 128 000 tokens
# Speed    : ~400 tokens/sec (fastest free LLM API)
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2          # Lower = more faithful to document facts
LLM_MAX_TOKENS = 1024


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict document-based question answering system.

You MUST follow these rules:

1. Answer ONLY using the provided context.
2. DO NOT use prior knowledge.
3. DO NOT guess or infer beyond the context.
4. If the answer is not explicitly present, say:
    "I couldn't find a clear answer in the uploaded document."

5. If context is insufficient, DO NOT attempt to complete the answer.

6. Always cite page numbers if available.

Context:
{context}
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the conversation history and the user's latest question,
rephrase the question to be a self-contained query that captures the full intent.
Do NOT answer the question; only rephrase it.

Chat History:
{chat_history}

Latest Question: {question}

Rephrased Question:"""
)


def build_qa_chain(retriever, memory: ConversationBufferWindowMemory) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain that:
    1. Condenses follow-up questions using chat history.
    2. Retrieves relevant chunks from ChromaDB.
    3. Synthesises an answer using the LLM.
    """
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": qa_prompt},    
        return_source_documents=True,
        verbose=False,
    )
    return chain


def create_memory(session_id: str, window_size: int = 5) -> ConversationBufferWindowMemory:
    """Create a sliding-window conversation memory (last k turns)."""
    return ConversationBufferWindowMemory(
        k=window_size,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
