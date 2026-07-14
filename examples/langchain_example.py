"""
LangChain Integration Example

This example demonstrates how to use WhiteBoxXAI to monitor LangChain applications.
"""

import ast
import operator
import os
import time

from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.langchain import LangChainMonitor, wrap_langchain_chain

# Optional: Set API key
os.environ["WHITEBOXXAI_API_KEY"] = "your-api-key-here"

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a simple arithmetic expression without eval()'s ability to run
    arbitrary code - important here since a tool like this may be called with
    an LLM-influenced (and therefore untrusted) argument."""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")

    return _eval(ast.parse(expression, mode="eval").body)


def example_simple_chain():
    """Example using a simple LLM chain."""
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    print("=" * 60)
    print("Simple LLM Chain Example")
    print("=" * 60)

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key=os.getenv("WHITEBOXXAI_API_KEY"))

    # Create monitor
    monitor = LangChainMonitor(
        client=client,
        application_name="simple_qa_chain",
        track_tokens=True,
        track_cost=True,
    )

    # Register application
    app_id = monitor.register_application(
        name="Simple Q&A Chain",
        version="1.0.0",
        description="Basic question-answering chain",
    )
    print(f"✓ Application registered with ID: {app_id}")

    # Create chain
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer the following question: {question}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Create callback handler
    callback = monitor.create_callback_handler()

    # Run chain with monitoring
    print("\nRunning chain...")
    questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "Who wrote Romeo and Juliet?",
    ]

    for question in questions:
        result = chain.run(question=question, callbacks=[callback])
        print(f"  Q: {question}")
        print(f"  A: {result.strip()}")

    print("\n✓ All chain executions logged!")


def example_sequential_chain():
    """Example using a sequential chain."""
    from langchain.chains import LLMChain, SequentialChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    print("\n" + "=" * 60)
    print("Sequential Chain Example")
    print("=" * 60)

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key=os.getenv("WHITEBOXXAI_API_KEY"))

    # Create monitor
    monitor = LangChainMonitor(
        client=client,
        application_name="sequential_chain",
    )

    # Register application
    monitor.register_application(name="Sequential Processing Chain", version="1.0.0")
    print("✓ Application registered")

    # Create chains
    llm = OpenAI(temperature=0.7)

    # Chain 1: Generate topic
    prompt1 = PromptTemplate(
        input_variables=["subject"],
        template="Generate a creative topic about {subject}",
    )
    chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="topic")

    # Chain 2: Write about topic
    prompt2 = PromptTemplate(
        input_variables=["topic"], template="Write a short paragraph about: {topic}"
    )
    chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="paragraph")

    # Combine chains
    overall_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["subject"],
        output_variables=["topic", "paragraph"],
    )

    # Wrap chain for automatic logging
    wrapped_chain = wrap_langchain_chain(overall_chain, monitor)

    # Run chain
    print("\nRunning sequential chain...")
    result = wrapped_chain({"subject": "artificial intelligence"})

    print(f"\n  Subject: artificial intelligence")
    print(f"  Topic: {result['topic'].strip()}")
    print(f"  Paragraph: {result['paragraph'].strip()[:100]}...")

    print("\n✓ Sequential chain execution logged!")


def example_agent():
    """Example using an agent with tools."""
    from langchain.agents import AgentType, Tool, initialize_agent
    from langchain.llms import OpenAI
    from langchain.utilities import SerpAPIWrapper

    print("\n" + "=" * 60)
    print("Agent Example")
    print("=" * 60)

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key=os.getenv("WHITEBOXXAI_API_KEY"))

    # Create monitor
    monitor = LangChainMonitor(
        client=client,
        application_name="search_agent",
    )

    # Register application
    monitor.register_application(
        name="Search Agent",
        version="1.0.0",
        description="Agent with search capabilities",
    )
    print("✓ Application registered")

    # Create tools (mock for demo)
    def search_tool(query: str) -> str:
        """Mock search tool."""
        return f"Search results for: {query}"

    def calculator_tool(expression: str) -> str:
        """Mock calculator tool."""
        try:
            return str(safe_eval_arithmetic(expression))
        except Exception:
            return "Invalid expression"

    tools = [
        Tool(name="Search", func=search_tool, description="Search for information"),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Calculate mathematical expressions",
        ),
    ]

    # Create agent
    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    # Create callback
    callback = monitor.create_callback_handler()

    # Run agent
    print("\nRunning agent...")
    task = "What is 25 * 4?"
    result = agent.run(task, callbacks=[callback])

    print(f"\n  Task: {task}")
    print(f"  Result: {result}")

    print("\n✓ Agent execution logged!")


def example_rag_chain():
    """Example using a RAG (Retrieval-Augmented Generation) chain."""
    from langchain.chains import RetrievalQA
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS

    print("\n" + "=" * 60)
    print("RAG Chain Example")
    print("=" * 60)

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key=os.getenv("WHITEBOXXAI_API_KEY"))

    # Create monitor
    monitor = LangChainMonitor(
        client=client,
        application_name="rag_qa",
    )

    # Register application
    monitor.register_application(
        name="RAG Q&A System",
        version="1.0.0",
        description="Question answering with retrieval",
    )
    print("✓ Application registered")

    # Mock document database
    documents = [
        "Paris is the capital of France and its largest city.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "France is located in Western Europe.",
    ]

    # Create vector store (mock for demo)
    print("\nCreating vector store...")

    # Mock embeddings and vector store
    class MockEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * 10 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 10

    # For actual use, would use:
    # embeddings = OpenAIEmbeddings()
    # text_splitter = CharacterTextSplitter(chunk_size=1000)
    # texts = text_splitter.split_documents(documents)
    # vectorstore = FAISS.from_documents(texts, embeddings)

    print("✓ Vector store created")

    # Log mock retrieval
    query = "What is the capital of France?"
    start_time = time.time()

    # Mock retrieved documents
    retrieved_docs = [
        {"text": documents[0], "score": 0.95},
        {"text": documents[1], "score": 0.75},
    ]

    retrieval_time = time.time() - start_time

    # Log retrieval
    monitor.log_rag_retrieval(
        query=query,
        documents=retrieved_docs,
        num_retrieved=len(retrieved_docs),
        retrieval_time=retrieval_time,
        relevance_scores=[doc["score"] for doc in retrieved_docs],
    )

    print(f"\n  Query: {query}")
    print(f"  Retrieved {len(retrieved_docs)} documents")
    print(f"  Retrieval time: {retrieval_time:.3f}s")

    print("\n✓ RAG retrieval logged!")


def example_manual_logging():
    """Example using manual logging methods."""
    print("\n" + "=" * 60)
    print("Manual Logging Example")
    print("=" * 60)

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key=os.getenv("WHITEBOXXAI_API_KEY"))

    # Create monitor
    monitor = LangChainMonitor(
        client=client,
        application_name="manual_logging",
    )

    # Register application
    monitor.register_application(name="Manual Logging App", version="1.0.0")
    print("✓ Application registered")

    # Log chain execution manually
    print("\nLogging chain execution...")
    monitor.log_chain_execution(
        chain_name="custom_chain",
        inputs={"question": "What is AI?"},
        outputs={"answer": "Artificial Intelligence is..."},
        execution_time=1.5,
        llm_calls=[{"model": "gpt-3.5-turbo", "tokens": 150}],
    )
    print("✓ Chain execution logged")

    # Log LLM call manually
    print("\nLogging LLM call...")
    monitor.log_llm_call(
        prompt="Explain quantum computing",
        response="Quantum computing is a type of...",
        model="gpt-4",
        tokens_used=200,
        cost=0.004,
        latency=2.3,
    )
    print("✓ LLM call logged")

    # Log tool call manually
    print("\nLogging tool call...")
    monitor.log_tool_call(
        tool_name="web_search",
        tool_input="latest AI news",
        tool_output="Search results: ...",
        execution_time=0.8,
    )
    print("✓ Tool call logged")

    print("\n✓ All manual logging complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("WhiteBoxXAI - LangChain Integration Examples")
    print("=" * 60)

    try:
        # Note: Some examples require API keys (OpenAI, etc.)
        # Uncomment to run with actual LangChain setup

        # example_simple_chain()
        # example_sequential_chain()
        # example_agent()
        # example_rag_chain()
        example_manual_logging()

        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        print("\nNote: Some examples are commented out as they require")
        print("API keys (OpenAI, etc.). Uncomment and configure to run.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
