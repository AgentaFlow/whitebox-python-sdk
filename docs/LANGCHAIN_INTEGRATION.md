# LangChain Integration Guide

This guide demonstrates how to use WhiteBoxXAI to monitor LangChain applications including chains, agents, and RAG pipelines.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Installation

Install WhiteBoxXAI SDK with LangChain support:

```bash
pip install whiteboxxai langchain
```

For specific LLM providers, install additional dependencies:

```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# Hugging Face
pip install transformers
```

## Quick Start

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.langchain import LangChainMonitor

# Initialize client
client = WhiteBoxXAI(api_key="your-api-key")

# Create monitor
monitor = LangChainMonitor(
    client=client,
    application_name="qa_bot",
    track_tokens=True,
    track_cost=True
)

# Register application
monitor.register_application(
    name="Q&A Bot",
    version="1.0.0"
)

# Create callback handler
callback = monitor.create_callback_handler()

# Create and run chain
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Run with monitoring
result = chain.run(question="What is AI?", callbacks=[callback])
```

## Core Concepts

### LangChainMonitor

Main class for monitoring LangChain applications. Tracks:
- Chain executions
- Agent runs
- LLM calls
- Tool usage
- RAG retrievals

### WhiteBoxXAICallbackHandler

LangChain callback handler that automatically logs events to WhiteBoxXAI.

### Supported Components

- **Chains**: LLMChain, SequentialChain, SimpleSequentialChain, etc.
- **Agents**: Zero-shot, ReAct, Conversational, etc.
- **Tools**: Search, Calculator, custom tools
- **Memory**: ConversationBufferMemory, etc.
- **Retrievers**: Vector store retrievers, web search

## Basic Usage

### Method 1: Callback Handler (Recommended)

```python
from whiteboxxai.integrations.langchain import LangChainMonitor

# Create monitor
monitor = LangChainMonitor(client, application_name="my_app")
monitor.register_application(name="My App")

# Create callback
callback = monitor.create_callback_handler()

# Use with any LangChain component
chain.run(input="...", callbacks=[callback])
agent.run(input="...", callbacks=[callback])
```

### Method 2: Wrap Chain

```python
from whiteboxxai.integrations.langchain import wrap_langchain_chain

# Wrap chain for automatic logging
wrapped_chain = wrap_langchain_chain(chain, monitor)

# All executions automatically logged
result = wrapped_chain.run(input="...")
```

### Method 3: Manual Logging

```python
# Log chain execution manually
monitor.log_chain_execution(
    chain_name="my_chain",
    inputs={"question": "What is AI?"},
    outputs={"answer": "AI is..."},
    execution_time=1.5
)

# Log LLM call
monitor.log_llm_call(
    prompt="Explain AI",
    response="AI explanation...",
    model="gpt-3.5-turbo",
    tokens_used=100,
    latency=0.8
)
```

## Advanced Features

### Simple Chain

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Create chain
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a haiku about {topic}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Create callback
callback = monitor.create_callback_handler()

# Run with monitoring
result = chain.run(topic="artificial intelligence", callbacks=[callback])
```

### Sequential Chain

```python
from langchain.chains import LLMChain, SequentialChain

# Chain 1: Generate topic
prompt1 = PromptTemplate(
    input_variables=["subject"],
    template="Generate a creative topic about {subject}"
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="topic")

# Chain 2: Write about topic
prompt2 = PromptTemplate(
    input_variables=["topic"],
    template="Write a paragraph about: {topic}"
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="paragraph")

# Combine chains
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["subject"],
    output_variables=["topic", "paragraph"]
)

# Wrap for automatic logging
wrapped = wrap_langchain_chain(overall_chain, monitor)

# Run
result = wrapped({"subject": "space exploration"})
```

### Agent with Tools

```python
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.llms import OpenAI

# Define tools
def search_tool(query: str) -> str:
    # Your search implementation
    return f"Search results for: {query}"

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search for information"
    )
]

# Create agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run with callback
callback = monitor.create_callback_handler()
result = agent.run("Find information about quantum computing", callbacks=[callback])
```

### RAG (Retrieval-Augmented Generation)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# Create vector store
documents = ["...", "...", "..."]
text_splitter = CharacterTextSplitter(chunk_size=1000)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Run with callback
callback = monitor.create_callback_handler()
result = qa_chain.run("What is the main topic?", callbacks=[callback])

# Log retrieval separately
monitor.log_rag_retrieval(
    query="What is the main topic?",
    documents=retrieved_docs,
    num_retrieved=len(retrieved_docs),
    retrieval_time=0.5,
    relevance_scores=[0.95, 0.85, 0.75]
)
```

### Conversational Agent with Memory

```python
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create agent with memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Run multiple turns with callback
callback = monitor.create_callback_handler()

result1 = agent.run("My name is Alice", callbacks=[callback])
result2 = agent.run("What's my name?", callbacks=[callback])
```

### Token and Cost Tracking

```python
# Enable tracking
monitor = LangChainMonitor(
    client=client,
    application_name="cost_tracker",
    track_tokens=True,
    track_cost=True
)

# Tokens and costs are automatically tracked
callback = monitor.create_callback_handler()
result = chain.run(input="...", callbacks=[callback])

# Access metrics via WhiteBoxXAI dashboard
```

### Custom Metadata

```python
# Add custom metadata to logs
monitor.log_chain_execution(
    chain_name="custom_chain",
    inputs={"query": "..."},
    outputs={"response": "..."},
    execution_time=1.2,
    user_id="user_123",
    session_id="sess_456",
    environment="production"
)
```

## Integration Patterns

### Pattern 1: Global Callback

```python
# Create single callback for entire application
callback = monitor.create_callback_handler()

# Use with all chains/agents
chain1.run(input="...", callbacks=[callback])
chain2.run(input="...", callbacks=[callback])
agent.run(input="...", callbacks=[callback])
```

### Pattern 2: Per-Component Monitoring

```python
# Different monitors for different components
qa_monitor = LangChainMonitor(client, application_name="qa_system")
qa_monitor.register_application(name="Q&A")
qa_callback = qa_monitor.create_callback_handler()

agent_monitor = LangChainMonitor(client, application_name="agent_system")
agent_monitor.register_application(name="Agent")
agent_callback = agent_monitor.create_callback_handler()

# Use specific callbacks
qa_chain.run(input="...", callbacks=[qa_callback])
agent.run(input="...", callbacks=[agent_callback])
```

### Pattern 3: Hybrid Approach

```python
# Use callback for automatic logging
callback = monitor.create_callback_handler()
result = chain.run(input="...", callbacks=[callback])

# Add manual logging for custom metrics
monitor.log_custom_metric("user_satisfaction", {"score": 4.5})
```

## Best Practices

### 1. Register Applications Early

```python
# ✅ Good - Register once at startup
monitor = LangChainMonitor(client, application_name="my_app")
monitor.register_application(name="My App", version="1.0.0")

callback = monitor.create_callback_handler()

# Use throughout application
for query in queries:
    chain.run(input=query, callbacks=[callback])
```

### 2. Reuse Callback Handlers

```python
# ✅ Good - Create callback once
callback = monitor.create_callback_handler()

# Reuse for multiple calls
for i in range(100):
    chain.run(input=f"Query {i}", callbacks=[callback])
```

### 3. Handle Errors Gracefully

```python
try:
    result = chain.run(input="...", callbacks=[callback])
except Exception as e:
    print(f"Chain failed: {e}")
    # Callback still logs what it captured
```

### 4. Monitor Different Application Stages

```python
# Development
dev_monitor = LangChainMonitor(client, application_name="app_dev")

# Staging
staging_monitor = LangChainMonitor(client, application_name="app_staging")

# Production
prod_monitor = LangChainMonitor(client, application_name="app_prod")
```

### 5. Track Costs for Budget Management

```python
monitor = LangChainMonitor(
    client=client,
    application_name="cost_aware_app",
    track_tokens=True,
    track_cost=True  # Enable cost tracking
)

# Costs automatically tracked per LLM call
```

## Troubleshooting

### Issue: "langchain is not installed"

**Solution**: Install LangChain:
```bash
pip install langchain
```

### Issue: Callbacks not working

**Solution**: Ensure callbacks are passed correctly:
```python
# ✅ Correct
chain.run(input="...", callbacks=[callback])

# ❌ Wrong
chain.run(input="...", callback=callback)  # Missing 's'
```

### Issue: Missing token counts

**Solution**: Ensure LLM provider returns token usage:
```python
# Some providers require explicit configuration
llm = OpenAI(model_kwargs={"logprobs": True})
```

### Issue: High overhead

**Solution**: Reduce logging frequency or disable detailed tracking:
```python
monitor = LangChainMonitor(
    client=client,
    application_name="app",
    track_tokens=False,  # Disable if not needed
    track_cost=False
)
```

### Issue: Callback conflicts with other callbacks

**Solution**: Combine multiple callbacks:
```python
# Use multiple callbacks together
my_callback1 = MyCallback()
my_callback2 = MyCallback()
wb_callback = monitor.create_callback_handler()

chain.run(input="...", callbacks=[my_callback1, my_callback2, wb_callback])
```

## Examples

See complete examples in:
- `sdk/examples/langchain_example.py` - Comprehensive examples
- `sdk/examples/notebooks/` - Jupyter notebooks

## API Reference

### LangChainMonitor

Main class for monitoring LangChain applications.

**Methods**:
- `register_application()` - Register application with WhiteBoxXAI
- `create_callback_handler()` - Create callback handler
- `log_chain_execution()` - Log chain execution
- `log_agent_execution()` - Log agent run
- `log_llm_call()` - Log LLM call
- `log_tool_call()` - Log tool usage
- `log_rag_retrieval()` - Log RAG retrieval

### WhiteBoxXAICallbackHandler

LangChain callback handler for automatic logging.

**Tracked Events**:
- Chain start/end/error
- Agent actions/finish
- LLM start/end/error
- Tool start/end/error

### wrap_langchain_chain()

Function to wrap chains for automatic logging.

## Support

For issues or questions:
- GitHub Issues: https://github.com/whiteboxxai/whiteboxxai
- Documentation: https://docs.whiteboxxai.com
- Email: support@whiteboxxai.com
