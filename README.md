## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/pdf?id=MS9nWFY7LG",
    "https://openreview.net/pdf?id=CCSPm6V5EF",
    "https://openreview.net/pdf?id=M7TNf5J26u",
]

papers = [
    "Q_RAG_Long_Context_Multi.pdf",
    "WebDevJudge_Evaluating_M.pdf",
    "AstaBench_Rigorous_Bench.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the WEBDEVJUDGE benchmark dataset, and then explain how it is used to evaluate LLM-as-a-judge performance."
)
response = agent.query("Give me a summary of both Q-RAG and AstaBench.")
print(str(response))
```

### OUTPUT:
![image alt](https://github.com/Sanjeev23062007/genai-multidoc-retrieval/blob/85d0807ace0b6d6f2fe813048ae0965e2a0fe9f8/Screenshot%202026-03-15%20200123.png)
![image alt](https://github.com/Sanjeev23062007/genai-multidoc-retrieval/blob/30bb0f3352107fc5306f1aa8eea2ff86ac771ca5/Screenshot%202026-03-15%20201440.png)
![image alt](https://github.com/Sanjeev23062007/genai-multidoc-retrieval/blob/30bb0f3352107fc5306f1aa8eea2ff86ac771ca5/Screenshot%202026-03-15%20201558.png)

### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.    

