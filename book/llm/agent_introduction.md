# Introduction to AI Agents 

<p align="center">
  <img src="https://surfdrive.surf.nl/files/index.php/s/5j3QFEPoym6bKdG/preview" alt="Augmented LLM Illustration"  width="800">
</p>

## Limitations of LLMs
Large Language Models (LLMs) can already be effective assistants: they can explain concepts, draft text, summarize documents, and help you reason through problems. But an LLM **by itself** has major limitations, especially when tasks require *reliable execution* rather than *plausible language*.

A classic example is **computation**. Even basic arithmetic can go very wrong. The reason is that an LLM primarily learns patterns in text and produces the next token that seems most likely given its context. It does not “run” mathematics as a formal algorithm, and it has not encountered (and cannot store) all possible computations that could exist. So, even when it sounds confident, it can still be very wrong.

A straightforward workaround is to let the LLM **use a calculator**. Instead of forcing the model to perform the computation internally, we let it write a *tool call* (structured text) describing:
- what operation is needed,
- what the inputs are,
- and how to return the result.

This already goes beyond “an LLM answering questions”: it is the first step toward an **agent**.

## What is an agent?
An **agent** is essentially an LLM plus an **orchestration layer** that enables it to **use tools and resources** to solve tasks that the LLM alone would struggle with. In practice, this orchestration layer provides at least three things:

1. **Tool use**  
   The agent can call external tools (e.g., calculators, Python, web search, databases) and incorporate the results into its final answer.

2. **Planning and decomposition**  
   The agent can break a complex task into smaller steps, choose appropriate tools for each step, and execute them in sequence.

3. **Iteration and checking**  
   The agent can verify intermediate results (sanity checks, constraints, self-tests), revise its plan when needed, and only then produce a final output.

This combination is what turns an LLM from a “chatbot that responds” into a system that can **execute multi-step workflows**, which is why agents are often described by AI enthusiasts as "unlocking the next level of automation for complex knowledge work and beyond".

> **A word of caution:** There is considerable hype surrounding AI agents, and they are certainly fascinating-sometimes demonstrating surprisingly capable behavior. However, it remains unclear whether this paradigm represents a sustainable path forward for reliable and economically viable automation of knowledge workers' tasks. Current agents can be brittle, expensive to run at scale, and prone to compounding errors across multi-step tasks. As with any emerging technology, a healthy dose of skepticism is warranted alongside the excitement.

## You might already be using agents

Many modern AI features behave like agents under the hood, even if they don’t explicitly call themselves that.

**Deep Research** is a good example. In practice, it typically:
- searches across many sources, then iteratively refines queries as it learns what’s relevant,  
- extracts and cross-checks claims across references rather than relying on a single page,  
- synthesizes a structured answer with traceable evidence rather than relying purely on model memory.  

Similarly, if you code with **Visual Studio Code + Copilot** (or similar coding platforms like Cursor or Claude Code), you’ve probably seen agent-like behavior: the assistant can propose changes across multiple files, navigate a codebase, run or suggest commands, and iteratively fix issues.

---



<p align="center">
  <img src="https://surfdrive.surf.nl/files/index.php/s/gYB7XL9o8E6zBg6/preview" alt="LLM using multiple tools and resources" width="800">
</p>

## What tools and resources can we give an agent?

An agent becomes more capable (and more practical) when it can interact with the same kinds of resources you use for real work. Below are common tools and what they enable:

### Web search and browsing

Agents can query the web to fetch information that is:
- too recent to be in the model’s training data,
- niche or domain-specific,
- or simply too detailed to trust from memory.

A well-designed agent doesn’t just “search once”; it can **iterate**:
search → read → refine query → read again → synthesize. This helps reduce confident but outdated answers and supports evidence-based "reasoning".


### Retrieval-Augmented Generation (RAG)

RAG allows the agent to retrieve relevant content from a curated knowledge source, e.g., your own documents, lecture notes, reports, manuals, or internal knowledge bases, then use that content while generating an answer.

Typical RAG workflow:
- documents are stored in an index (often via embeddings),
- relevant passages are retrieved for a given question and addded to the context of the LLM,
- the model generates an answer grounded in those passages.

RAG aligns the agent with *your* materials and reduces hallucinations by grounding responses in trusted sources.


### Python interface

Giving the agent access to Python allows it to:
- run numeric computations reliably,
- manipulate datasets and dataframes,
- fit models and evaluate metrics,
- generate plots and figures,
- implement algorithms rather than merely describing them,
- ...

Python allows the agent to analyze data, automate tasks, and generate concrete outputs, from calculations and visualizations to running code and working with real datasets.


### Shell / command-line access

Shell access allows an agent to:
- inspect folders and files,
- run scripts,
- install dependencies,
- execute CLI tools (linters, tests, build tools),
- automate repetitive workflows.

Shell access lets agents automate complex workflows, inspect and modify environments, and connect existing command-line tools.

> **Safety note:**  Shell access is powerful enough to cause real damage (e.g., deleting files, modifying systems). That is why it should normally be provided only in a **sandboxed environment** with restricted permissions, limited access, and careful monitoring.


### Databases and data services

Agents can be connected to databases to:
- query structured data,
- run analytics,
- generate reports,
- monitor operational systems.

Agents can leverage structured data (such as SQL or time-series databases, or APIs that expose data services) to deliver analytics, insights, or automate reporting. This must be done by enforcing controlled access through safe interfaces rather than granting unrestricted credentials, you protect sensitive data and ensure both reliability and security.


### File systems and document editors

Agents can also work with structured file tools to:
- create and edit files (code, configs, markdown, LaTeX),
- update notebooks,
- refactor projects across many files,
- generate reports automatically.

This is where agents begin to resemble real collaborators rather than chat interfaces.


### Specialized tools and APIs

Beyond generic tools, agents can also use domain-specific systems, such as:
- simulation engines,
- GIS software,
- ML experiment trackers,
- cloud services,
- engineering analysis tools,
- internal institutional platforms.

If you can build a structured interface, an agent can theoretically to use it. 

---

## Chain of Thought and Reasoning in Agents

One of the key ideas behind modern agentic systems is that good performance does not come only from better tools, but also from better internal **"reasoning"** strategies of the LLM itself. **Chain of Thought (CoT)** refers to the idea that a model improves its problem-solving by generating intermediate reasoning steps rather than jumping directly to an answer.  

Instead of simply producing:

> “Here is the answer.”

the model effectively follows a process closer to:

> “First I consider X, then Y, then Z, therefore the answer is …”

These intermediate steps help the model structure its reasoning, decompose complex problems, and avoid shallow pattern-matching. CoT improves performance on tasks requiring multi-step reasoning, planning, and logical consistency-making it particularly valuable for **agents** that must decide which tools to use, when to use them, and whether results make sense.

Recent **reasoning models** are explicitly fine-tuned (e.g., via **RLHF**) to use structured reasoning and can adaptively allocate more computation to harder problems. This makes them well suited for agents, where some steps are trivial while others require careful multi-step thinking.

When combined with tool use, CoT enables agents to plan before acting, reflect on intermediate outputs, and revise their strategy when needed-the difference between reacting to each step in isolation and following a coherent reasoning trajectory across an entire task.

---

## Measuring Real-World Task Performance

As AI agents capabilities grow, evaluations increasingly aim to measure performance on **realistic knowledge work**, such as analyzing documents, planning complex workflows, conducting multi-step research, or solving open-ended technical problems. Designing meaningful benchmarks for this kind of real-world work is extremely difficult, yet substantial progress has been made in **software engineering**, which is arguably the domain where AI agents are currently having the most tangible and transformative impact. 

In this context, initiatives such as [METR](https://metr.org) have introduced approaches like the **time-horizon metric**, which measures how long a model can reliably complete tasks compared to human effort. Rather than evaluating isolated answers, this framework captures whether an AI can sustain coherent performance over extended, multi-step workflows. Results suggest that the duration of tasks frontier models can complete with around 50% success has been growing rapidly over time, providing a more faithful signal of real-world usefulness than traditional short-form benchmarks.

<p align="center">
  <img src="https://surfdrive.surf.nl/s/ktGoFLpPEBNdLnt/preview" alt="Agentic Capabilities Illustration" width=1000>
</p>


However, when moving from software engineering to domains such as **civil engineering, environmental engineering, and applied earth sciences**, the challenge becomes significantly harder. Tasks are less standardized, more context-dependent, and harder to simulate, and we currently lack comparable, high-quality benchmarks for evaluation.

## Reliability in agentic workflows

Even if an agent has a **99%** chance of solving *each individual task* correctly, the probability that it completes an entire project with **N** tasks **without a single mistake** is `P(all correct) = 0.99^N`

Example (task success rate = 99%):

* 10 tasks:
  `0.99^10 ≈ 0.904`
  **≈ 90.4%** chance everything goes right (**≈ 9.6%** chance at least one failure)

* 50 tasks:
  `0.99^50 ≈ 0.605`
  **≈ 60.5%** chance everything goes right (**≈ 39.5%** chance at least one failure)

* 100 tasks:
  `0.99^100 ≈ 0.366`
  **≈ 36.6%** chance everything goes right (**≈ 63.4%** chance at least one failure)


In practice, multi-step projects are often even more fragile than this simple calculation suggests: tasks are not fully independent, early mistakes can cascade, and many steps involve ambiguity or hidden assumptions. That’s why “high per-task accuracy” can still translate into unreliable end-to-end performance. Robust verification, checkpoints, and human oversight are still essential.

## Multi-Agent Systems

A **multi-agent system** consists of multiple agents that interact to solve a problem collaboratively rather than relying on a single monolithic agent. Because individual agents are imperfect and can fail, especially for long and complex tasks, overloading one agent with too many responsibilities can increase the risk of errors and brittle behavior. Distributing tasks across multiple agents allows each agent to focus on a narrower role, which can statistically reduce failure rates and improve overall reliability. Agents can communicate by exchanging messages, sharing intermediate results, and checking each other’s outputs, enabling redundancy and cross-validation.  
However, multi-agent systems are also **more complex to design, debug, and control**, and they introduce coordination overhead and costs that can outweigh the benefits.   For this reason, they should be used deliberately: not because they look sophisticated, but only when task decomposition and redundancy genuinely improve performance.

---

## Limits of Current Automation

It is tempting to extrapolate early wins into “full automation”, but in the **vast majority of fields we are still far from reliable, general automation**. Real work is messy: requirements shift, data quality varies, edge cases dominate, and accountability matters. Treat agents as **assistive systems** that can accelerate parts of a workflow, not as autonomous replacements.

Also note that today’s LLM usage often feels “cheap” because the market is in a **high-growth, high-expectation phase**: pricing can be effectively subsidized by competition and investment. This may not last. If models do not become **good enough at sufficiently low cost** to be profitable, the economics can tighten abruptly (prices up, limits down, products discontinued, startups collapsing). **So be careful**: avoid building critical workflows that depend on brittle capabilities or unsustainably cheap model access without a fallback plan.