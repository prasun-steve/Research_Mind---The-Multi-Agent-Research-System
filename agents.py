from langchain.agents import create_agent
from tools import web_search, scrape_url
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_hf = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
    max_new_tokens=2048,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm_hf)


# 1st agent
def build_search_agent():
    return create_agent(
        model=model,
        tools=[web_search],
    )


# 2nd agent
def build_reader_agent():
    return create_agent(
        model=model,
        tools=[scrape_url],
    )

#writer_chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an elite research writer.

You convert raw research notes into publication-quality reports.

Rules:
1. Never hallucinate.
2. Never cite information absent from the provided research.
3. Prefer synthesis over summarization.
4. Explain significance, not just facts.
5. Preserve nuance and uncertainty.
6. Maintain academic and professional tone.
"""
    ),
    (
        "human",
        """
Topic:
{topic}

Research Corpus:
{research}

Generate a report with the following sections:

# Executive Summary

# Background

# Major Findings
(At least three substantial findings)

For each finding include:
- Description
- Supporting Evidence
- Interpretation
- Practical Implications

# Emerging Trends
(Optional)

# Risks or Challenges
(Optional)

# Limitations of Available Evidence

# Conclusion

# Sources

Constraints:
- Use markdown headings.
- Do not repeat information.
- Exclude unsupported claims.
- Explicitly flag uncertain conclusions.
- Include every URL found in the research corpus.
"""
    )
])

parser = StrOutputParser()

writer_chain = writer_prompt | model | parser

#critic chain

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research reviewer with high standards.

Your role is to critically evaluate research reports for:
- Accuracy
- Completeness
- Clarity
- Logical consistency
- Depth of analysis
- Evidence quality
- Structure and organization
- Actionability of conclusions

Be constructive but uncompromising.
Identify weaknesses explicitly rather than being overly polite.
Point out missing evidence, unsupported claims, shallow analysis,
or gaps in reasoning.

Scores should reflect the true quality of the report:
90-100 = exceptional
80-89  = strong with minor gaps
70-79  = acceptable but needs improvement
60-69  = weak
Below 60 = poor and requires major revision.
"""
    ),
    (
        "human",
        """Review the following research report rigorously.

Report:
{report}

Evaluate the report on:

1. Accuracy
2. Coverage and completeness
3. Depth of analysis
4. Quality of evidence
5. Logical coherence
6. Clarity and readability
7. Structure and organization
8. Practical usefulness

Respond ONLY in the following format:

Score: X/100

Strengths:
- ...
- ...
- ...

Areas to Improve:
- ...
- ...
- ...

Missing Elements:
- ...
- ...

One line verdict:
...

Suggested Next Steps:
- ...
- ...
"""
    )
])

critic_chain = critic_prompt | model | parser
