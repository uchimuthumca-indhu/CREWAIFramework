import os
import json
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool

# -------------------------------------------------------------------
# 1. Load Environment Variables
# -------------------------------------------------------------------
load_dotenv()

# Verify keys are loaded
if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPER_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY or SERPER_API_KEY in .env file.")

# -------------------------------------------------------------------
# 2. Define Custom Serper Search Tool
# -------------------------------------------------------------------
class SerperSearchInput(BaseModel):
    query: str = Field(..., description="The search term or topic to look up on Google.")


class SerperWebSearchTool(BaseTool):
    name: str = "Serper Web Search"
    description: str = (
        "Queries Google via the Serper API to retrieve current real-time search results, "
        "snippets, titles, and URLs for any topic."
    )
    args_schema: type[BaseModel] = SerperSearchInput

    def _run(self, query: str) -> str:
        """Executes HTTP POST request to Serper API and parses snippets."""
        url = "https://google.serper.dev/search"
        api_key = os.getenv("SERPER_API_KEY")

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = json.dumps({"q": query, "num": 5})

        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Process organic search results into clean markdown text
            organic_results = data.get("organic", [])
            if not organic_results:
                return f"No search results found for query: '{query}'"

            results_summary = []
            for item in organic_results:
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "No Snippet available.")
                link = item.get("link", "")
                results_summary.append(f"### {title}\n**Snippet:** {snippet}\n**URL:** {link}\n")

            return "\n".join(results_summary)

        except requests.exceptions.RequestException as e:
            return f"Failed to perform search: {str(e)}"


# -------------------------------------------------------------------
# 3. Configure LLM Model
# -------------------------------------------------------------------
# Load GPT-4o model instance via CrewAI's native wrapper
gpt4_llm = LLM(
    model="gpt-4o",
    temperature=0.2
)

# -------------------------------------------------------------------
# 4. Instantiate Tool and Agents
# -------------------------------------------------------------------
custom_search_tool = SerperWebSearchTool()

# Agent equipped with the custom search tool
research_agent = Agent(
    role="Senior Market Intelligence Researcher",
    goal="Discover current real-time trends, news, and market data on targeted topics.",
    backstory=(
        "You are an expert analyst skilled at searching the live web to gather "
        "accurate, up-to-date facts and synthesizing them into structured insights."
    ),
    tools=[custom_search_tool],
    llm=gpt4_llm,
    verbose=True
)

writer_agent = Agent(
    role="Technical Content Strategist",
    goal="Transform raw web research into concise, polished executive summaries.",
    backstory="You are a skilled editor who specializes in communicating technology trends clearly.",
    llm=gpt4_llm,
    verbose=True
)

# -------------------------------------------------------------------
# 5. Define Real-Time Tasks
# -------------------------------------------------------------------
search_topic = "latest AI agent frameworks trends and developments"

research_task = Task(
    description=(
        f"Use the search tool to find live information about: '{search_topic}'. "
        "Extract 3-4 key real-time developments, citing sources where applicable."
    ),
    expected_output="A bulleted list of 3-4 current news highlights with context and links.",
    agent=research_agent
)

summary_task = Task(
    description=(
        "Synthesize the research findings into an executive briefing. "
        "Highlight key takeaways, implications, and referenced URLs."
    ),
    expected_output="A clean markdown report with key findings and source references.",
    agent=writer_agent
)

# -------------------------------------------------------------------
# 6. Kickoff Crew Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    crew = Crew(
        agents=[research_agent, writer_agent],
        tasks=[research_task, summary_task]
    )

    print("\n--- Starting Real-Time CrewAI Research Workflow ---\n")
    result = crew.kickoff()
    
    print("\n--- Final Briefing Output ---\n")
    print(result)