import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM
from crewai_tools import SerperDevTool

# -------------------------------------------------------------------
# 1. Load Environment Variables from .env File
# -------------------------------------------------------------------
load_dotenv()

# Verify required keys are set
if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPER_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY or SERPER_API_KEY in environment or .env file.")

# -------------------------------------------------------------------
# 2. Configure LLM Model
# -------------------------------------------------------------------
# Load GPT-4o model instance using CrewAI's LLM class
llm_model = LLM(
    model="gpt-4o",
    temperature=0.2
)

# -------------------------------------------------------------------
# 3. Instantiate SerperDevTool
# -------------------------------------------------------------------
# SerperDevTool automatically detects SERPER_API_KEY from environment variables
web_search_tool = SerperDevTool()

# -------------------------------------------------------------------
# 4. Define Agents
# -------------------------------------------------------------------
researcher_agent = Agent(
    role="Senior Market Intelligence Researcher",
    goal="Gather live, real-time web intelligence and news on given subjects.",
    backstory=(
        "You are an expert market analyst who specializes in searching the live web "
        "for accurate, up-to-date facts using Google search results."
    ),
    tools=[web_search_tool],  # Attached SerperDevTool here
    llm=llm_model,
    verbose=True
)

writer_agent = Agent(
    role="Tech Industry Analyst",
    goal="Synthesize raw search data into executive-ready markdown briefs.",
    backstory=(
        "You take key insights and search outputs and structure them into "
        "clear, well-formatted technical summaries."
    ),
    llm=llm_model,
    verbose=True
)

# -------------------------------------------------------------------
# 5. Define Tasks
# -------------------------------------------------------------------
search_topic = "latest developments in agentic AI frameworks"

research_task = Task(
    description=(
        f"Search the web for real-time information on: '{search_topic}'. "
        "Find 3-4 significant recent updates, news highlights, or releases."
    ),
    expected_output="A list of 3-4 recent news highlights with source context.",
    agent=researcher_agent
)

write_task = Task(
    description=(
        "Review the findings from the research task and construct a concise, "
        "well-organized executive brief in Markdown format."
    ),
    expected_output="A structured markdown report detailing trends and key highlights.",
    agent=writer_agent
)

# -------------------------------------------------------------------
# 6. Kickoff Crew Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    crew = Crew(
        agents=[researcher_agent, writer_agent],
        tasks=[research_task, write_task]
    )

    print("\n--- Starting Research Execution ---\n")
    result = crew.kickoff()

    print("\n--- Final Generated Report ---\n")
    print(result)