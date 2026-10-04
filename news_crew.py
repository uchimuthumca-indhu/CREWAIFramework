import os
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

# ------------------------------------------------------------------------------
# 1. Environment & API Keys Verification
# ------------------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is missing.")
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY environment variable is missing.")

# Initialize the search tool
search_tool = SerperDevTool()

# ------------------------------------------------------------------------------
# 2. Agent Definitions
# ------------------------------------------------------------------------------
researcher = Agent(
    role="Tech News Researcher",
    goal="Discover and synthesize the top 3 tech news stories of today.",
    backstory=(
        "You are an expert technology journalist. Your sharp eye for breaking "
        "trends allows you to quickly locate, filter, and extract key details "
        "from major tech news sources."
    ),
    tools=[search_tool],
    verbose=True,
)

writer = Agent(
    role="Tech Newsletter Editor",
    goal="Format raw tech stories into a sharp, concise 5-line newsletter.",
    backstory=(
        "You are a concise editorial copywriter. You specialize in taking complex "
        "research reports and compressing them into punchy, easy-to-read "
        "newsletters without fluff."
    ),
    tools=[],  # Writer explicitly has no search tools
    verbose=True,
)

# ------------------------------------------------------------------------------
# 3. Task Definitions & Output Flow
# ------------------------------------------------------------------------------
research_task = Task(
    description=(
        "Search the web using SerperDevTool to identify today's top 3 tech news stories. "
        "For each story, gather a headline, a 1-2 sentence summary, and key takeaways."
    ),
    expected_output=(
        "A structured bulleted list containing 3 tech stories. Each item must include: "
        "- Headline\n"
        "- Summary\n"
        "- Key Takeaways"
    ),
    agent=researcher,
)

write_task = Task(
    description=(
        "Take the 3 tech news stories provided by the Researcher and write a 5-line newsletter. "
        "The format must consist of:\n"
        "Line 1: A compelling main headline for the newsletter issue.\n"
        "Line 2: A short introduction sentence.\n"
        "Lines 3-5: Exactly 3 distinct bullet points summarizing each of the 3 tech news stories."
    ),
    expected_output=(
        "A newsletter formatted in exactly 5 lines total:\n"
        "1 headline line, 1 intro line, and 3 bullet point lines corresponding to the 3 stories."
    ),
    agent=writer,
    context=[research_task],  # Explicitly passes the researcher's output to the writer
)

# ------------------------------------------------------------------------------
# 4. Crew Configuration & Execution
# ------------------------------------------------------------------------------
tech_news_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = tech_news_crew.kickoff()
    print("\n================ FINAL 5-LINE NEWSLETTER ================\n")
    print(result)