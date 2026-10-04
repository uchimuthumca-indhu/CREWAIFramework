import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import DallETool, WebsiteSearchTool

# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES FROM .env
# ==========================================
load_dotenv()

# Optional verification check
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found! Check your .env file.")

# ==========================================
# 2. SETUP OPENAI LLM MODEL
# ==========================================
openai_llm = LLM(
    model="gpt-4o-mini",
    temperature=0.3
)

# ==========================================
# 3. INSTANTIATE OPENAI-BASED & CUSTOM TOOLS
# ==========================================

# Tool 1: OpenAI DALL-E Tool (generates images from prompts)
dalle_tool = DallETool()

# Tool 2: OpenAI-backed RAG Web Search Tool
web_rag_tool = WebsiteSearchTool(
    website_url="https://en.wikipedia.org/wiki/Artificial_intelligence"
)

# Tool 3: Custom Python tool using the @tool decorator
@tool("Text Metrics Calculator")
def text_metrics_tool(text: str) -> str:
    """Calculates total word count and estimated reading time for a block of text."""
    words = text.split()
    word_count = len(words)
    reading_time_minutes = round(word_count / 200, 2)
    return f"Word Count: {word_count} | Estimated Reading Time: {reading_time_minutes} min"


# ==========================================
# 4. ASSIGN MULTIPLE TOOLS TO AGENT
# ==========================================

multitask_agent = Agent(
    role="AI Research & Visual Content Strategist",
    goal="Extract key topic information, evaluate metric statistics, and generate visual image prompts",
    backstory=(
        "You are a versatile content analyst equipped with web research tools, "
        "image generation engines, and metric tools to produce complete reports."
    ),
    tools=[
        web_rag_tool,
        text_metrics_tool,
        dalle_tool
    ],
    llm=openai_llm,
    verbose=True
)

# ==========================================
# 5. DEFINE TASK & CREW
# ==========================================

content_task = Task(
    description=(
        "1. Extract the main definition of Artificial Intelligence from the provided site.\n"
        "2. Run the 'Text Metrics Calculator' on the extracted text to get word count and reading time.\n"
        "3. Use the DALL-E Tool to generate a visual depiction based on the extracted definition."
    ),
    expected_output="A structured report containing the text definition, text metrics, and DALL-E image URL.",
    agent=multitask_agent
)

crew = Crew(
    agents=[multitask_agent],
    tasks=[content_task],
    process=Process.sequential,
    verbose=True
)

# ==========================================
# 6. EXECUTION
# ==========================================

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n================ FINAL REPORT ================\n")
    print(result.raw)