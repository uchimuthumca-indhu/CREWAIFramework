import os
from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# 1. Configure LLM Model
# -------------------------------------------------------------------
# CrewAI uses LiteLLM under the hood, supporting OpenAI, Anthropic, Ollama, Groq, etc.
# Make sure your API key is set in your environment:
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Option A: Initialize via CrewAI's LLM wrapper (Recommended for setting temp, params)
gpt4o_llm = LLM(
    model="gpt-4o",
    temperature=0.2
)

# Option B: Pass model string directly or use alternative providers
# Claude: LLM(model="anthropic/claude-3-5-sonnet-20240620")
# Local Ollama: LLM(model="ollama/llama3", base_url="http://localhost:11434")

# -------------------------------------------------------------------
# 2. Method 1: Custom Tool via `@tool` Decorator
# -------------------------------------------------------------------
@tool("Search Internal Database")
def search_database(query: str) -> str:
    """Useful to search internal company records for product pricing or specs.
    
    Args:
        query: The search term or product name to look up.
    """
    mock_db = {
        "enterprise plan": "$499/month, includes 24/7 support",
        "starter plan": "$29/month, up to 5 users"
    }
    return mock_db.get(query.lower(), "No matching record found.")

# Assigning tool AND explicit LLM model to Agent
sales_agent = Agent(
    role="Sales Representative",
    goal="Provide accurate pricing details to incoming leads.",
    backstory="You are an experienced sales agent accessing internal pricing DB.",
    tools=[search_database],
    llm=gpt4o_llm,  # <--- LLM specified here
    verbose=True
)

# -------------------------------------------------------------------
# 3. Method 2: Custom Tool via `BaseTool` Subclass
# -------------------------------------------------------------------
class WeatherInput(BaseModel):
    city: str = Field(..., description="The city name to get weather for.")
    units: str = Field(default="celsius", description="Temperature units: 'celsius' or 'fahrenheit'")

class WeatherTool(BaseTool):
    name: str = "Weather Lookup Tool"
    description: str = "Retrieves live weather conditions for a specified city."
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, city: str, units: str = "celsius") -> str:
        return f"The weather in {city} is 22° ({units})."

weather_tool = WeatherTool()

weather_agent = Agent(
    role="Travel Guide",
    goal="Help users plan trips based on weather.",
    backstory="A helpful travel consultant with live weather insights.",
    tools=[weather_tool],
    llm="gpt-4o-mini",  # <--- Can also pass direct model string name
    verbose=True
)

# -------------------------------------------------------------------
# 4. Execution
# -------------------------------------------------------------------
pricing_task = Task(
    description="Find the price for the enterprise plan.",
    expected_output="The exact monthly price and features of the enterprise plan.",
    agent=sales_agent
)

crew = Crew(
    agents=[sales_agent, weather_agent],
    tasks=[pricing_task]
)

# crew.kickoff()