import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM

# 1. Load Environment Variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file.")

# 2. Configure LLM Model
gpt4_llm = LLM(
    model="gpt-4o",
    temperature=0.2
)

# 3. Define Agents
customer_researcher = Agent(
    role="Customer Preferences Researcher",
    goal="Extract and remember user preferences, interests, and constraints from conversation.",
    backstory=(
        "You are an empathetic customer intelligence agent specializing in building detailed "
        "profiles of client requirements."
    ),
    llm=gpt4_llm,
    verbose=True
)

recommendation_agent = Agent(
    role="Personalized Travel Advisor",
    goal="Provide tailored travel recommendations using remembered client preferences.",
    backstory=(
        "You are an expert travel agent who utilizes stored customer memory to curate "
        "perfect custom itineraries."
    ),
    llm=gpt4_llm,
    verbose=True
)

# 4. Define Tasks
task1 = Task(
    description=(
        "Analyze the following user profile note: "
        "'Client: Sarah, Budget: $3,000, Preferences: Loves beach destinations, strictly vegetarian, "
        "allergic to peanuts, prefers quiet boutique hotels rather than large resorts.' "
        "Extract key entities and user traits."
    ),
    expected_output="A structured summary of Sarah's travel preferences, dietary constraints, and budget.",
    agent=customer_researcher
)

task2 = Task(
    description=(
        "Based on Sarah's remembered profile from memory, recommend 2 destination options in Southeast Asia "
        "and outline how they accommodate her budget, dietary needs, and hotel preferences."
    ),
    expected_output="A tailored recommendation brief addressing budget, diet, and hotel preference.",
    agent=recommendation_agent
)

# 5. Build Crew with Memory Enabled
memory_crew = Crew(
    agents=[customer_researcher, recommendation_agent],
    tasks=[task1, task2],
    # Enable Memory System
    memory=True,
    # Optional: Configure custom memory embedder (Defaults to OpenAI text-embedding-3-small)
    embedder={
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    verbose=True
)

# 6. Kickoff Execution
if __name__ == "__main__":
    print("\n--- Starting CrewAI Execution with Memory ---\n")
    result = memory_crew.kickoff()

    print("\n--- Final Recommendation Output ---\n")
    print(result)