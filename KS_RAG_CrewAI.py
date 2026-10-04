import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# 1. Load Environment Variables (.env)
load_dotenv()

# Verify API key availability
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing from environment variables.")

# 2. Define the Knowledge Source (RAG File Loader)
# CrewAI chunks, embeds, and stores this file into a local vector DB automatically.
content_source = TextFileKnowledgeSource(
    file_paths=["../company_policy.txt"],
    chunk_size=1000,
    chunk_overlap=100
)

# 3. Configure the LLM
# Explicitly initialize the LLM (defaults to OpenAI gpt-4o-mini or gpt-4o)
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.2
)

# 4. Define the Agent
# The agent is linked to the knowledge source via the `knowledge_sources` parameter.
hr_agent = Agent(
    role="HR Support Specialist",
    goal="Accurately answer employee inquiries based strictly on company policy documentation.",
    backstory=(
        "You are a helpful and meticulous HR specialist. "
        "You always ground your answers directly in the provided official policy sources "
        "and never hallucinate policies that do not exist."
    ),
    knowledge_sources=[content_source],
    llm=llm,
    verbose=True
)

# 5. Define the Task
policy_task = Task(
    description=(
        "Answer the user's inquiry regarding remote work and equipment stipends: "
        "'{user_query}'. Extract specific rules, limits, or hours mentioned in the policies."
    ),
    expected_output=(
        "A clear, concise summary bulleting the remote work allowance, core working hours, "
        "and equipment stipend details as stated in the official policy."
    ),
    agent=hr_agent
)

# 6. Assemble and Kickoff the Crew
def main():
    crew = Crew(
        agents=[hr_agent],
        tasks=[policy_task],
        process=Process.sequential,
        verbose=True
    )

    # Input prompt passed dynamically to the task
    inputs = {
        "user_query": "What are the rules regarding working remotely and buying work equipment?"
    }

    result = crew.kickoff(inputs=inputs)

    print("\n=================== FINAL RESPONSE ===================\n")
    print(result.raw)

if __name__ == "__main__":
    main()