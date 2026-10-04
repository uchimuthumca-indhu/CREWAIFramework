import os
from crewai import Agent, Task, Crew, Process, LLM

# Optional: Specify your model explicitly
# llm = LLM(model="gpt-4o-mini")

# ==========================================
# 1. DEFINE AGENTS
# ==========================================

researcher = Agent(
    role="Senior Technology Researcher",
    goal="Uncover cutting-edge developments in AI and software engineering",
    backstory=(
        "You are a tech analyst known for identifying major industry shifts "
        "and providing actionable, factual summaries."
    ),
    verbose=True
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft clear, engaging articles based on technical research",
    backstory=(
        "You convert complex research and data into easily readable, "
        "compelling stories for a professional audience."
    ),
    verbose=True
)

editor = Agent(
    role="Managing Editor",
    goal="Review content for technical clarity, tone, and overall quality",
    backstory=(
        "An experienced editor who ensures every article meets publishing standards "
        "and aligns with target reader expectations."
    ),
    verbose=True
)

# ==========================================
# 2. DEFINE TASKS
# ==========================================

research_task = Task(
    description=(
        "Conduct research on {topic}. Identify key trends, practical "
        "applications, and future implications."
    ),
    expected_output="A structured summary of key findings with bullet points.",
    agent=researcher
)

write_task = Task(
    description=(
        "Using the research output, write a short, cohesive article on {topic}. "
        "Include an introduction, 3 main takeaways, and a concise conclusion."
    ),
    expected_output="A 300-word article formatted in Markdown.",
    agent=writer
)

edit_task = Task(
    description=(
        "Review the article for flow, tone, and accuracy. Refine any weak phrasing "
        "and correct grammar errors."
    ),
    expected_output="The final, polished Markdown article ready for publication.",
    agent=editor
)

# ==========================================
# 3. IMPLEMENTATION: SEQUENTIAL PROCESS
# ==========================================

def run_sequential_crew(topic_input):
    print("\n--- Running Sequential Process ---\n")
    
    sequential_crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, write_task, edit_task],
        process=Process.sequential,  # Tasks execute step-by-step in order
        verbose=True
    )
    
    return sequential_crew.kickoff(inputs={"topic": topic_input})


# ==========================================
# 4. IMPLEMENTATION: HIERARCHICAL PROCESS
# ==========================================

def run_hierarchical_crew(topic_input):
    print("\n--- Running Hierarchical Process ---\n")
    
    # Manager Agent automatically plans, delegates, and evaluates output
    manager = Agent(
        role="Editorial Director",
        goal="Coordinate researchers, writers, and editors to produce high-quality reports",
        backstory=(
            "You oversee the entire media production line, delegating work to "
            "the right specialists and approving the final draft."
        ),
        verbose=True
    )

    hierarchical_crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, write_task, edit_task],
        process=Process.hierarchical,
        manager_agent=manager,  # Requires a designated manager agent
        verbose=True
    )
    
    return hierarchical_crew.kickoff(inputs={"topic": topic_input})


# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    topic = "Agentic AI Frameworks in 2026"
    
    # Run Sequential Workflow
    sequential_result = run_sequential_crew(topic)
    print("\n=== SEQUENTIAL OUTPUT ===")
    print(sequential_result.raw)

    # Run Hierarchical Workflow
    # hierarchical_result = run_hierarchical_crew(topic)
    # print("\n=== HIERARCHICAL OUTPUT ===")
    # print(hierarchical_result.raw)