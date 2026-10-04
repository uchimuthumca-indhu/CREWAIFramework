import os
from crewai import Agent, Task, Crew, Process, LLM

# -------------------------------------------------------------------
# 0. DEFINE LLM MODELS
# -------------------------------------------------------------------

# You can use a single model or assign different models to different agents
gpt4o = LLM(
    model="gpt-4o",
    temperature=0.7
)

gpt4o_mini = LLM(
    model="gpt-4o-mini",
    temperature=0.5
)

# Example: Anthropic Claude model option
# claude = LLM(
#     model="anthropic/claude-3-5-sonnet-20240620",
#     api_key=os.getenv("ANTHROPIC_API_KEY")
# )


# -------------------------------------------------------------------
# 1. DEFINE SPECIALIZED AGENTS WITH LLM PARAMETER
# -------------------------------------------------------------------

# Step 1 Agent: Scriptwriter (Using GPT-4o for complex writing)
scriptwriter = Agent(
    role="Senior YouTube Content Strategist & Scriptwriter",
    goal="Write engaging, retention-optimized YouTube video scripts with clear hooks and calls to action.",
    backstory=(
        "You are an expert YouTube creator with millions of views. You know how to "
        "structure videos to maximize watch time, write compelling hooks in the first 10 seconds, "
        "and structure visual cues alongside verbal narrative."
    ),
    llm=gpt4o,  # <--- Explicitly assigned LLM model
    verbose=True,
    memory=True
)

# Step 2 Agent: Visual & Title Designer (Using GPT-4o-mini for fast brainstorming)
thumbnail_designer = Agent(
    role="YouTube Packaging & CTR Specialist",
    goal="Design high-click-through-rate (CTR) video titles and visual thumbnail concepts.",
    backstory=(
        "You specialize in YouTube psychology and packaging. You analyze scripts to extract "
        "high-curiosity titles and generate high-contrast visual concepts for video thumbnails."
    ),
    llm=gpt4o_mini,  # <--- Can assign a lighter model to save costs
    verbose=True
)

# Step 3 Agent: SEO & Distribution Manager
seo_manager = Agent(
    role="YouTube SEO & Metadata Specialist",
    goal="Generate SEO-optimized descriptions, tags, and chapter timestamps for video upload.",
    backstory=(
        "You understand search algorithms and metadata ranking factors. You ensure the "
        "video description includes strong target keywords, structured timestamps, and social links."
    ),
    llm=gpt4o,  # <--- Explicitly assigned LLM model
    verbose=True
)


# -------------------------------------------------------------------
# 2. DEFINE TASKS (Multi-Step Pipeline)
# -------------------------------------------------------------------

task_script = Task(
    description=(
        "Create a detailed 3-5 minute YouTube video script on the topic: '{topic}'."
        "\nInclude:"
        "\n- A strong 15-second visual and verbal hook"
        "\n- Core educational/entertainment body split into 3 logical sections"
        "\n- On-screen visual/b-roll cues in brackets [e.g., Visual: show graph]"
        "\n- Clear Call to Action (CTA) at the end"
    ),
    expected_output="A full video script with spoken dialogue and bracketed b-roll instructions.",
    agent=scriptwriter
)

task_packaging = Task(
    description=(
        "Based on the generated script, create high-performing packaging assets:"
        "\n1. Provide 5 viral, click-worthy YouTube titles (under 60 characters)."
        "\n2. Provide 3 detailed Thumbnail Concepts (describe colors, main focal subject, text overlay)."
    ),
    expected_output="5 catchy title options and 3 detailed thumbnail design specs.",
    agent=thumbnail_designer
)

task_seo = Task(
    description=(
        "Using the generated script and selected packaging concepts:"
        "\n1. Write a 200-word SEO-friendly video description containing relevant keywords."
        "\n2. Generate 15 relevant YouTube tags (comma-separated)."
        "\n3. Generate estimated video timestamps (e.g., 00:00 - Intro, 01:15 - Point 1)."
    ),
    expected_output="A structured markdown block with description, timestamps, and tags list.",
    agent=seo_manager,
    output_file="youtube_production_plan.md"
)


# -------------------------------------------------------------------
# 3. ASSEMBLE CREW & EXECUTE WORKFLOW
# -------------------------------------------------------------------

youtube_workflow = Crew(
    agents=[scriptwriter, thumbnail_designer, seo_manager],
    tasks=[task_script, task_packaging, task_seo],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    inputs = {
        "topic": "How to Build AI Agents with CrewAI in 2026"
    }
    
    result = youtube_workflow.kickoff(inputs=inputs)
    
    print("\n\n================ WORKFLOW COMPLETE ================\n")
    print(result)