import os
from dotenv import load_dotenv
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

# ------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------
# Load API keys from .env file automatically
load_dotenv()

# Verify that required API keys are available in environment variables
if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPER_API_KEY"):
    st.warning(
        "⚠️ Please set OPENAI_API_KEY and SERPER_API_KEY in your environment or .env file."
    )


# ------------------------------------------------------------------
# Helper Function to Build and Run the Crew
# ------------------------------------------------------------------
def run_support_crew(user_query: str):
    """Initializes agents, tasks, and executes the sequential CrewAI workflow."""

    # 1. Initialize Tools
    search_tool = SerperDevTool()

    # 2. Define Agents
    assistant_agent = Agent(
        role="Assistant",
        goal="Answer the user's query directly using internal knowledge.",
        backstory=(
            "You are a helpful and knowledgeable customer support assistant. "
            "You answer user queries concisely and directly based on standard knowledge."
        ),
        verbose=True,
        memory=False,
    )

    web_search_agent = Agent(
        role="Web Search Assistant",
        goal="Search the web for up-to-date information regarding the user's query and summarize the findings.",
        backstory=(
            "You are an expert research assistant skilled in using search tools to find "
            "accurate, recent, and specific factual details from the internet."
        ),
        tools=[search_tool],
        verbose=True,
        memory=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal="Save the query and both generated answers into a text file and prepare a consolidated summary.",
        backstory=(
            "You are a meticulous record keeper. You aggregate customer support interactions, "
            "log them into files for auditing, and format the output clearly for the user UI."
        ),
        verbose=True,
        memory=False,
    )

    # 3. Define Tasks
    task1 = Task(
        description=f"Answer the following user query directly using your own knowledge:\n'{user_query}'",
        expected_output="A direct, concise answer to the query based on internal knowledge.",
        agent=assistant_agent,
    )

    task2 = Task(
        description=(
            f"Search the web for the user query: '{user_query}'. "
            "Provide a comprehensive answer based on the web search results."
        ),
        expected_output="An up-to-date answer derived from online search results.",
        agent=web_search_agent,
    )

    task3 = Task(
        description=(
            f"Collect the user query ('{user_query}'), the standard assistant answer from Task 1, "
            "and the web search answer from Task 2.\n"
            "1. Write/append this entire log into a file named 'answers.txt'.\n"
            "2. Output both answers clearly formatted so they can be presented directly to the user."
        ),
        expected_output=(
            "A summary containing both 'Assistant Answer' and 'Web Search Answer', "
            "and confirmation that 'answers.txt' has been updated."
        ),
        agent=entry_agent,
        output_file="answers.txt",  # Automatically saves final task output to answers.txt
    )

    # 4. Assemble the Crew
    support_crew = Crew(
        agents=[assistant_agent, web_search_agent, entry_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,  # Run sequentially in order
        verbose=True,
    )

    # 5. Execute Workflow
    result = support_crew.kickoff()
    return result, task1.output.raw, task2.output.raw


# ------------------------------------------------------------------
# Streamlit Interface
# ------------------------------------------------------------------
st.set_page_config(page_title="Multi-Agent Customer Support", page_icon="🤖")

st.title("🤖 Multi-Agent Customer Support System")
st.write(
    "Powered by **CrewAI** | Runs direct analysis, web search, and logs history sequentially."
)

# User Query Input
user_query = st.text_input(
    "Enter your query or task:",
    placeholder="e.g., How do I reset my password?",
)

if st.button("Submit Query", type="primary"):
    if not user_query.strip():
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Processing through Support Crew agents..."):
            try:
                # Execute crew
                final_output, assistant_ans, web_ans = run_support_crew(user_query)

                st.success("Processing complete! Log saved to `answers.txt`.")

                # Layout results in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("💡 1. Assistant Answer")
                    st.info(assistant_ans)

                with col2:
                    st.subheader("🌐 2. Web Search Answer")
                    st.success(web_ans)

                # Show aggregated output from Entry Agent
                with st.expander("📋 Consolidated Output (Entry Agent)"):
                    st.markdown(final_output)

            except Exception as e:
                st.error(f"An error occurred while executing the Crew: {e}")