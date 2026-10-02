import os
from crewai import Agent, Crew, Process, Task


def run_travel_planner():
    # 1. Get city input from the user
    city = input("Enter a city you'd like to visit: ").strip()

    if not city:
        print("Please enter a valid city name.")
        return

    # 2. Define the Travel Planner Agent
    travel_planner_agent = Agent(
        role="Expert Local Travel Guide",
        goal=f"Suggest top places to visit and help travelers get the most out of their trip to {city}.",
        backstory=(
            "You are a world-traveled expert who has spent years exploring hidden gems, "
            "iconic landmarks, and local secrets across global destinations. "
            "Your recommendations are tailored to give visitors a rich, memorable experience."
        ),
        verbose=True,
    )

    # 3. Define the Task for the Agent
    recommendation_task = Task(
        description=(
            f"Provide a curated list of exactly 3 places to visit in {city}. "
            "For each place, include: \n"
            "1. Name of the place\n"
            "2. Brief description of what makes it special\n"
            "3. One practical local tip for visitors (e.g., best time to go, hidden feature, or trick)."
        ),
        expected_output=(
            "A well-structured list containing 3 recommended places to visit, "
            "each with its name, description, and a local insider tip."
        ),
        agent=travel_planner_agent,
    )

    # 4. Assemble the Agent and Task into a Crew
    travel_crew = Crew(
        agents=[travel_planner_agent],
        tasks=[recommendation_task],
        process=Process.sequential,
    )

    # 5. Kick off the Crew and print the result
    print(f"\n--- Planning trip recommendations for {city} ---\n")
    result = travel_crew.kickoff()

    print("\n" + "=" * 40)
    print("      TRAVEL PLANNER RECOMMENDATIONS      ")
    print("=" * 40 + "\n")
    print(result)


if __name__ == "__main__":
    run_travel_planner()