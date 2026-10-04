import os
from dotenv import load_dotenv

from crewai import Agent, Crew, Task, LLM
from crewai_tools import PDFSearchTool, FileReadTool, FileWriterTool

# -------------------------------------------------------------------
# 1. Environment Setup & Validation
# -------------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file.")

# Ensure working directories exist
os.makedirs("documents", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Define path to target document
TARGET_FILE = os.path.join("documents", "sample_invoice.pdf")

# Create a sample text file if target doesn't exist yet for testing
if not os.path.exists(TARGET_FILE):
    TARGET_FILE = os.path.join("documents", "sample_contract.txt")
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(
            "SERVICE AGREEMENT\n"
            "Client: Acme Corp\n"
            "Vendor: TechSolutions LLC\n"
            "Contract Value: $15,000 USD\n"
            "Effective Date: October 1, 2026\n"
            "Scope: Cloud infrastructure migration and CrewAI agent integration.\n"
            "Term: 6 Months\n"
        )

# -------------------------------------------------------------------
# 2. Configure LLM Model
# -------------------------------------------------------------------
gpt4_llm = LLM(
    model="gpt-4o",
    temperature=0.1  # Low temperature for precise document parsing
)

# -------------------------------------------------------------------
# 3. Initialize File Processing Tools
# -------------------------------------------------------------------
# Tool 1: General file reading tool
file_read_tool = FileReadTool()

# Tool 2: PDF semantic/search tool (RAG enabled)
pdf_search_tool = PDFSearchTool(pdf=TARGET_FILE) if TARGET_FILE.endswith(".pdf") else None

# Tool 3: File writing tool to save output reports
file_write_tool = FileWriterTool()

# Assemble active retrieval tools based on file type
document_tools = [file_read_tool]
if pdf_search_tool:
    document_tools.append(pdf_search_tool)

# -------------------------------------------------------------------
# 4. Define Agents
# -------------------------------------------------------------------

# Agent 1: Reads and extracts text from files
doc_extractor_agent = Agent(
    role="Senior Document Reader & Extractor",
    goal="Extract raw text, numerical figures, metadata, and key entities from provided files.",
    backstory=(
        "You are an expert digital archivist. You excel at opening documents, "
        "locating crucial key-value pairs, clauses, and unformatted data without altering details."
    ),
    tools=document_tools,
    llm=gpt4_llm,
    verbose=True
)

# Agent 2: Processes and structures raw information
data_analyst_agent = Agent(
    role="Information & Compliance Analyst",
    goal="Analyze extracted document contents, identify structured metrics, and verify data accuracy.",
    backstory=(
        "You are an analytical auditor skilled at converting unstructured raw text "
        "into clear, structured JSON/Markdown reports with actionable intelligence."
    ),
    llm=gpt4_llm,
    verbose=True
)

# Agent 3: Saves formatted output to disk
writer_agent = Agent(
    role="Executive Documentation Specialist",
    goal="Compile formatted findings and write the resulting summary report to an output file.",
    backstory="You specialize in technical documentation and persist structured reports to the file system.",
    tools=[file_write_tool],
    llm=gpt4_llm,
    verbose=True
)

# -------------------------------------------------------------------
# 5. Define Workflow Tasks
# -------------------------------------------------------------------

# Task 1: Document Retrieval & Extraction
extraction_task = Task(
    description=(
        f"Read the document located at: '{TARGET_FILE}'. "
        "Extract all key entities such as dates, monetary figures, party names, terms, and core topics."
    ),
    expected_output="A list of raw extracted text sections and key fields found in the document.",
    agent=doc_extractor_agent
)

# Task 2: Data Synthesis & Analysis
analysis_task = Task(
    description=(
        "Review the extracted content from the previous task. "
        "Organize the information into a structured summary containing:\n"
        "1. Executive Overview\n"
        "2. Extracted Entities (Entities, Dates, Monetary Values)\n"
        "3. Core Terms & Conditions\n"
        "4. Risk/Action Items"
    ),
    expected_output="A structured markdown report representing the document breakdown.",
    agent=data_analyst_agent
)

# Task 3: Save Report to Disk
output_filepath = "output/processed_summary.md"

persistence_task = Task(
    description=(
        f"Take the structured analysis report and write it directly to disk using the file writer tool. "
        f"Target file path: '{output_filepath}'."
    ),
    expected_output=f"A confirmation message that the file was written to '{output_filepath}'.",
    agent=writer_agent
)

# -------------------------------------------------------------------
# 6. Kickoff Crew Workflow
# -------------------------------------------------------------------
if __name__ == "__main__":
    doc_crew = Crew(
        agents=[doc_extractor_agent, data_analyst_agent, writer_agent],
        tasks=[extraction_task, analysis_task, persistence_task]
    )

    print("\n--- Starting File & Document Processing Crew ---\n")
    results = doc_crew.kickoff()

    print("\n--- Processing Completed ---")
    print(results)