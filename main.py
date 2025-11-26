import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load Environment Variables
load_dotenv()

# Check API Keys
if not os.getenv("GROQ_API_KEY"):
    print("❌ Error: GROQ_API_KEY not found in .env file")
    exit()

if not os.getenv("TAVILY_API_KEY"):
    print("❌ Error: TAVILY_API_KEY not found in .env file")
    exit()

print("🚀 Initializing Medical AI Agent (Universal Mode)...")

# Setup LLM (Language Model)
llm = ChatGroq(
    #model="llama3-70b-8192",
    model="llama-3.3-70b-versatile", 
    temperature=0
)

# Setup Database Connections
db_dir = "databases"
heart_db = SQLDatabase.from_uri(f"sqlite:///{db_dir}/heart_disease.db")
cancer_db = SQLDatabase.from_uri(f"sqlite:///{db_dir}/cancer.db")
diabetes_db = SQLDatabase.from_uri(f"sqlite:///{db_dir}/diabetes.db")

# Create Tools

# Heart Disease Tool
def query_heart_db(query: str):
    """Queries the Heart Disease database."""
    chain = QuerySQLDataBaseTool(db=heart_db)
    return chain.invoke(query)

heart_tool = Tool(
    name="HeartDiseaseDBTool",
    func=query_heart_db,
    description="Useful for querying Heart Disease patient data, statistics, or numbers."
)

# Cancer Tool
def query_cancer_db(query: str):
    """Queries the Cancer database."""
    chain = QuerySQLDataBaseTool(db=cancer_db)
    return chain.invoke(query)

cancer_tool = Tool(
    name="CancerDBTool",
    func=query_cancer_db,
    description="Useful for querying Cancer prediction data and patient records."
)

# Diabetes Tool
def query_diabetes_db(query: str):
    """Queries the Diabetes database."""
    chain = QuerySQLDataBaseTool(db=diabetes_db)
    return chain.invoke(query)

diabetes_tool = Tool(
    name="DiabetesDBTool",
    func=query_diabetes_db,
    description="Useful for querying Diabetes patient data and statistics."
)

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)
web_search_tool.name = "MedicalWebSearchTool"
web_search_tool.description = "Useful for general medical knowledge, symptoms, cures, or definitions (NOT for database statistics)."

# Tool List
tools = [heart_tool, cancer_tool, diabetes_tool, web_search_tool]

# Create Agent Logic (Using older but stable method)
print("🤖 Building Agent Logic...")

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # Best for multi-tool handling
    verbose=True,
    handle_parsing_errors=True
)

# Main User Loop
if __name__ == "__main__":
    print("\n👨‍⚕️ Medical AI Agent is Ready! (Type 'exit' to stop)\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye! 👋")
                break
            
            # Using simple invoke
            response = agent_executor.invoke({"input": user_input})
            
            # Handling response format
            output = response.get('output') if isinstance(response, dict) else str(response)
            print(f"\n🤖 Agent: {output}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")