import os
from agno.agent import Agent
from agno.models.openrouter import OpenRouterResponses
from dotenv import load_dotenv
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
load_dotenv()

compilance_file = "bank_compilance_rules.txt"
if not os.path.exists(compilance_file):
    print(f"error: file cannot be accessed")
else:
    with open(compilance_file, "r") as f:
        rules = f.read()
    print("RAG system have accessed the file contents")

def investigation_report(user_input: dict, prediction_label: str, model_id = "liquid/lfm-2.5-2.6b:free") -> str:
    model_provider = OpenRouterResponses(id = model_id,
                                api_key = os.getenv("OPENROUTER_API_KEY"))

    knowledge_base = Knowledge(
       name = "fraud detection knowledge base",
       description="Agno 2.0 Knowledge Implementation with fraud detection SOP rules",
       vector_db=ChromaDb(collection="fraud_compliance_vectors",
                           path = "tmp/chromadb",
                           persistent_client = True)
    )

    knowledge_base.insert(
    name="fraud detection assistant",
    path=compilance_file,)

    # Agno agent
    agent = Agent(
        model = model_provider,
        instructions = [ 
            # Core Persona & Operational Mandate
            "ROLE DEFINITION: You are an elite Banking Fraud Compliance Officer and Risk Analyst. Your core function is to audit raw transaction data points and validate machine learning classifications against corporate compliance mandates.",
            
            # 2. RAG Source Grounding Constraints
            f"KNOWLEDGE GROUNDING PROTOCOL: You must use the reference data injected below as your single source of truth. This context was safely extracted from the project's '{compilance_file}' script layer:\n\n[START OF DATA LAB RULES]\n{rules}\n[END OF DATA LAB RULES]",
            
            # 3. Step-by-Step Processing Pipeline
            "EXECUTION PIPELINE RULES:",
            "  Step A - Retrieval: Scan the injected reference text rules specifically looking for sections that mention the active features in the payload (e.g., if Transaction_Distance > 2000, retrieve Section 5; if Risk_Score is checked, retrieve Section 2).",
            "  Step B - Analysis: Factually cross-reference the user's active numerical variables against the numerical boundaries defined inside the retrieved SOP rules.",
            "  Step C - Mitigation Formulation: Determine authorized corporate response vectors strictly using Sections 14, 15, and 17 of the SOP guidelines. Do not invent automated card termination powers unless explicitly backed by the file rules context.",
            
            # 4. Strict Formatting Restrictions (Enforcing Section 18)
            "REPORTING COMPLIANCE STRUCTURE: Your entire final response must be formatted using clean, professional executive Markdown. You must exactly use the structural layout mandated by Section 18 of the document, containing these exact headers:",
            "  ### 📋 Transaction Summary",
            "  ### 🧠 Machine Learning Model Prediction",
            "  ### 🔍 Relevant Risk Indicators",
            "  ### 📑 Retrieved SOP Rules (Cite specific Section Numbers from the context file)",
            "  ### ⚖️ Risk Assessment and Supporting Evidence",
            "  ### 🛡️ Recommended Mitigation Actions",
            "  ### ⚠️ Uncertainties and Missing Information",
            "  ### 📝 Final Review Status",
            
            # 5. Hallucination Guardrails
            "ANTI-HALLUCINATION ENFORCEMENT: Never fabricate missing user values or make assumptions about unlisted account parameters. If a parameter is not explicitly provided in the payload context, label it clearly under the 'Uncertainties and Missing Information' section as mandated by Section 19."
        ],
        markdown = True,
        knowledge = knowledge_base,
        search_knowledge=True
    )

    agent_prmt = f"""
    Execute an automated fraud audit review based on the following real-time transaction parameters:
    
    - Real-Time Feature Matrix: {user_input}
    - Machine Learning Core Prediction Label: {prediction_label}
    
    Task Instructions: Process this numerical vector against your corporate rule knowledge base and assemble your compliance report output.
    """
    agent_response = agent.run(agent_prmt)
    return agent_response.content