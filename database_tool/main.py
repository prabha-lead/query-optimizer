import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import AIMessage
from fastapi.templating import Jinja2Templates
import re
from .logger import log_token_usage

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates for server-side rendering
templates = Jinja2Templates(directory="templates")

# Set up logging (Console Output)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define Pydantic model for input data
class OptimizationRequest(BaseModel):
    sql_query: str

# Initialize Anthropic API client
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables.")

def parse_optimized_response(response: str):
    """
    Extracts the optimized query, index creation suggestions, and recommendations.
    """
    optimized_query = ""
    index_suggestion = ""
    recommendations = []
    
    sections = re.split(r'\n\n+', response.strip())
    for section in sections:
        if section.lower().startswith("optimized sql query:"):
            optimized_query = section.split("\n", 1)[-1].strip()
        elif section.lower().startswith("index creation suggestion:"):
            index_suggestion = section.split("\n", 1)[-1].strip()
        elif section.lower().startswith("additional recommendations:"):
            recommendations = section.split("\n")[1:]
    
    return optimized_query, index_suggestion, recommendations

def optimize_query(sql_query: str):
    """
    Optimize SQL queries using LangChain and ChatAnthropic.
    """
    try:
        logger.info("Starting SQL optimization task...")
        start_time = time.time()
        
        # Initialize AI model
        model = ChatAnthropic(
            temperature=0.7,
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key
        )
        
        # Define strict response format in the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in database query optimization. Optimize the given SQL query for performance."),
            ("human", "Optimize the following SQL query and return the response in this strict format:\n\n"
                      "Optimized SQL Query:\n```sql\n[Optimized SQL Query]\n```\n\n"
                      "Index Creation Suggestion (if applicable):\n```sql\n[Index Query]\n```\n\n"
                      "Additional Recommendations:\n- [Recommendation 1]\n- [Recommendation 2]\n- [Recommendation 3]\n\n"
                      "SQL Query: {sql_query}")
        ])
        
        # Ensure correct input format (dictionary with expected key)
        optimizer = RunnableSequence(prompt, model)
        result = optimizer.invoke({"sql_query": sql_query})
        
        # Log token usage using external function
        log_token_usage(result)
        
        # Extract text content from AIMessage
        result_content = result.content if isinstance(result, AIMessage) else str(result)

        optimized_query, index_suggestion, recommendations = parse_optimized_response(result_content)
        elapsed_time = time.time() - start_time
        logger.info(f"Optimization completed in {elapsed_time:.2f} seconds.")
        
        return optimized_query, index_suggestion, recommendations
    except Exception as e:
        logger.error(f"Optimization failed due to: {str(e)}")
        return "", "", [f"Optimization failed due to: {str(e)}"]

@app.get("/", response_class=HTMLResponse)
def render_form(request: Request):
    """
    Render the HTML form for SQL query input.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/optimize", response_class=HTMLResponse)
def optimize(request: Request, sql_query: str = Form(...)):
    """
    Process the SQL query optimization request and return the response in a web page.
    """
    optimized_query, index_suggestion, recommendations = optimize_query(sql_query)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sql_query": sql_query,
        "optimized_query": f"<pre><code class='language-sql'>{optimized_query.replace('```sql', '').replace('```', '')}</code></pre>" if optimized_query else "",
        "index_suggestion": f"<pre><code class='language-sql'>{index_suggestion.replace('```sql', '').replace('```', '')}</code></pre>" if index_suggestion else "",
        "recommendations": [recommendation.replace("-", "") for recommendation in recommendations]
    })

@app.post("/api/optimize")
def optimize_api(request: OptimizationRequest):
    """
    API endpoint to process SQL query optimization and return JSON response.
    """
    optimized_query, index_suggestion, recommendations = optimize_query(request.sql_query)
    return {
        "status": "completed",
        "optimized_query": optimized_query,
        "index_suggestion": index_suggestion if index_suggestion else None,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    uvicorn.run("database_tool.main:app", host="0.0.0.0", port=8000, reload=True)
