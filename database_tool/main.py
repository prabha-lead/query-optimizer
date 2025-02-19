import os
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
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
import tempfile
from fpdf import FPDF
import markdown
import webbrowser

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
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

# Define function to convert SQL to Markdown
def sql_to_markdown(sql_content: str):
    return f"```sql\n{sql_content}\n```"

# Define function to process DDL SQL file
def optimize_ddl(ddl_sql: str):
    """
    Optimize DDL SQL using LangChain and ChatAnthropic.
    """
    try:
        logging.info("Starting DDL optimization task...")
        start_time = time.time()
        
        model = ChatAnthropic(
            temperature=0.7,
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key
        )
        
        markdown_input = sql_to_markdown(ddl_sql)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in database schema optimization. Optimize the given DDL SQL for performance and best practices."),
            ("human", "Analyze the following DDL SQL file and return the response in this format:\n\n"
                      "Optimized DDL:\n```sql\n[Optimized SQL]\n```\n\n"
                      "Indexing Strategy:\n```sql\n[Index Creation Queries]\n```\n\n"
                      "Additional Recommendations:\n- [Recommendation 1]\n- [Recommendation 2]\n- [Recommendation 3]\n\n"
                      "DDL SQL: {markdown_input}")
        ])
        
        optimizer = RunnableSequence(prompt, model)
        result = optimizer.invoke({"markdown_input": markdown_input})
        log_token_usage(result)
        
        result_content = result.content if isinstance(result, AIMessage) else str(result)
        
        optimized_sql, index_suggestions, recommendations = parse_ddl_response(result_content)
        elapsed_time = time.time() - start_time
        logging.info(f"DDL Optimization completed in {elapsed_time:.2f} seconds.")
        
        return optimized_sql, index_suggestions, recommendations
    except Exception as e:
        logging.error(f"DDL Optimization failed: {str(e)}")
        return "", "", [f"Optimization failed: {str(e)}"]

# Function to parse response
def parse_ddl_response(response: str):
    optimized_sql = ""
    index_suggestions = ""
    recommendations = []
    
    sections = re.split(r'\n\n+', response.strip())
    for section in sections:
        if section.lower().startswith("optimized ddl:"):
            optimized_sql = section.split("\n", 1)[-1].strip()
        elif section.lower().startswith("indexing strategy:"):
            index_suggestions = section.split("\n", 1)[-1].strip()
        elif section.lower().startswith("additional recommendations:"):
            recommendations = section.split("\n")[1:]
    
    return optimized_sql, index_suggestions, recommendations

# Open PDF in a new tab
def open_pdf_in_browser(pdf_path):
    webbrowser.open_new_tab(f"file://{pdf_path}")

# Function to generate PDF report
def generate_pdf_report(optimized_sql, index_suggestions, recommendations):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "DDL Optimization Report", ln=True, align='C')
    
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(200, 10, "Optimized DDL:", ln=True)
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 5, optimized_sql)
    
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(200, 10, "Indexing Strategy:", ln=True)
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 5, index_suggestions)
    
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(200, 10, "Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    for rec in recommendations:
        pdf.multi_cell(0, 5, f"- {rec}")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    open_pdf_in_browser(temp_file.name)
    return temp_file.name

@app.get("/ddl-optimize-report", response_class=HTMLResponse)
def render_ddl_report_page(request: Request):
    return templates.TemplateResponse("ddl_report.html", {"request": request})

@app.post("/ddl-optimize-report")
def process_ddl_report(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".sql"):
        return {"error": "Only .sql files are allowed."}
    
    content = file.file.read().decode("utf-8")
    optimized_sql, index_suggestions, recommendations = optimize_ddl(content)
    
    if not optimized_sql:
        return {"error": "Failed to optimize the DDL file."}
    
    pdf_path = generate_pdf_report(optimized_sql, index_suggestions, recommendations)
    
    return {"pdf_report": pdf_path}

if __name__ == "__main__":
    uvicorn.run("database_tool.main:app", host="0.0.0.0", port=8000, reload=True)
