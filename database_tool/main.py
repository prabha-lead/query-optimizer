import os
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import time
from openai import AzureOpenAI
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

# Initialize Azure OpenAI client
open_api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
if not open_api_key or not api_base:
    raise ValueError("OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is not set in the environment variables.")

client = AzureOpenAI(
    api_key=open_api_key,  
    api_version="2024-08-01-preview",
    azure_endpoint=api_base
)

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
            optimized_query = section.split("```sql", 1)[-1].rsplit("```", 1)[0].strip()
        elif section.lower().startswith("index creation suggestion (if applicable):"):
            index_suggestion = section.split("```sql", 1)[-1].rsplit("```", 1)[0].strip()
        elif section.lower().startswith("additional recommendations:"):
            recommendations = [line.strip() for line in section.split("\n")[1:] if line.strip().startswith("-")]

    return optimized_query, index_suggestion, recommendations

def optimize_query(sql_query: str):
    """
    Optimize SQL queries using Azure OpenAI.
    """
    try:
        logger.info("Starting SQL optimization task...")
        start_time = time.time()
        
        system_prompt = "You are an expert in database query optimization. Optimize the given SQL query for performance."
        user_prompt = f"Optimize the following SQL query and return the response in this strict format:\n\n" \
                      f"Optimized SQL Query:\n```sql\n[Optimized SQL Query]\n```\n\n" \
                      f"Index Creation Suggestion (if applicable):\n```sql\n[Index Query]\n```\n\n" \
                      f"Additional Recommendations:\n- [Recommendation 1]\n- [Recommendation 2]\n- [Recommendation 3]\n- [Recommendation n+]\n\n" \
                      f"SQL Query: {sql_query}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        
        response_message = response.choices[0].message.content
        
        # Log token usage using external function
        log_token_usage(response)
        
        optimized_query, index_suggestion, recommendations = parse_optimized_response(response_message)
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
    Optimize DDL SQL using Azure OpenAI.
    """
    try:
        logging.info("Starting DDL optimization task...")
        start_time = time.time()
        
        markdown_input = sql_to_markdown(ddl_sql)
        
        system_prompt = "You are an expert in database schema optimization. Optimize the given DDL SQL for performance and best practices."
        user_prompt = f"Analyze the following DDL SQL file and return the response in this format:\n\n" \
                      f"Optimized DDL:\n```sql\n[Optimized SQL]\n```\n\n" \
                      f"Indexing Strategy:\n```sql\n[Index Creation Queries]\n```\n\n" \
                      f"Additional Recommendations:\n- [Recommendation 1]\n- [Recommendation 2]\n- [Recommendation 3]\n\n" \
                      f"DDL SQL: {markdown_input}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        
        response_message = response.choices[0].message.content
        
        log_token_usage(response)
        
        optimized_sql, index_suggestions, recommendations = parse_ddl_response(response_message)
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
            recommendations = [line.strip() for line in section.split("\n")[1:] if line.strip().startswith("-")]
    
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
