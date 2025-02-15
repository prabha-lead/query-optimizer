# SQL Query Optimization Tool

This project is a **FastAPI-based SQL Query Optimization Tool** that utilizes **Anthropic Claude** to analyze and optimize SQL queries. The application provides recommendations to improve database performance, including query restructuring, index suggestions, and best practices.

## Features

- Accepts SQL queries as input.
- Uses **LangChain + Anthropic Claude** for AI-based SQL optimization.
- Provides **optimized queries, index creation suggestions, and additional recommendations**.
- Offers both **API and Web UI** for interaction.
- Includes a **CodeMirror-powered SQL editor** for a rich user experience.
- Supports **copy-to-clipboard** for optimized queries and indexes.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/prabha-lead/query-optimizer.git
cd query-optimizer
```

### 2️⃣ Install Dependencies (Using Poetry)

Make sure you have [Poetry](https://python-poetry.org/docs/) installed.

```sh
poetry install
```

### 3️⃣ Set Up Environment Variables

Create a `.env` file in the root directory and add the following:

```ini
ANTHROPIC_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your **Anthropic Claude API Key**.

### 4️⃣ Run the Application

```sh
poetry run uvicorn database_tool.main:app --reload
```

### 5️⃣ Access the Application

- **Web UI**: Open `http://127.0.0.1:8000/` in your browser.
- **API Endpoint**: Use Postman or CURL to send requests to `http://127.0.0.1:8000/api/optimize`.

---

## 🛠 API Usage

### **Endpoint: POST `/api/optimize`**

#### **Request:**

```json
{
  "sql_query": "SELECT * FROM users WHERE age > 30;"
}
```

#### **Response:**

```json
{
  "status": "completed",
  "optimized_query": "SELECT id, name, age FROM users WHERE age > 30;",
  "index_suggestion": "CREATE INDEX idx_users_age ON users(age);",
  "recommendations": [
    "Avoid using SELECT * to reduce I/O",
    "Add an index on the filtered column (age) to improve WHERE clause performance"
  ]
}
```

---

## 🔧 Technologies Used

- **FastAPI** - API framework
- **LangChain** - AI-powered query optimization
- **Anthropic Claude** - AI model for analysis
- **CodeMirror** - SQL editor in the UI
- **Poetry** - Dependency management
- **Tailwind CSS** - UI styling

---

## 🤝 Contributing

Feel free to **fork** this repository and contribute! Pull requests are welcome. 😊
