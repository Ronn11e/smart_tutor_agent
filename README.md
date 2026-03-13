# SmartTutor
A lightweight, cost-efficient AI tutoring application designed to assist university-level students with math and history homework questions. Features context-aware question validation, LaTeX formula rendering, clean UI, and strict token optimization to minimize API costs.

## Prerequisites
- Python 3.8+ installed
- Azure OpenAI API Key and endpoint (or OpenAI API Key as alternative)
- Required Python packages: `flask`, `openai>=1.0.0`

## Installation & Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd smarttutor
   ```
2. Install dependencies:
   ```bash
   pip install flask openai>=1.0.0
   ```
3. Configure API credentials in `config.py`:
   ```python
   CSIT5900_API_KEY = "your-azure-openai-api-key"
   AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com/"
   API_VERSION = "2024-02-15-preview"
   MODEL_NAME = "gpt-4o-mini"
   ```
   *Alternative (OpenAI official API): Modify `tutor_agent.py` to use `OpenAI` client instead of `AzureOpenAI`.*

## Running the Application
### Web Interface (Recommended)
```bash
python app.py
```
Access the chat interface at `http://localhost:5000`.

### CLI Mode
```bash
python tutor_agent.py
```
Type "exit"/"quit"/"bye" to terminate the session.

## Usage Examples (Valid Homework Questions)
```
# Math - Rational Number Check
Is square root of 1000 a rational number?

# Math - Word Problem
Beth bakes 4.2 dozen batches of cookies in a week. If shared among 16 people equally, how many cookies per person?

# Math - Distance Calculation
How to compute the distance between Hong Kong and Shenzhen?

# History
Who was the first president of France?
```

## Project Structure
```
smarttutor/
├── config.py          # API & server configuration
├── tutor_agent.py     # Core AI logic (validation, LLM calls, formatting)
├── app.py             # Flask server & frontend generation
└── templates/
    └── index.html     # Auto-generated chat interface
```
