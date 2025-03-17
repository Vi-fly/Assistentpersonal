import os
import openai
import sqlite3
import psycopg2.pool
import streamlit as st
from datetime import datetime, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
import json
import plotly.express as px
import tempfile
import time
from jinja2 import Template
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import psycopg2
from psycopg2 import pool
from psycopg2 import Error


# ====== Unified Database Handler ======
st.set_page_config(page_title="DB Manager", layout="wide")


# ====== Unified Database Handler ======
def get_db_connection():
    """Create a new PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT'),
        sslmode='require'
    )

def get_contacts():
    """Fetch name and skills from Contacts table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT NAME, SKILLS FROM CONTACTS;")
    contacts = cursor.fetchall()
    conn.close()
    return contacts

def query_groq(prompt):
    """Send a query to the Groq API and get the response."""
    client = openai.Client(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_prompt(task):
    """Generate a prompt to ask Groq for resource suggestions."""
    contacts = get_contacts()
    skills_dict = {name: skills for name, skills in contacts}
    prompt = f"For the task '{task}', suggest required resources like budget, tools, and storage. "
    prompt += "Also, suggest suitable employees based on these available skills: "
    prompt += ", ".join([f"{name} ({skills})" for name, skills in skills_dict.items()])
    prompt += " If employees with necessary skills are not available, ask to hire employees with necessary skills."
    return prompt
def save_suggestion(task, suggestion):
    """Save the AI suggestion to a text file."""
    filename = os.path.join(CHAT_HISTORY_DIR, f"{task.replace(' ', '_')}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(suggestion.strip())
        st.success(f"Suggestion saved: {filename}")
    except Exception as e:
        st.error(f"Error saving suggestion: {e}")

def list_saved_chats():
    """List all saved chat history files."""
    try:
        return [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".txt")]
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []


def safe_db_query(query, params=None, read=True):
    """Universal database query executor"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params or ())
        
        if read:
            columns = [desc[0] for desc in cur.description] if cur.description else []
            data = cur.fetchall()
            return columns, data
        else:
            conn.commit()
            return cur.rowcount
            
    except psycopg2.Error as e:
        st.error(f"Database error: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

# ====== Modified Task Count Example ======
def get_task_count():
    """Safe task counter with error handling"""
    try:
        columns, data = safe_db_query("SELECT COUNT(*) FROM TASKS")
        return data[0][0] if data else 0
    except Exception as e:
        st.error(f"Count error: {str(e)}")
        return 0

def get_total_tasks():
    try:
        columns, data = safe_db_query("SELECT COUNT(*) FROM TASKS")
        return data[0][0] if data else 0
    except Exception as e:
        st.error(f"Error getting task count: {str(e)}")
        return 0

def get_overdue_tasks():
    try:
        conn = get_db_connection()
        result = pd.read_sql(
            "SELECT COUNT(*) FROM TASKS WHERE DEADLINE < CURRENT_TIMESTAMP", 
            conn
        )
        return result.iloc[0,0]
    except Exception as e:
        st.error(f"Error fetching overdue tasks: {str(e)}")
        return 0


conn = get_db_connection()  # Use PostgreSQL connection
conn.close()  # Close the connection properly


# Load environment variables
load_dotenv('.env')

# Initialize ChatGroq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
)
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize SQL Agent
try:
    # Ensure your DB_URI is correctly formatted
    DB_URI = "postgresql://neondb_owner:npg_LgZ4en1QpsyD@ep-broad-bush-a5xjz0wm-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

    # Initialize database agent with connection string
    db_agent = SQLDatabase.from_uri(
        DB_URI,
        include_tables=['contacts', 'tasks'],  # Ensure these tables exist in the database
        sample_rows_in_table_info=2
    )

    # Initialize the LLM (Language Model) using the correct model name and API key
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    # Create SQL Agent using LLM and Database
    agent_executor = create_sql_agent(
        llm=llm,
        db=db_agent,
        verbose=True,
        top_k=5,
        max_iterations=10
    )

    # Optional: you could use `agent_executor` to query the database through the agent.

except Exception as e:
    # Display the error using Streamlit's error display
    st.error(f"Failed to initialize SQL agent: {e}")

# Helper Functions
def send_task_email(recipient_email, subject, message):
    sender_email = "myhw765@gmail.com"  # Replace with your email
    sender_password = "airukqfciidluzzm"  # Replace with your email password or app password

    # Setting up SMTP server
    smtp_server = "smtp.gmail.com"  # Change for Outlook/Yahoo
    smtp_port = 587

    # Create message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        # Connect to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade the connection to secure
        server.login(sender_email, sender_password)
        server.send_message(msg)
        print(f"Email sent to {recipient_email}")
    except smtplib.SMTPAuthenticationError:
        print("Error: Authentication failed. Check your email and password.")
    except smtplib.SMTPException as e:
        print(f"Error: Failed to send email. {e}")
    finally:
        server.quit()

def generate_task_details(description: str) -> dict:
    """Generate detailed task parameters from description"""
    system_prompt = """Analyze the task description and generate detailed task parameters. Return STRICTLY ONLY VALID JSON with:
    {
        "title": "string",
        "description": "string",
        "category": "Work|Personal|Project|Other",
        "priority": "Low|Medium|High",
        "deadline": "natural language date",
        "assigned_to": "string",
        "status": "Not Started|In Progress|On Hold|Completed",
        "dependencies": "string",
        "required_resources": "string",
        "expected_outcome": "string",
        "review_process": "string",
        "performance_metrics": "string",
        "estimated_time": "string",
        "instructions": "string"
    }
    
    Example Input: "Develop login page with Auth0 by next week, assign to John"
    Example Output:
    {
        "title": "Develop Login Page",
        "description": "Implement secure authentication using Auth0",
        "category": "Work",
        "priority": "High",
        "deadline": "next week",
        "assigned_to": "John",
        "status": "Not Started",
        "dependencies": "Auth0 API access, design mockups",
        "required_resources": "React, Node.js, Auth0 SDK",
        "expected_outcome": "Secure user login system",
        "review_process": "1. Code review\\n2. Security audit",
        "performance_metrics": "Load time <2s, 100% test coverage",
        "estimated_time": "3 days",
        "instructions": "1. Create UI components\\n2. Integrate Auth0"
    }"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=description)
    ]
    
    try:
        response = llm.invoke(messages)
        raw_response = response.content.strip()
        
        # Enhanced cleaning process
        def clean_json_response(text):
            # Remove non-JSON characters at start
            if text.startswith('n'):
                text = text[1:]
            # Remove markdown code blocks
            text = text.replace('```json', '').replace('```', '')
            # Remove any text before {
            start_idx = text.find('{')
            if start_idx > 0:
                text = text[start_idx:]
            return text.strip()
        
        cleaned_response = clean_json_response(raw_response)
        
        # Debugging output
        st.write("Raw LLM Response:", raw_response)
        st.write("Cleaned Response:", cleaned_response)
        
        parsed = json.loads(cleaned_response)
        
        # Validate and default missing fields
        return {
            "title": parsed.get("title", "Unnamed Task"),
            "description": parsed.get("description", ""),
            "category": parsed.get("category", "Work"),
            "priority": parsed.get("priority", "Medium"),
            "deadline": parsed.get("deadline", ""),
            "assigned_to": parsed.get("assigned_to", ""),
            "status": parsed.get("status", "Not Started"),
            "dependencies": parsed.get("dependencies", ""),
            "required_resources": parsed.get("required_resources", ""),
            "expected_outcome": parsed.get("expected_outcome", ""),
            "review_process": parsed.get("review_process", ""),
            "performance_metrics": parsed.get("performance_metrics", ""),
            "estimated_time": parsed.get("estimated_time", "1 day"),
            "instructions": parsed.get("instructions", "")
        }
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format in response: {e}\nRaw LLM response: {raw_response}")
        return {}
    except Exception as e:
        st.error(f"Error parsing task parameters: {str(e)}")
        return {}

def generate_sql_query(prompt: str, action: str) -> str:
    """Generate SQL query based on selected action and user input."""
    system_prompts = {
        "add": (
            "You are an expert in generating SQL INSERT statements for a contacts database. "
            "The database has one table: CONTACTS.\n\n"
            "CONTACTS Table Structure:\n"
            "- ID (INTEGER, PRIMARY KEY, AUTOINCREMENT)\n"
            "- NAME (VARCHAR, NOT NULL)\n"
            "- PHONE (INTEGER, UNIQUE, NOT NULL, 10 digits)\n"
            "- EMAIL (VARCHAR, UNIQUE, NOT NULL)\n"
            "- ADDRESS (TEXT)\n\n"
            "Rules for INSERT Statements:\n"
            "1. For CONTACTS: INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS) VALUES (...);\n"
            "2. Phone numbers must be 10-digit integers.\n"
            "3. Use single quotes for string values.\n"
            "4. Return only the SQL query, no explanations.\n\n"
            "Examples:\n"
            "1. +/(Add new) contact: INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS) VALUES ('John Doe', 5551234567, 'john@email.com', '123 Main St');\n"
            "2. Add another contact: INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS) VALUES ('Jane Smith', 9876543210, 'jane@email.com', '456 Oak Ave');"
        ),
        "view": (
            "You are an expert in generating SQL SELECT queries with JOINs for a contacts and tasks database.\n\n"
            "Rules for SELECT Statements:\n"
            "1. Always use JOINs when showing tasks to include assignee names.\n"
            "2. Use LOWER() for case-insensitive comparisons in WHERE clauses.\n"
            "3. Use proper table aliases (C for CONTACTS, T for TASKS).\n"
            "4. Return only the SQL query, no explanations.\n\n"
            "Examples:\n"
            "1. Show all tasks: SELECT T.ID, T.TITLE, T.DESCRIPTION, T.CATEGORY, T.PRIORITY, T.STATUS, C.NAME AS ASSIGNEE FROM TASKS T LEFT JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID;\n"
            "2. Find contacts from Delhi: SELECT * FROM CONTACTS WHERE LOWER(ADDRESS) LIKE '%delhi%';\n"
            "3. Show ongoing tasks for John: SELECT T.ID, T.TITLE, T.DEADLINE FROM TASKS T JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID WHERE LOWER(C.NAME) = LOWER('John Doe') AND T.STATUS = 'In Progress';"
            "4. Display task 1: SELECT T.* FROM TASKS T JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID WHERE T.ID = 1;"
        ),
        "update": (
            "You are an expert in generating SQL UPDATE statements for a contacts and tasks database.\n\n"
            "Rules for UPDATE Statements:\n"
            "1. For contacts, use ID as the identifier in WHERE clause.\n"
            "2. For tasks, use ID as the identifier in WHERE clause.\n"
            "3. Use single quotes for string values.\n"
            "4. Include only one SET clause per statement.\n"
            "5. Return only the SQL query, no explanations.\n\n"
            "Examples:\n"
            "1. Update contact email: UPDATE CONTACTS SET EMAIL = 'new@email.com' WHERE ID = 2;\n"
            "2. Mark task as completed: UPDATE TASKS SET STATUS = 'Completed' WHERE ID = 5;\n"
            "3. Change task deadline: UPDATE TASKS SET DEADLINE = '2024-12-31 23:59' WHERE ID = 3;\n"
            "4. Reassign task: UPDATE TASKS SET ASSIGNED_TO = (SELECT ID FROM CONTACTS WHERE NAME = 'vivek') WHERE ID = 10;\n"
            "5. Update contact based on name: UPDATE CONTACTS SET ADDRESS = 'New Address' WHERE LOWER(NAME) = LOWER('John Doe');\n"
            "6. Update task status based on title: UPDATE TASKS SET STATUS = 'Reviewed & Approved' WHERE LOWER(TITLE) = LOWER('Project Planning');"
        )
    }
    
    messages = [
        SystemMessage(content=system_prompts[action]),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        sql_query = response.content.strip()
        
        # Clean markdown formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:-3].strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query[3:-3].strip()
            
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return ""

def calculate_initial_date(deadline_input: str) -> datetime:
    """Convert natural language deadline into a datetime object."""
    default_date = datetime.today()
    if not deadline_input:
        return default_date
        
    deadline_input = deadline_input.lower()
    
    if 'tomorrow' in deadline_input:
        return default_date + timedelta(days=1)
    elif 'next week' in deadline_input:
        return default_date + timedelta(weeks=1)
    elif 'in 2 days' in deadline_input:
        return default_date + timedelta(days=2)
    elif 'in 3 days' in deadline_input:
        return default_date + timedelta(days=3)
    elif 'next month' in deadline_input:
        return default_date + timedelta(days=30)
    else:
        # Default to today if no valid deadline is found
        return default_date

POSTGRES_URI = "postgresql://neondb_owner:npg_LgZ4en1QpsyD@ep-broad-bush-a5xjz0wm-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
conn_pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=POSTGRES_URI)

def execute_query(sql_query: str, params=None):
    conn = None
    cur = None
    try:
        # Get connection from the pool
        conn = conn_pool.getconn()
        cur = conn.cursor()

        if sql_query.strip().upper().startswith("SELECT"):
            cur.execute(sql_query, params or ())
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return columns, rows
        else:
            cur.execute(sql_query, params or ())
            affected_rows = cur.rowcount
            conn.commit()
            return None, affected_rows
            
    except psycopg2.Error as e:
        st.error(f"SQL error: {e}")
        return None, None
    finally:
        if cur:
            cur.close()
        if conn:
            conn_pool.putconn(conn)  # Return connection to the pool

def classify_action(prompt: str) -> str:
    """Classify user intent into add/view/update actions using LLM."""
    system_prompt = (
        "Classify the user's database request into one of: add, view, or update. "
        "Respond ONLY with the action keyword. Rules:\n"
        "- 'add' for creating new records (insert)\n"
        "- 'view' for read operations (select)\n"
        "- 'update' for modifying existing records\n"
        "Examples:\n"
        "User: Add new contact -> add\n"
        "User: Show tasks -> view\n"
        "User: Change email -> update\n"
        "User: List contacts in NY -> view\n"
        "User: Mark task 5 completed -> update"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        action = response.content.strip().lower()
        return action if action in ['add', 'view', 'update'] else 'view'
    except Exception as e:
        st.error(f"Error classifying action: {e}")
        return 'view'

def format_response(action: str, sql_query: str, rowcount: int = None, data: tuple = None):
    """Format the response based on the action."""
    responses = {
        "add": lambda: f"✅ Successfully added {rowcount} record(s)",
        "update": lambda: f"✅ Successfully updated {rowcount} record(s)",
        "view": lambda: (f"🔍 Found {len(data[1])} results:", data)
    }
    return responses[action]()


# Streamlit UI Setup
st.sidebar.title("Navigation")

# Modify the page selection section
if 'target_page' not in st.session_state:
    st.session_state.target_page = "🏠 Home"

# Override page selection if redirected
if st.session_state.target_page != "🏠 Home":
    page = st.session_state.target_page
else:
    page = st.sidebar.radio("Go to", ["🏠 Home",
                                      "📝 New Contact",
                                      "✅ New Task",
                                      "📅 Gantt Chart",
                                      "📧 Send Email",
                                      "🤖 Resource Bot",
                                      "📊 View All Data",
                                      "🚀 PerformX",
                                      ]
                            )
# Home Page
if page == "🏠 Home":
    
    if 'task_status' in st.session_state:
        st.success(st.session_state.task_status)
        del st.session_state.task_status
    
    st.header("💬 Database Chat Assistant")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main chat logic
    if prompt := st.chat_input("What would you like to do?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Check if it's a task creation command
        if any(keyword in prompt.lower() for keyword in ["add task", "create task", "new task"]):
            # Parse parameters and redirect to task page
            params = generate_task_details(prompt)
            if params:
                st.session_state.prefill_task = {
                    'title': params.get('title', ''),
                    'description': params.get('description', ''),
                    'category': params.get('category', 'Work'),
                    'priority': params.get('priority', 'Medium'),
                    'deadline': params.get('deadline', ''),
                    'assigned_to': params.get('assigned_to', ''),
                    'status': params.get('status', 'Not Started'),
                    'estimated_time': params.get('estimated_time', ''),
                    'required_resources': params.get('required_resources', ''),
                    'dependencies': params.get('dependencies', ''),
                    'instructions': params.get('instructions', ''),
                    'review_process': params.get('review_process', ''),
                    'performance_metrics': params.get('performance_metrics', ''),
                    'notes': params.get('notes', '')
                }
                st.session_state.target_page = "✅ New Task"
                st.rerun()
            else:
                st.error("Could not parse task details from your request")
        else:
            # Existing classification and SQL handling
            action_type = classify_action(prompt)
            sql_query = generate_sql_query(prompt, action_type)
            
            if sql_query:
                # st.session_state.messages.append({"role": "assistant", "content": f"Generated SQL:\n```sql\n{sql_query}\n```"})  # Debugging
                
                # Execute query
                columns, result = execute_query(sql_query)

                # Format response
                if action_type == "view" and columns:
                    response_text = format_response(action_type, sql_query, data=(columns, result))
                    df = pd.DataFrame(result, columns=columns)
                    response = f"{response_text[0]}\n\n{df.to_markdown(index=False)}"
                elif action_type in ["add", "update"]:
                    response = format_response(action_type, sql_query, rowcount=result)
                else:
                    response = "❌ No results found or invalid query"

                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# New Contact Page
elif page == "📝 New Contact":
    st.header("📝 Create New Contact")
    
    with st.form("contact_form", clear_on_submit=True):
        cols = st.columns(2)
        with cols[0]:
            name = st.text_input("Full Name*", help="Required field")
            phone = st.text_input("Phone Number*", max_chars=10, help="10 digits without country code")
            skills = st.text_input("Skills")
            # skills = st.text_input("Enter skills")
        with cols[1]:
            email = st.text_input("Email Address*")
            address = st.text_input("Physical Address")
        
        submitted = st.form_submit_button("💾 Save Contact")
        
        if submitted:
            if not all([name, phone, email]):
                st.error("Please fill required fields (*)")
            elif not phone.isdigit() or len(phone) != 10:
                st.error("Phone must be 10 digits")
            else:
                try:
                    sql = """INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS, SKILLS) VALUES (%s, %s, %s, %s, %s)"""  # ✅ Use %s for PostgreSQL
                    params = (name.strip(), int(phone), email.strip(), address.strip(), skills.strip() or None)  # ✅ Convert empty address to None
                    _, affected = execute_query(sql, params)
                    if affected:
                        st.success("Contact created successfully!")
                        st.balloons()
                    else:
                        st.error("Error creating contact")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")


# New Task Page
elif page == "✅ New Task":
    st.header("✅ Create New Task")

    # Initialize prefill_task if not exists
    if 'prefill_task' not in st.session_state:
        st.session_state.prefill_task = {}

    # Get contacts for assignment
    conn = get_db_connection()  # Use your PostgreSQL connection function
    cur = conn.cursor()
    cur.execute('SELECT ID, NAME FROM CONTACTS ORDER BY NAME')
    contacts = cur.fetchall()
    contact_names = [name for _, name in contacts]
    contact_dict = {name: id for id, name in contacts}
    cur.close()
    conn.close()

    with st.form("task_form", clear_on_submit=True):
        # Basic Info
        col1, col2 = st.columns([2, 1])
        with col1:
            title = st.text_input("Task Title*", 
                                  value=st.session_state.prefill_task.get('title', ''))
            description = st.text_area("Detailed Description", 
                                       value=st.session_state.prefill_task.get('description', ''),
                                       height=100)
        with col2:
            # Handle natural language deadlines
            deadline_input = st.session_state.prefill_task.get('deadline', '')
            default_date = datetime.today()

            if deadline_input:
                try:
                    if 'tomorrow' in deadline_input.lower():
                        default_date += timedelta(days=1)
                    elif 'next week' in deadline_input.lower():
                        default_date += timedelta(weeks=1)
                    elif 'in 2 days' in deadline_input.lower():
                        default_date += timedelta(days=2)
                except Exception:
                    pass

            deadline_date = st.date_input("Deadline Date*", 
                                          min_value=datetime.today(),
                                          value=default_date)
            deadline_time = st.time_input("Deadline Time*", datetime.now().time())
        
        # Task Metadata
        st.subheader("Task Details", divider="rainbow")
        cols = st.columns(3)
        with cols[0]:
            category_options = ["Work", "Personal", "Project", "Other"]
            category = st.selectbox("Category", category_options,
                                    index=category_options.index(
                                        st.session_state.prefill_task.get('category', 'Work')))
            priority = st.select_slider("Priority*", options=["Low", "Medium", "High"],
                                        value=st.session_state.prefill_task.get('priority', 'Medium'))
            expected_outcome = st.text_input("Expected Outcome", 
                                             value=st.session_state.prefill_task.get('expected_outcome', ''),
                                             placeholder="e.g., Complete project setup")
        with cols[1]:
            assigned_to_index = 0
            if st.session_state.prefill_task.get('assigned_to'):
                lower_names = [name.lower() for name in contact_names]
                try:
                    assigned_to_index = lower_names.index(
                        st.session_state.prefill_task['assigned_to'].lower())
                except ValueError:
                    assigned_to_index = 0
            
            assigned_to = st.selectbox("Assign To*", 
                                       options=contact_names,
                                       index=assigned_to_index)
            
            status_options = ["Not Started", "In Progress", "On Hold", "Completed"]
            status_index = status_options.index(
                st.session_state.prefill_task.get('status', 'Not Started'))
            status = st.selectbox("Status*", status_options, index=status_index)
            
            support_index = 0
            if st.session_state.prefill_task.get('support_contact'):
                try:
                    support_index = contact_names.index(
                        st.session_state.prefill_task['support_contact'])
                except ValueError:
                    support_index = 0
            
            support_contact = st.selectbox("Support Contact", 
                                           options=contact_names,
                                           index=support_index)
        with cols[2]:
            estimated_time = st.text_input("Estimated Time", 
                                           value=st.session_state.prefill_task.get('estimated_time', ''),
                                           placeholder="e.g., 2 hours")
            required_resources = st.text_input("Required Resources",
                                               value=st.session_state.prefill_task.get('required_resources', ''),
                                               placeholder="e.g., Laptop, Software")
        
        # Additional Details
        st.subheader("Additional Information", divider="rainbow")
        dependencies = st.text_area("Dependencies", 
                                    value=st.session_state.prefill_task.get('dependencies', ''),
                                    placeholder="What does this task depend on?")
        instructions = st.text_area("Instructions", 
                                    value=st.session_state.prefill_task.get('instructions', ''),
                                    placeholder="Step-by-step instructions")
        review_process = st.text_area("Review Process", 
                                      value=st.session_state.prefill_task.get('review_process', ''),
                                      placeholder="Steps for reviewing the task")
        performance_metrics = st.text_area("Success Metrics",
                                           value=st.session_state.prefill_task.get('performance_metrics', ''),
                                           placeholder="How will success be measured?")
        notes = st.text_area("Internal Notes", 
                             value=st.session_state.prefill_task.get('notes', ''),
                             placeholder="Any additional notes")
        
        # Proper submit button
        submitted = st.form_submit_button("🚀 Create Task")
        
        if submitted:
            if not all([title, priority, assigned_to, status, deadline_date]):
                st.error("Please fill required fields (*)")
            else:
                try:
                    deadline = datetime.combine(deadline_date, deadline_time).strftime("%Y-%m-%d %H:%M:%S")
                    sql = """INSERT INTO TASKS (
                        TITLE, DESCRIPTION, CATEGORY, PRIORITY, EXPECTED_OUTCOME,
                        DEADLINE, ASSIGNED_TO, DEPENDENCIES, REQUIRED_RESOURCES,
                        ESTIMATED_TIME, INSTRUCTIONS, REVIEW_PROCESS, PERFORMANCE_METRICS,
                        SUPPORT_CONTACT, NOTES, STATUS, STARTED_AT, COMPLETED_AT
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                    
                    params = (
                        title.strip(),
                        description.strip(),
                        category,
                        priority,
                        expected_outcome.strip(),
                        deadline,
                        contact_dict[assigned_to],
                        dependencies.strip(),
                        required_resources.strip(),
                        estimated_time.strip(),
                        instructions.strip(),
                        review_process.strip(),
                        performance_metrics.strip(),
                        contact_dict.get(support_contact, None),
                        notes.strip(),
                        status,
                        None,  # STARTED_AT
                        None   # COMPLETED_AT
                    )
                    
                    _, affected = execute_query(sql, params)
                    if affected:
                        st.session_state.task_status = f"✅ Task '{title.strip()}' created successfully!"
                        st.session_state.target_page = "🏠 Home"
                        if 'prefill_task' in st.session_state:
                            del st.session_state.prefill_task
                        st.rerun()
                    else:
                        st.error("Error creating task")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")


# Send emails
elif page == " Send Email":
    st.header(" Send Email to Assigned Contacts")

    # Get all tasks and their assigned contacts from PostgreSQL
    try:
        conn = get_db_connection()
        tasks_df = pd.read_sql("""
            SELECT T.ID, T.TITLE, T.DESCRIPTION, C.NAME, C.EMAIL 
            FROM TASKS T 
            JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID
        """, conn)
        conn.close()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        tasks_df = pd.DataFrame()

    if tasks_df.empty:
        st.warning("No tasks with assigned contacts found.")
    else:
        # Select a task
        task_options = tasks_df.apply(lambda row: f"{row['TITLE']} (Assigned to: {row['NAME']})", axis=1)
        selected_task = st.selectbox("Select a Task", task_options)

        if selected_task:
            # Extract task details
            selected_index = task_options.tolist().index(selected_task)
            task_id = tasks_df.iloc[selected_index]['ID']
            task_title = tasks_df.iloc[selected_index]['TITLE']
            task_description = tasks_df.iloc[selected_index]['DESCRIPTION']
            recipient_name = tasks_df.iloc[selected_index]['NAME']
            recipient_email = tasks_df.iloc[selected_index]['EMAIL']

            st.subheader("Email Details", divider="rainbow")
            st.write(f"**Recipient:** {recipient_name} ({recipient_email})")
            st.write(f"**Task:** {task_title}")

            # Generate email content using LLM
            if st.button("\u2728 Generate Email Content"):
                with st.spinner("Generating email content..."):
                    try:
                        system_prompt = """You are an AI assistant that writes professional emails. 
                        Write a concise email to remind the recipient about their assigned task. 
                        Include the task title, description, and a polite reminder to complete it on time."""

                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"Task: {task_title}\nDescription: {task_description}")
                        ]

                        response = llm.invoke(messages)
                        email_content = response.content.strip()
                        st.session_state.email_content = email_content
                    except Exception as e:
                        st.error(f"Error generating email content: {str(e)}")

            # Display and edit email content
            if 'email_content' in st.session_state:
                email_content = st.text_area("Email Content", 
                                           value=st.session_state.email_content,
                                           height=200)

                # Send email
                if st.button("\ud83d\udce4 Send Email"):
                    try:
                        send_task_email(
                            recipient_email=recipient_email,
                            subject=f"Reminder: {task_title}",
                            message=email_content
                        )
                        st.success(f"Email sent to {recipient_name} ({recipient_email})")
                        del st.session_state.email_content  # Clear generated content
                    except Exception as e:
                        st.error(f"Failed to send email: {str(e)}")

# Gantt Chart
elif page == "📅 Gantt Chart":
    st.header("📅 Task Timeline Visualization")
    
    @st.cache_data
    def get_tasks_for_gantt():
        try:
            conn = get_db_connection()  # Use your PostgreSQL connection function
            query = """
                SELECT 
                    title AS "TITLE", 
                    deadline AS "DEADLINE",
                    status AS "STATUS",
                    priority AS "PRIORITY",
                    category AS "CATEGORY"
                FROM tasks
            """
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Convert and calculate dates
            df["DEADLINE"] = pd.to_datetime(df["DEADLINE"], errors='coerce')
            df = df.dropna(subset=["DEADLINE"])
            df["START_DATE"] = df["DEADLINE"] - pd.Timedelta(days=5)
            
            return df
        
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame()

    tasks_df = get_tasks_for_gantt()
    
    if tasks_df.empty:
        st.warning("No tasks with valid deadlines found in the database.")
    else:
        # Create interactive filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_status = st.multiselect(
                "Filter by Status",
                options=tasks_df['STATUS'].unique(),
                default=tasks_df['STATUS'].unique()
            )
        with col2:
            selected_priority = st.multiselect(
                "Filter by Priority",
                options=tasks_df['PRIORITY'].unique(),
                default=tasks_df['PRIORITY'].unique()
            )
        with col3:
            selected_category = st.multiselect(
                "Filter by Category",
                options=tasks_df['CATEGORY'].unique(),
                default=tasks_df['CATEGORY'].unique()
            )

        # Apply filters
        filtered_df = tasks_df[
            (tasks_df['STATUS'].isin(selected_status)) &
            (tasks_df['PRIORITY'].isin(selected_priority)) &
            (tasks_df['CATEGORY'].isin(selected_category))
        ]

        # Create Gantt chart with Plotly
        fig = px.timeline(
            filtered_df,
            x_start="START_DATE",
            x_end="DEADLINE",
            y="TITLE",
            color="PRIORITY",
            color_discrete_map={
                "High": "#FF4B4B",
                "Medium": "#FFA500",
                "Low": "#00CC96"
            },
            title="Task Schedule Overview",
            labels={"TITLE": "Task Name"},
            hover_data=["STATUS", "CATEGORY"]
        )

        # Customize layout
        fig.update_layout(
            height=600,
            xaxis_title="Timeline",
            yaxis_title="Tasks",
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={'categoryorder': 'total ascending'}
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data summary
        st.subheader("📊 Task Statistics", divider="grey")
        cols = st.columns(4)
        cols[0].metric("Total Tasks", len(filtered_df))
        cols[1].metric("Earliest Deadline", filtered_df['DEADLINE'].min().strftime("%Y-%m-%d"))
        cols[2].metric("Latest Deadline", filtered_df['DEADLINE'].max().strftime("%Y-%m-%d"))
        cols[3].metric("Avg Duration", f"{(filtered_df['DEADLINE'] - filtered_df['START_DATE']).mean().days} days")



elif page == "🤖 Resource Bot":
    st.header("🤖 Resource Manager & Chat History")
    
    tab1, tab2 = st.tabs(["Chatbot", "Chat History"])
    
    with tab1:
        # Chatbot Section
        with st.container(border=True):
            st.subheader("Task Assistant", divider="rainbow")
            user_input = st.text_input("Enter a task (e.g., website development, marketing campaign):")
            
            if st.button("Get Suggestions", use_container_width=True):
                if user_input:
                    with st.spinner("Analyzing task and resources..."):
                        # Initialize chat history in session if not exists
                        if "chat_history" not in st.session_state:
                            st.session_state.chat_history = []
                            
                        # Generate prompt and response
                        prompt = generate_prompt(user_input)
                        
                        # Use existing chat history for context
                        bot_response = query_groq(prompt)
                        
                        # Keep only the last 3 conversations
                        MAX_HISTORY = 3  
                        if len(st.session_state.chat_history) > MAX_HISTORY:
                            st.session_state.chat_history.pop(0)
                            
                        # Add to session memory
                        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
                        
                        st.session_state.current_suggestion = bot_response
                else:
                    st.warning("Please enter a task description")
            
            if "current_suggestion" in st.session_state:
                with st.expander("Current Suggestion", expanded=True):
                    st.write(st.session_state.current_suggestion)
                    
                if st.button("Save to History", use_container_width=True):
                    if user_input and "current_suggestion" in st.session_state:
                        save_suggestion(user_input, st.session_state.current_suggestion)
                        st.rerun()
            
            # Show recent conversations
            if "chat_history" in st.session_state and st.session_state.chat_history:
                st.subheader("Recent Conversations", divider="gray")
                for chat in reversed(st.session_state.chat_history):
                    with st.expander(f"Task: {chat['user'][:50]}...", expanded=False):
                        st.text_area("You asked:", chat["user"], height=70, disabled=True)
                        st.text_area("Bot response:", chat["bot"], height=150, disabled=True)
    
    with tab2:
        # Chat History Section
        with st.container(border=True):
            st.subheader("Saved Suggestions", divider="rainbow")
            chat_files = list_saved_chats()
            
            if chat_files:
                selected_file = st.selectbox("Select saved suggestion:", chat_files)
                
                cols = st.columns([1, 4])
                with cols[0]:
                    if st.button("Load Selected", help="Load this suggestion into the assistant"):
                        try:
                            with open(os.path.join(CHAT_HISTORY_DIR, selected_file), "r", encoding="utf-8") as file:
                                content = file.read().strip()
                            st.session_state.current_suggestion = content
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading file: {e}")
                
                with cols[1]:
                    if selected_file:
                        try:
                            with open(os.path.join(CHAT_HISTORY_DIR, selected_file), "r", encoding="utf-8") as file:
                                content = file.read().strip()
                            st.text_area("Suggestion Content", value=content, height=200, key=f"view_{selected_file}")
                        except Exception as e:
                            st.error(f"Error reading file: {e}")
            else:
                st.info("No saved suggestions yet. Get some recommendations first!")

# All Data Page
elif page == "📊 View All Data":
    st.header("📊 View All Database Records")
    
    try:
        conn = get_db_connection() # Use your PostgreSQL connection function
        
        # Contacts Table
        st.subheader("👥 Contacts", divider="rainbow")
        df_contacts = pd.read_sql("SELECT * FROM CONTACTS ORDER BY NAME", conn)
        if not df_contacts.empty:
            st.dataframe(
                df_contacts,
                column_config={
                    "PHONE": st.column_config.NumberColumn(format="%d")
                },
                use_container_width=True
            )
            st.download_button(
                label="📥 Export Contacts",
                data=df_contacts.to_csv(index=False),
                file_name="contacts.csv",
                mime="text/csv"
            )
        else:
            st.warning("No contacts found in database")

        # Tasks Table
        st.subheader("✅ Tasks", divider="rainbow")
        df_tasks = pd.read_sql("""SELECT 
            T.ID,
            T.TITLE,
            T.DESCRIPTION,
            T.CATEGORY,
            T.PRIORITY,
            T.STATUS,
            T.DEADLINE,
            C.NAME AS ASSIGNED_TO
            FROM TASKS T
            LEFT JOIN CONTACTS C ON T.ASSIGNED_TO = C.ID
            ORDER BY DEADLINE""", conn)
        
        if not df_tasks.empty:
            st.dataframe(
                df_tasks,
                column_config={
                    "DEADLINE": st.column_config.DatetimeColumn(),
                    "PRIORITY": st.column_config.SelectboxColumn(options=["Low", "Medium", "High"])
                },
                use_container_width=True
            )
            st.download_button(
                label="📥 Export Tasks",
                data=df_tasks.to_csv(index=False),
                file_name="tasks.csv",
                mime="text/csv"
            )
        else:
            st.warning("No tasks found in database")


    except Error as e:  # Properly using psycopg2.Error for PostgreSQL
        st.error(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


#  perform X page
elif page == "🚀 PerformX":
    st.header("🚀 Performance Tracking Dashboard")
    
    # Add meta tag for redirection with delay
    st.markdown("""
    <meta http-equiv="refresh" content="0; url='https://performtrackdb.streamlit.app/'">
    """, unsafe_allow_html=True)
    
    # Show loading message
    
    # Fallback link
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <h3>⏳ Redirecting to Performance Dashboard</h3>
        <p>If you're not redirected automatically, click below:</p>
        <a href="https://performtrack.streamlit.app/" target="_blank">
            <button style="
                background: #FF4B4B;
                color: white;
                padding: 1em 2em;
                border: none;
                border-radius: 10px;
                font-size: 1.2rem;
                cursor: pointer;
                margin-top: 1rem;">
                🔥 Open Performance Dashboard
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Examples Guide
st.sidebar.markdown("### Examples Guide")
st.sidebar.markdown("""
**Add Data Examples:**
- "Add new contact: John, 5551234567, john@email.com, London"
- "Create task: Project Setup, Initialize repo, 2024-12-31, 5551234567"

**View Data Examples:**
- "Show contacts from Delhi"
- "List ongoing tasks for John"
- "Display completed tasks"

**Update Data Examples:**
- "Change John's email to new@email.com"
- "Mark task 5 as completed"
- "Update task 3's due date to tomorrow"
""")

if st.button("Push Database Changes to GitHub"):
    print('hello')