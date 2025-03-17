import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()



# Neon connection with SSL
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    port=os.getenv('DB_PORT'),
    sslmode='require'
)
cursor = conn.cursor()

# Create Contacts table (Neon-compatible)
cursor.execute("""
CREATE TABLE IF NOT EXISTS CONTACTS (
    ID SERIAL PRIMARY KEY,
    NAME VARCHAR(100) NOT NULL,
    PHONE BIGINT UNIQUE CHECK(PHONE BETWEEN 1000000000 AND 9999999999),
    EMAIL VARCHAR(100) UNIQUE,
    ADDRESS TEXT,
    SKILLS TEXT NOT NULL
);
""")

# Insert contacts (using ON CONFLICT DO NOTHING)
contacts_data = [
    ('Ansh', 9876543210, 'vivekchoudhary75@gmail.com', '123, Lorem Ipsum Street, New York, NY 10001', 'Project Management'),
    ('Vivek', 9999701072, 'vivekchoudhary765@gmail.com', 'Delhi', 'Python, SQL, Data Analysis'),
    ('Akshit', 9999701034, 'vivekchoudhary565@gmail.com', 'Delhi', 'Web Development, JavaScript'),
    ('Vivek Choudhary', 9999701071, 'vivekchoudhary789@gmail.com', 'Delhi 110088', 'Team Leadership, Strategic Planning'),
    ('John Doe', 5551234123, 'john@example.com', '123 Main St', 'UI/UX Design, Graphic Design'),
    ('Ishani', 1234567890, 'john.new@example.com', '456 New St', 'Digital Marketing, SEO'),
    ('Vaibhav', 9999807097, 'vaibhav@gmail.com', 'Jaipur', 'Cloud Computing, DevOps'),
    ('Mohit', 9920128977, 'mohit@gmail.com', 'Mumbai', 'Mobile Development, Flutter')
]

insert_contact = """
INSERT INTO CONTACTS (NAME, PHONE, EMAIL, ADDRESS, SKILLS) 
VALUES (%s, %s, %s, %s, %s) ON CONFLICT (PHONE) DO NOTHING
"""
cursor.executemany(insert_contact, contacts_data)

# Retrieve contact IDs based on phone numbers
cursor.execute("SELECT PHONE, ID FROM CONTACTS")
contact_ids = {row[0]: row[1] for row in cursor.fetchall()}

# Create Tasks table
cursor.execute("""
CREATE TABLE IF NOT EXISTS TASKS (
    ID SERIAL PRIMARY KEY,
    TITLE VARCHAR(255) NOT NULL,
    DESCRIPTION TEXT,
    CATEGORY VARCHAR(50),
    PRIORITY TEXT CHECK(PRIORITY IN ('Low', 'Medium', 'High')),
    EXPECTED_OUTCOME TEXT,
    DEADLINE TIMESTAMP NOT NULL,
    ASSIGNED_TO INT NOT NULL REFERENCES CONTACTS(ID),
    DEPENDENCIES TEXT,
    REQUIRED_RESOURCES TEXT,
    ESTIMATED_TIME TEXT NOT NULL,
    INSTRUCTIONS TEXT,
    REVIEW_PROCESS TEXT,
    PERFORMANCE_METRICS TEXT,
    SUPPORT_CONTACT INT REFERENCES CONTACTS(ID),
    NOTES TEXT,
    STATUS TEXT CHECK(STATUS IN ('Not Started', 'In Progress', 'On Hold', 'Completed', 'Reviewed & Approved')) NOT NULL DEFAULT 'Not Started',
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    STARTED_AT TIMESTAMP,
    COMPLETED_AT TIMESTAMP
);
""")

# Insert tasks data
tasks_data = [
    ('Project Planning', 'Plan the initial phase of the project', 'Project Management', 'High', 'Completed project plan document', '2025-03-01 23:59', contact_ids[9999701072], 'None', 'Project management software', '1 week', '1. Define scope\n2. Identify stakeholders', 'Review by project manager', 'Adherence to timeline', contact_ids[9999701072], 'Critical initial task', 'In Progress', '2024-01-15 09:00', '2025-02-25 10:00', None),
    ('Database Setup', 'Set up the database schema and tables', 'Technical', 'Medium', 'Functional database system', '2025-03-05 18:00', contact_ids[9999701034], 'Project Planning', 'SQL tools, Server access', '3 days', '1. Create schema\n2. Define tables', 'Review by lead developer', 'Schema normalization', contact_ids[9999701034], 'Ensure backup strategy', 'Not Started', '2024-01-16 10:30', None, None),
    ('UI Design', 'Design the user interface', 'Creative', 'Medium', 'Approved UI mockups', '2025-03-10 12:00', contact_ids[9999807097], 'Database Setup', 'Design software', '2 weeks', '1. Wireframe\n2. Prototype', 'Client review', 'User feedback score', contact_ids[9999807097], 'Mobile-first approach', 'In Progress', '2025-03-01 12:30', '2025-03-01 14:30', None),
    ('Testing', 'Perform unit and integration testing', 'QA', 'High', 'Test report', '2025-03-15 17:00', contact_ids[9920128977], 'UI Design', 'Testing frameworks', '5 days', '1. Write test cases\n2. Execute tests', 'QA manager review', 'Bug count', contact_ids[9920128977], 'Automate where possible', 'Not Started', '2025-03-01 12:30', None, None),
    ('Deployment', 'Deploy application to production', 'Operations', 'High', 'Successful deployment', '2025-03-20 20:00', contact_ids[5551234123], 'Testing', 'Cloud access', '2 days', '1. Prepare environment\n2. Deploy', 'Ops team review', 'Downtime duration', contact_ids[5551234123], 'Monitor post-deployment', 'Completed', '2025-03-10 09:00', '2025-03-18 09:00', '2025-03-20 15:30')
]

insert_task = """
INSERT INTO TASKS (TITLE, DESCRIPTION, CATEGORY, PRIORITY, EXPECTED_OUTCOME, DEADLINE, ASSIGNED_TO, DEPENDENCIES, REQUIRED_RESOURCES, ESTIMATED_TIME, INSTRUCTIONS, REVIEW_PROCESS, PERFORMANCE_METRICS, SUPPORT_CONTACT, NOTES, STATUS, CREATED_AT, STARTED_AT, COMPLETED_AT)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
cursor.executemany(insert_task, tasks_data)

conn.commit()
conn.close()
