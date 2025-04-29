import pymysql as mysql

# Connect to MySQL
conn = mysql.connect(
    host="localhost",
    user="root",
    password="livewire",
    database="company"
)
cursor = conn.cursor()

# Create Table (if not exists)
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
)
""")

# ALTER COLUMN (Example: Change name column to allow NULLs)
cursor.execute("ALTER TABLE users MODIFY COLUMN name VARCHAR(100) NULL")

# CREATE - Insert Data
def create_user(name, email):
    cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)", (name, email))
    conn.commit()
    print("User created successfully.")

# READ - Get All Users
def get_all_users():
    cursor.execute("SELECT * FROM users")
    for row in cursor.fetchall():
        print(row)

# READ - Get Specific User by Email
def get_user_by_email(email):
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    print("User found:", user if user else "No user found.")

# UPDATE - Update User Email
def update_user_email(old_email, new_email):
    cursor.execute("UPDATE users SET email = %s WHERE email = %s", (new_email, old_email))
    conn.commit()
    print("User updated.")

# DELETE - Delete User by Email
def delete_user_by_email(email):
    cursor.execute("DELETE FROM users WHERE email = %s", (email,))
    conn.commit()
    print("User deleted.")

# Example usage
create_user("Alice", "alice@example.com")
get_all_users()
get_user_by_email("alice@example.com")
update_user_email("alice@example.com", "alice_new@example.com")
delete_user_by_email("alice_new@example.com")


