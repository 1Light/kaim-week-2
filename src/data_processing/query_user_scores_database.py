import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root', 
    'password': os.getenv("POSTGRES_PASSWORD"),  
    'database': 'kaim-week-2'  
}

try:
    # Connect to PostgreSQL
    connection = psycopg2.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    cursor = connection.cursor()

    # Query the top 10 rows ordered by 'Satisfaction Score'
    cursor.execute("SELECT * FROM user_scores ORDER BY \"Satisfaction Score\" DESC LIMIT 10")
    rows = cursor.fetchall()

    # Fetch column names
    column_names = [desc[0] for desc in cursor.description]

    # Print the column names (headlines)
    print(" | ".join(column_names))

    # Print the data
    for row in rows:
        print(" | ".join(map(str, row)))

except Exception as e:
    print(f"Error while querying the database: {e}")
finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()