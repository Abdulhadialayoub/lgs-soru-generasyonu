import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                sslmode=os.getenv("SSL_MODE", "require")
            )
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
    
    def execute_query(self, query, params=None):
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Query execution error: {e}")
            return None
    
    def execute_single_query(self, query, params=None):
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Query execution error: {e}")
            return None

# Database instance
db = Database()