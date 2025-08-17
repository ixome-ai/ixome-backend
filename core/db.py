import sqlite3
import logging

logger = logging.getLogger(__name__)

def init_db():
    """Initialize the SQLite database and create dealer_info table."""
    try:
        conn = sqlite3.connect('/home/vincent/ixome/core/database.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dealer_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT,
                solution TEXT,
                component TEXT
            )
        ''')
        conn.commit()
        logger.info("Initialized dealer_info table")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
        conn.close()

def insert_dealer_info(brand: str, solution: str, component: str):
    """Insert dealer info into SQLite database."""
    try:
        conn = sqlite3.connect('/home/vincent/ixome/core/database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO dealer_info (brand, solution, component) VALUES (?, ?, ?)', (brand, solution, component))
        conn.commit()
        logger.info(f"Inserted dealer info: brand={brand}, component={component}")
    except Exception as e:
        logger.error(f"Error inserting dealer info: {e}")
    finally:
        conn.close()

def query_sqlite(brand: str, component: str) -> str:
    """Query dealer info from SQLite database."""
    try:
        conn = sqlite3.connect('/home/vincent/ixome/core/database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT solution FROM dealer_info WHERE brand = ? AND component = ?', (brand, component))
        result = cursor.fetchone()
        if result:
            logger.info(f"Found dealer info for brand={brand}, component={component}")
            return result[0]
        logger.info(f"No dealer info found for brand={brand}, component={component}")
        return "No exact match found in dealer database."
    except Exception as e:
        logger.error(f"Error querying dealer info: {e}")
        return f"Error querying database: {str(e)}"
    finally:
        conn.close()

# Initialize database on import
init_db()