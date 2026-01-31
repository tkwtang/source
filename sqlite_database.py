
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types.
    Ensures scientific constants (like np.pi) and arrays are serializable.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DatabaseManager:
    """
    Dedicated manager for storing and retrieving physics simulation results.
    Aligned to use 'experiment_id' as the primary identifier.
    """
    def __init__(self, db_path: str = "simulations.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the database schema for simulation results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sim_results (
                        id TEXT PRIMARY KEY,
                        config TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database setup error: {e}")

    def get_cursor(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                return cursor
        except sqlite3.Error as e:
            print(f"Database setup error: {e}")


    def save_simulation(self, data: Dict[str, Any]):
        """
        Saves simulation data using the 'experiment_id' found inside the dictionary.
        Uses UPSERT logic (Insert or Replace) to update existing records.
        """
        exp_id = data.get("experiment_id")

        if not exp_id:
            print("Error: 'experiment_id' is missing from the data dictionary. Save aborted.")
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                json_data = json.dumps(
                    data, 
                    cls=NumpyEncoder,
                    indent=4, 
                    ensure_ascii=False
                )
                
                cursor.execute(
                    "INSERT OR REPLACE INTO sim_results (id, config, timestamp) VALUES (?, ?, ?)",
                    (
                        exp_id,
                        json_data,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                conn.commit()
            print(f"Simulation saved successfully: {exp_id}")
        except sqlite3.Error as e:
            print(f"Failed to save simulation '{exp_id}': {e}")

    def load_simulation(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a simulation dictionary using ONLY the experiment_id string.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT config FROM sim_results WHERE experiment_id = ?", (experiment_id,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            
            print(f"No simulation found with ID: {experiment_id}")
            return None
        except sqlite3.Error as e:
            print(f"Failed to load simulation '{experiment_id}': {e}")
            return None

    def get_all_simulations(self) -> List[Dict[str, Any]]:
        """
        Retrieves all simulation records stored in the database.
        Returns a list of all simulation dictionaries.
        """
        all_sims = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT config FROM sim_results ORDER BY timestamp DESC")
                rows = cursor.fetchall()
                
                for row in rows:
                    all_sims.append(json.loads(row[0]))
                    
            return all_sims
        except sqlite3.Error as e:
            print(f"Failed to retrieve all simulations: {e}")
            return []