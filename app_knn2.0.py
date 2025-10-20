
import os
import math
import random
import re
import csv
import io
import json
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from collections import Counter, defaultdict

from flask import Flask, request, render_template_string, Response, jsonify
import numpy as np
import mysql.connector
from mysql.connector import Error
from urllib.parse import urlparse

#  CONFIG

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root") 
DB_PASS = os.environ.get("DB_PASS", "") 
DB_NAME = os.environ.get("DB_NAME", "travel_etl")
DATABASE_URL = os.environ.get("DATABASE_URL") 

TABLE_NAME_OVERRIDE = os.environ.get("TABLE_NAME")
TARGET_FIELD_OVERRIDE = os.environ.get("TARGET_FIELD")

PRICE_CANDIDATES = [
    "price_eur_norm", "price_eur_per_person",
    "price_rsd_norm", "price_rsd_per_person", 
    "price_eur", "price_rsd"
]

MIN_NIGHTS = int(os.environ.get("MIN_NIGHTS", "1"))
MAX_NIGHTS = int(os.environ.get("MAX_NIGHTS", "30"))
MIN_STARS = int(os.environ.get("MIN_STARS", "0"))
MAX_STARS = int(os.environ.get("MAX_STARS", "5"))

TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.2"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

LEAD_TIME_MAX = int(os.environ.get("LEAD_TIME_MAX", "365"))
CITY_MIN_FREQ = int(os.environ.get("CITY_MIN_FREQ", "25"))
HOTEL_MIN_FREQ = int(os.environ.get("HOTEL_MIN_FREQ", "25"))
ROOM_MIN_FREQ = int(os.environ.get("ROOM_MIN_FREQ", "30"))
TRANSP_MIN_FREQ = int(os.environ.get("TRANSP_MIN_FREQ", "30"))

# KNN hiperparametri
K_VALUE = int(os.environ.get("KNN_K", "11")) 
USE_DISTANCE_WEIGHTING = os.environ.get("KNN_WEIGHTED", "1") == "1"  

HIST_DEFAULT_PER = int(os.environ.get("HIST_DEFAULT_PER", "25"))
HIST_MAX_PER = int(os.environ.get("HIST_MAX_PER", "200"))


BASE_STYLE = """
  body {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; 
    max-width: 1100px; 
    margin: 2rem auto; 
    padding: 0 1rem;
    line-height: 1.5;
  }
  .card {
    border: 1px solid #ddd; 
    border-radius: 12px; 
    padding: 1rem; 
    margin-top: 1rem;
    background: #fafafa;
  }
  label {
    display: block; 
    margin-top: 0.5rem; 
    font-weight: 600;
    color: #333;
  }
  input, select {
    width: 100%; 
    padding: 0.5rem; 
    margin-top: 0.25rem;
    border: 1px solid #ccc;
    border-radius: 6px;
  }
  button {
    padding: 0.6rem 1rem; 
    margin-top: 1rem; 
    border: 0; 
    border-radius: 10px; 
    background: #111; 
    color: #fff; 
    cursor: pointer;
    font-weight: 500;
  }
  button:hover { background: #333; }
  .grid {
    display: grid; 
    gap: 1rem; 
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  }
  .kpi {
    display: flex; 
    gap: 1rem; 
    flex-wrap: wrap;
  }
  .kpi div {
    background: #fff;
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    min-width: 100px;
  }
  .pred {
    font-size: 1.2rem; 
    font-weight: 700; 
    margin-top: 0.5rem;
    color: #d63384;
  }
  nav {
    display: flex; 
    gap: 0.75rem; 
    align-items: center; 
    flex-wrap: wrap; 
    margin: 0.5rem 0 0 0;
  }
  nav a {
    text-decoration: none; 
    color: #111; 
    background: #f3f3f3; 
    border: 1px solid #e6e6e6; 
    border-radius: 9px; 
    padding: 0.4rem 0.7rem;
    transition: all 0.2s;
  }
  nav a:hover { background: #e9e9e9; }
  table {
    width: 100%; 
    border-collapse: collapse;
    margin-top: 1rem;
  }
  th, td {
    border-bottom: 1px solid #eee; 
    padding: 0.5rem 0.4rem; 
    text-align: left; 
    font-size: 0.92rem;
  }
  th {
    background: #fafafa;
    font-weight: 600;
  }
  .row {
    display: flex; 
    gap: 0.75rem; 
    flex-wrap: wrap;
  }
  .muted {
    color: #666; 
    font-size: 0.9rem;
  }
"""

HTML_TPL = """
<!doctype html>
<html lang="sr">
<head>
  <meta charset="utf-8">
  <title>KNN Price Classification - v1.1</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>""" + BASE_STYLE + """</style>
</head>
<body>
  <h1>🔍 KNN Classification of Price Categories</h1>
  <nav>
    <a href="/">🏠 Home</a>
    <a href="/history">🕘 History</a>
    <a href="/history.csv">⬇️ Export CSV</a>
    <a href="/api/history">🧩 JSON API</a>
  </nav>

  <div class="card">
    <div><b>Table:</b> {{ meta.table }} &nbsp;|&nbsp; <b>Target:</b> {{ meta.target }} &nbsp;|&nbsp;
      <b>K:</b> {{ meta.k }} &nbsp;|&nbsp; <b>Distance Weighted:</b> {{ "YES" if meta.weighted else "NO" }}</div>
  </div>

  <div class="card">
    <h3>Predict Price Category</h3>
    <form method="post" action="/predict">
      <div class="grid">
        <div>
          <label>City</label>
          <select name="city" required>
            {% for c in cities %}
            <option value="{{c}}">{{c}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Number of Nights</label>
          <input type="number" name="nights" value="7" min="1" max="30" required>
        </div>
        <div>
          <label>Hotel Stars</label>
          <select name="stars" required>
            {% for s in [0,1,2,3,4,5] %}
            <option value="{{s}}" {% if s==4 %}selected{% endif %}>{{s}} stars</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Board Type</label>
          <select name="board" required>
            {% for b in boards %}
            <option value="{{b}}">{{b}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Transport</label>
          <select name="transport" required>
            {% for t in transports %}
            <option value="{{t}}">{{t}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Room Type</label>
          <select name="room" required>
            {% for r in rooms %}
            <option value="{{r}}">{{r}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Travel Month (1-12)</label>
          <select name="month" required>
            {% for m in range(1,13) %}
            <option value="{{m}}" {% if m==7 %}selected{% endif %}>{{m}}</option>
            {% endfor %}
          </select>
        </div>
      </div>
      <button type="submit">🎯 Predict Price Category</button>
    </form>
  </div>

  {% if prediction is not none %}
  <div class="card">
    <div>Prediction Result:</div>
    <div class="pred">Category {{ prediction }} <span class="muted">({{ label }})</span></div>
    <div class="muted">
      Price ranges: [0: ≤500], [1: 501-1500], [2: 1501-3000], [3: ≥3000] EUR per person. 
      This prediction has been saved to history.
    </div>
  </div>
  {% endif %}

  <div class="card">
    <h3>📊 Model Performance (Test Set)</h3>
    <div class="kpi">
      <div><b>Accuracy</b><br>{{ "%.3f" % metrics.acc }}</div>
      <div><b>Macro F1-Score</b><br>{{ "%.3f" % metrics.f1m }}</div>
      <div><b>Test Samples</b><br>{{ counts.test }}</div>
      <div><b>Training Samples</b><br>{{ counts.train }}</div>
      <div><b>Features</b><br>{{ counts.features }}</div>
    </div>
    <p><b>Class Support:</b> {{ support }}</p>
    <p><b>Confusion Matrix</b> (rows = actual, columns = predicted):<br>
      <code style="font-family: monospace;">{{ cm }}</code>
    </p>
  </div>
</body>
</html>
"""

HTML_HIST = """
<!doctype html>
<html lang="sr">
<head>
  <meta charset="utf-8">
  <title>KNN Prediction History</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>""" + BASE_STYLE + """</style>
</head>
<body>
  <h1>🕘 KNN Prediction History</h1>
  <nav>
    <a href="/">🏠 Home</a>
    <a href="/history.csv">⬇️ Export CSV</a>
    <a href="/api/history">🧩 JSON API</a>
  </nav>

  <div class="card">
    <h3>Filter Results</h3>
    <form method="get" class="row">
      <div style="min-width:190px;">
        <label>City</label>
        <select name="city">
          <option value="">(all cities)</option>
          {% for c in cities %}
          <option value="{{c}}" {% if c==flt.city %}selected{% endif %}>{{c}}</option>
          {% endfor %}
        </select>
      </div>
      <div style="min-width:140px;">
        <label>Month</label>
        <select name="month">
          <option value="">(all months)</option>
          {% for m in range(1,13) %}
          <option value="{{m}}" {% if flt.month == m %}selected{% endif %}>{{m}}</option>
          {% endfor %}
        </select>
      </div>
      <div style="min-width:160px;">
        <label>Per Page</label>
        <input type="number" name="per" value="{{per}}" min="1" max="200">
      </div>
      <div style="min-width:160px;">
        <label>&nbsp;</label>
        <button type="submit">Apply Filters</button>
      </div>
    </form>
  </div>

  <div class="card">
    <div class="muted">Page {{page}} of {{pages}} • Total: {{total}} records</div>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Timestamp</th><th>City</th><th>Hotel</th><th>Nights</th><th>Stars</th>
            <th>Board</th><th>Transport</th><th>Room</th><th>Month</th>
            <th>Class</th><th>Label</th><th>K</th><th>Weighted</th><th>IP</th><th>User Agent</th>
          </tr>
        </thead>
        <tbody>
          {% for it in items %}
          <tr>
            <td>{{it.id}}</td>
            <td>{{it.created_at}}</td>
            <td>{{it.city}}</td>
            <td>{{it.hotel}}</td>
            <td>{{it.nights}}</td>
            <td>{{it.stars}}</td>
            <td>{{it.board}}</td>
            <td>{{it.transport}}</td>
            <td>{{it.room}}</td>
            <td>{{it.month}}</td>
            <td>{{it.pred_class}}</td>
            <td>{{it.pred_label}}</td>
            <td>{{it.k}}</td>
            <td>{{"YES" if it.weighted else "NO"}}</td>
            <td>{{it.ip}}</td>
            <td>{{it.user_agent_short}}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div class="row" style="margin-top: 0.75rem;">
      {% if page > 1 %}
      <a href="{{pager_url(page-1)}}">⬅️ Previous</a>
      {% endif %}
      {% if page < pages %}
      <a href="{{pager_url(page+1)}}">Next ➡️</a>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""



def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def get_db_connection():
    if DATABASE_URL:
        parsed = urlparse(DATABASE_URL)
        username = parsed.username or DB_USER
        password = parsed.password or DB_PASS  
        hostname = parsed.hostname or DB_HOST
        port_num = parsed.port or 3306
        database = (parsed.path or "/").lstrip("/") or DB_NAME
        
        return mysql.connector.connect(
            host=hostname, 
            port=port_num, 
            user=username, 
            password=password, 
            database=database
        )
    else:
        return mysql.connector.connect(
            host=DB_HOST, 
            user=DB_USER, 
            password=DB_PASS, 
            database=DB_NAME
        )

def execute_query(sql_query: str, params: Optional[tuple] = None) -> list:
    """Execute a SELECT query and return all results"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(sql_query, params or ())
        results = cursor.fetchall()
        cursor.close()
        return results
    finally:
        connection.close()

def execute_query_dict(sql_query: str, params: Optional[tuple] = None) -> list:
    """Execute query and return results as list of dictionaries"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(sql_query, params or ())
        results = cursor.fetchall()
        cursor.close()
        return results
    finally:
        connection.close()

def execute_insert(sql_query: str, params: tuple = ()) -> int:
    """Execute INSERT/UPDATE query and return last insert ID"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(sql_query, params)
        last_insert_id = cursor.lastrowid or 0
        connection.commit()
        cursor.close()
        return int(last_insert_id)
    finally:
        connection.close()

def get_current_database_name() -> str:
    """Get the name of the currently connected database"""
    connection = get_db_connection()
    try:
        return connection.database
    finally:
        connection.close()

def check_table_exists(database: str, table_name: str) -> bool:
    """Check if a table exists in the given database"""
    query = """
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
    """
    results = execute_query(query, (database, table_name))
    return len(results) > 0

def get_table_columns(database: str, table_name: str) -> List[str]:
    """Get list of column names for a table"""
    query = """
    SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s 
    ORDER BY ORDINAL_POSITION
    """
    results = execute_query(query, (database, table_name))
    return [row[0] for row in results]

def column_has_data(table_name: str, column_name: str) -> bool:
    """Check if a column has any non-null values"""
    try:
        query = f"SELECT COUNT(1) FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT 1"
        results = execute_query(query)
        return (results[0][0] if results else 0) > 0
    except Error:
        return False

def find_best_table_and_target() -> Tuple[str, str]:
    """Find the best table and target column combination"""
    current_db = get_current_database_name()
    
    required_columns = {"city", "board", "nights", "stars", "departure_date", "title"}
    
    if TABLE_NAME_OVERRIDE and check_table_exists(current_db, TABLE_NAME_OVERRIDE):
        available_cols = set(get_table_columns(current_db, TABLE_NAME_OVERRIDE))
        if not required_columns.issubset(available_cols):
            raise RuntimeError(f"Table '{TABLE_NAME_OVERRIDE}' missing required columns: {required_columns}")
        
        if TARGET_FIELD_OVERRIDE and TARGET_FIELD_OVERRIDE in available_cols:
            if column_has_data(TABLE_NAME_OVERRIDE, TARGET_FIELD_OVERRIDE):
                return TABLE_NAME_OVERRIDE, TARGET_FIELD_OVERRIDE
        
        for price_col in PRICE_CANDIDATES:
            if price_col in available_cols and column_has_data(TABLE_NAME_OVERRIDE, price_col):
                return TABLE_NAME_OVERRIDE, price_col
    
    common_tables = ["arrangements_clean", "arrangements"]
    for table_name in common_tables:
        if check_table_exists(current_db, table_name):
            available_cols = set(get_table_columns(current_db, table_name))
            if required_columns.issubset(available_cols):
                # Try all price candidates
                price_columns_to_try = []
                if TARGET_FIELD_OVERRIDE:
                    price_columns_to_try.append(TARGET_FIELD_OVERRIDE)
                price_columns_to_try.extend(PRICE_CANDIDATES)
                
                for price_col in price_columns_to_try:
                    if price_col and price_col in available_cols and column_has_data(table_name, price_col):
                        return table_name, price_col
    
    all_tables_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = %s"
    all_tables = execute_query(all_tables_query, (current_db,))
    
    for (table_name,) in all_tables:
        available_cols = set(get_table_columns(current_db, table_name))
        if required_columns.issubset(available_cols):
            for price_col in PRICE_CANDIDATES:
                if price_col in available_cols and column_has_data(table_name, price_col):
                    return table_name, price_col
    
    raise RuntimeError("Could not find a valid (table, target_column) combination in the database")


BOARD_NORMALIZATION_PATTERNS = [
    (r"\broom\s*only\b|\bro\b", "ROOM_ONLY"),
    (r"\bself\s*cater|\bapartment\b|\bapt\b", "SELF_CATERING"), 
    (r"\bbed\s*&?\s*breakfast\b|\bbb\b|no[cć]enje.*doru", "BED_BREAKFAST"),
    (r"\bhalf\s*board\b|\bhb\b|polupansi", "HALF_BOARD"),
    (r"\bfull\s*board\b|\bfb\b", "FULL_BOARD"),
    (r"\bultra\s*all\b|\buai\b", "ULTRA_AI"),
    (r"\b(all\s*in(c(lusive)?)?|ai\s*plus|all\s*inclusive\s*plus)\b", "ALL_INCLUSIVE"),
    (r"\b(premium\s*all|elite\s*all|all\s*in\s*concept)\b", "AI_PREMIUM"),
]

def normalize_board_type(board_value: Optional[str]) -> str:
    """Normalize board type strings to standard categories"""
    if not board_value:
        return "BOARD_OTHER"
    
    normalized = str(board_value).strip().lower()
    if not normalized:
        return "BOARD_OTHER"
    
    # Check each pattern
    for pattern, label in BOARD_NORMALIZATION_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return label
    
    return "BOARD_OTHER"

def normalize_room_type(room_value: Optional[str]) -> str:
    """Normalize room type strings to standard categories"""
    if not room_value:
        return "ROOM_OTHER"
    
    normalized = str(room_value).lower()
    
    if "suite" in normalized:
        return "ROOM_SUITE"
    elif "studio" in normalized:
        return "ROOM_STUDIO"  
    elif "family" in normalized:
        return "ROOM_FAMILY"
    elif "apartment" in normalized or "apt" in normalized:
        return "ROOM_APART"
    elif "standard" in normalized or "std" in normalized:
        return "ROOM_STD"
    elif "deluxe" in normalized or "dlx" in normalized:
        return "ROOM_DELUXE"
    else:
        return "ROOM_OTHER"

def safe_parse_datetime(value):
    """Safely parse datetime from various input formats"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except:
        return None



def load_data_from_database(table_name: str, target_column: str) -> List[Dict[str, Any]]:
    """Load and preprocess data from the database"""
    
    # Build the SELECT query
    column_list = """
        title, city, place, board, room_type, stars, nights, 
        departure_date, created_at, transport_mode, is_air, 
        `{}` AS price
    """.format(target_column)
    
    query = f"""
        SELECT {column_list}
        FROM `{table_name}`
        WHERE `{target_column}` IS NOT NULL
          AND city IS NOT NULL AND city <> ''
          AND board IS NOT NULL AND board <> ''
          AND nights BETWEEN %s AND %s
          AND stars BETWEEN %s AND %s
    """
    
    raw_rows = execute_query_dict(query, (MIN_NIGHTS, MAX_NIGHTS, MIN_STARS, MAX_STARS))
    
    processed_rows = []
    for row in raw_rows:
        try:
            # Basic type conversions and validation
            processed_row = {}
            processed_row["price"] = float(row["price"])
            processed_row["nights"] = int(row["nights"])  
            processed_row["stars"] = int(row["stars"])
            
            # String field processing
            processed_row["title"] = str(row.get("title") or "").strip() or "OTHER_HOTEL"
            processed_row["city"] = str(row.get("city") or "").strip()
            processed_row["place"] = str(row.get("place") or "").strip() or "OTHER_PLACE"
            
            # Normalize categorical fields
            processed_row["board_cat"] = normalize_board_type(row.get("board"))
            processed_row["room_group"] = normalize_room_type(row.get("room_type"))
            
            # Transport processing
            transport_raw = str(row.get("transport_mode") or "").strip().lower()
            processed_row["transport_mode"] = transport_raw if transport_raw else "other"
            processed_row["is_air"] = int(row.get("is_air")) if row.get("is_air") is not None else 1
            
            # Date processing
            departure_date = safe_parse_datetime(row.get("departure_date"))
            processed_row["month"] = int(departure_date.month) if departure_date else 0
            
            created_date = safe_parse_datetime(row.get("created_at"))
            processed_row["created_at"] = created_date
            
            # Lead time calculation
            processed_row["lead_time"] = 0
            if departure_date and created_date:
                days_diff = (departure_date - created_date).days
                processed_row["lead_time"] = max(0, min(LEAD_TIME_MAX, days_diff))
            
            # Stars feature engineering
            processed_row["stars_unknown"] = 1 if (processed_row["stars"] == 0) else 0
            
            processed_rows.append(processed_row)
            
        except Exception as e:
            # Skip rows that can't be processed - maybe log this in production
            continue
    
    # Add price class targets (EUR-based thresholds)
    def price_to_class(price_eur):
        if price_eur <= 500:
            return 0
        elif price_eur <= 1500: 
            return 1
        elif price_eur <= 3000:
            return 2
        else:
            return 3
    
    for row in processed_rows:
        row["target_class"] = price_to_class(row["price"])
    
    return processed_rows


class ManualOneHotEncoder:
    """Simple one-hot encoder implementation"""
    
    def __init__(self):
        self.vocabularies: Dict[str, List[str]] = {}
    
    def fit(self, data_rows: List[Dict[str, Any]], column_names: List[str]):
        """Learn vocabularies from training data"""
        for col in column_names:
            unique_values = sorted(set(str(row.get(col, "")).strip() for row in data_rows))
            self.vocabularies[col] = list(unique_values)
    
    def transform(self, data_rows: List[Dict[str, Any]], column_names: List[str]) -> np.ndarray:
        """Transform data to one-hot encoded features"""
        feature_matrices = []
        
        for col in column_names:
            vocab = self.vocabularies.get(col, [])
            value_to_index = {val: idx for idx, val in enumerate(vocab)}
            
            encoded_matrix = np.zeros((len(data_rows), len(vocab)), dtype=float)
            
            for row_idx, row in enumerate(data_rows):
                key = str(row.get(col, "")).strip()
                if key in value_to_index:
                    encoded_matrix[row_idx, value_to_index[key]] = 1.0
            
            feature_matrices.append(encoded_matrix)
        
        if feature_matrices:
            return np.concatenate(feature_matrices, axis=1)
        else:
            return np.zeros((len(data_rows), 0), dtype=float)

class ManualStandardScaler:
    """Simple standard scaler implementation"""
    
    def __init__(self):
        self.feature_means = {}
        self.feature_stds = {}
    
    def fit(self, data_rows: List[Dict[str, Any]], column_names: List[str]):
        """Learn scaling parameters from training data"""
        for col in column_names:
            values = np.array([float(row.get(col, 0.0)) for row in data_rows], dtype=float)
            
            mean_val = float(values.mean()) if values.size > 0 else 0.0
            std_val = float(values.std()) if values.size > 0 else 1.0
            
            if std_val <= 1e-12:
                std_val = 1.0
            
            self.feature_means[col] = mean_val
            self.feature_stds[col] = std_val
    
    def transform(self, data_rows: List[Dict[str, Any]], column_names: List[str]) -> np.ndarray:
        """Apply standard scaling to data"""
        scaled_matrix = np.zeros((len(data_rows), len(column_names)), dtype=float)
        
        for col_idx, col in enumerate(column_names):
            mean_val = self.feature_means.get(col, 0.0)
            std_val = self.feature_stds.get(col, 1.0)
            
            if std_val <= 1e-12:
                std_val = 1.0
            
            for row_idx, row in enumerate(data_rows):
                raw_value = float(row.get(col, 0.0))
                scaled_value = (raw_value - mean_val) / std_val
                scaled_matrix[row_idx, col_idx] = scaled_value
        
        return scaled_matrix



class KNearestNeighbors:
    """K-Nearest Neighbors classifier with optional distance weighting"""
    
    def __init__(self, n_neighbors=11, use_weighting=True):
        self.n_neighbors = n_neighbors
        self.use_weighting = use_weighting
        self.training_features = None
        self.training_labels = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Store training data for prediction"""
        self.training_features = np.asarray(X_train, dtype=float)
        self.training_labels = np.asarray(y_train, dtype=int)
    
    def predict_single(self, sample_features: np.ndarray) -> int:
        """Predict class for a single sample"""
        # Calculate distances to all training samples
        distances = np.linalg.norm(self.training_features - sample_features, axis=1)
        
        # Find k nearest neighbors
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        neighbor_labels = self.training_labels[nearest_indices]
        neighbor_distances = distances[nearest_indices]
        
        if not self.use_weighting:
            # Simple majority vote
            vote_counts = Counter(neighbor_labels)
            return int(vote_counts.most_common(1)[0][0])
        else:
            # Distance-weighted voting
            epsilon = 1e-9  
            weights = 1.0 / (np.square(neighbor_distances) + epsilon)
            
            weighted_votes = defaultdict(float)
            for label, weight in zip(neighbor_labels, weights):
                weighted_votes[int(label)] += float(weight)
            
            # Return class with highest weighted vote
            best_class = max(weighted_votes.items(), key=lambda item: item[1])[0]
            return best_class
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict classes for multiple samples"""
        predictions = []
        for sample in X_test:
            pred = self.predict_single(sample)
            predictions.append(pred)
        return np.array(predictions, dtype=int)



def calculate_accuracy(y_true, y_predicted):
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted) 
    
    if y_true.size == 0:
        return 0.0
    
    correct_predictions = (y_true == y_predicted).sum()
    total_predictions = y_true.size
    return float(correct_predictions / total_predictions)

def calculate_macro_f1(y_true, y_predicted, num_classes=4):
    """Calculate macro-averaged F1 score"""
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    
    class_f1_scores = []
    
    for class_label in range(num_classes):
        # Calculate precision, recall, F1 for this class
        true_positives = int(np.sum((y_true == class_label) & (y_predicted == class_label)))
        false_positives = int(np.sum((y_true != class_label) & (y_predicted == class_label)))
        false_negatives = int(np.sum((y_true == class_label) & (y_predicted != class_label)))
        
        # Calculate precision and recall
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
            
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        # Calculate F1 score
        if (precision + recall) > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        class_f1_scores.append(f1_score)
    
    # Return macro average
    return float(np.mean(class_f1_scores))

def create_confusion_matrix(y_true, y_predicted, num_classes=4):
    """Create confusion matrix"""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    
    for true_label, pred_label in zip(y_true, y_predicted):
        matrix[int(true_label)][int(pred_label)] += 1
    
    return matrix


class FeaturePipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self):
        # Define which columns to use for each encoder
        self.categorical_columns = [
            "city", "place", "transport_mode", "board_cat", "room_group", "hotel_title"
        ]
        self.numerical_columns = [
            "nights", "stars", "stars_unknown", "is_air", "lead_time", "is_peak",
            "month_sin", "month_cos", "nights_squared", "stars_squared", "nights_times_stars"
        ]
        
        self.categorical_encoder = ManualOneHotEncoder()
        self.numerical_scaler = ManualStandardScaler()
    
    def preprocess_rare_categories(self, data_rows: List[Dict[str, Any]]):
        
        # Extract values for each categorical column
        city_values = [row["city"] for row in data_rows]
        place_values = [row["place"] for row in data_rows]  
        room_values = [row["room_group"] for row in data_rows]
        transport_values = [row["transport_mode"] for row in data_rows]
        hotel_values = [row["title"] for row in data_rows]
        
        # Apply rare value bucketing
        bucketed_cities = apply_rare_bucketing(city_values, CITY_MIN_FREQ, "OTHER_CITY")
        bucketed_places = apply_rare_bucketing(place_values, CITY_MIN_FREQ, "OTHER_PLACE")
        bucketed_rooms = apply_rare_bucketing(room_values, ROOM_MIN_FREQ, "ROOM_OTHER")
        bucketed_transport = apply_rare_bucketing(transport_values, TRANSP_MIN_FREQ, "other")
        bucketed_hotels = apply_rare_bucketing(hotel_values, HOTEL_MIN_FREQ, "OTHER_HOTEL")
        
        # Update the data
        for idx, row in enumerate(data_rows):
            row["city"] = bucketed_cities[idx]
            row["place"] = bucketed_places[idx]
            row["room_group"] = bucketed_rooms[idx] 
            row["transport_mode"] = bucketed_transport[idx]
            row["hotel_title"] = bucketed_hotels[idx]
    
    def add_derived_features(self, data_rows: List[Dict[str, Any]]):
        """Add engineered features to the data"""
        for row in data_rows:
            # Cyclical encoding for month
            month_val = int(row["month"]) % 12
            angle = 2 * math.pi * (month_val / 12.0)
            row["month_sin"] = math.sin(angle)
            row["month_cos"] = math.cos(angle)
            
            # Peak season indicator (July/August)
            row["is_peak"] = 1 if int(row["month"]) in (7, 8) else 0
            
            # Polynomial features
            row["nights_squared"] = row["nights"] ** 2
            row["stars_squared"] = row["stars"] ** 2
            row["nights_times_stars"] = row["nights"] * row["stars"]
    
    def create_feature_matrix(self, data_rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert processed data to feature matrix and target vector"""
        
        # Encode categorical features
        categorical_features = self.categorical_encoder.transform(data_rows, self.categorical_columns)
        
        # Scale numerical features  
        numerical_features = self.numerical_scaler.transform(data_rows, self.numerical_columns)
        
        # Combine feature matrices
        if categorical_features.size > 0:
            combined_features = np.hstack([categorical_features, numerical_features])
        else:
            combined_features = numerical_features
        
        # Extract target values
        target_values = np.array([int(row["target_class"]) for row in data_rows], dtype=int)
        
        return combined_features, target_values

def apply_rare_bucketing(values_list: List[str], min_frequency: int, replacement_token: str) -> List[str]:
    """Replace rare values with a common token"""
    value_counts = Counter(values_list)
    
    bucketed_values = []
    for value in values_list:
        if value_counts[value] >= min_frequency:
            bucketed_values.append(value)
        else:
            bucketed_values.append(replacement_token)
    
    return bucketed_values

def split_train_test(data_rows: List[Dict[str, Any]], test_size: float = 0.2, random_seed: int = 42):
    """Split data into training and test sets"""
    random_generator = random.Random(random_seed)
    
    # Create shuffled indices
    indices = list(range(len(data_rows)))
    random_generator.shuffle(indices)
    
    # Calculate split point
    split_point = int(len(indices) * (1 - test_size))
    
    # Split indices
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # Split data
    train_data = [data_rows[i] for i in train_indices]
    test_data = [data_rows[i] for i in test_indices]
    
    return train_data, test_data



HISTORY_TABLE_NAME = "prediction_history_knn"

def create_history_table_if_needed():
    """Create the prediction history table if it doesn't exist"""
    current_db = get_current_database_name()
    if check_table_exists(current_db, HISTORY_TABLE_NAME):
        return  # Table already exists
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS `{HISTORY_TABLE_NAME}` (
      `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
      `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `ip` VARCHAR(64) NULL,
      `user_agent` VARCHAR(512) NULL,
      `city` VARCHAR(128) NOT NULL,
      `hotel` VARCHAR(256) NULL,
      `nights` INT NOT NULL,
      `stars` INT NOT NULL,
      `board` VARCHAR(64) NOT NULL,
      `transport` VARCHAR(64) NOT NULL,
      `room` VARCHAR(64) NOT NULL,
      `month` INT NOT NULL,
      `pred_class` INT NOT NULL,
      `pred_label` VARCHAR(32) NOT NULL,
      `k` INT NOT NULL,
      `weighted` TINYINT(1) NOT NULL,
      `acc` DOUBLE NULL,
      `f1m` DOUBLE NULL,
      INDEX (`created_at`),
      INDEX (`city`),
      INDEX (`month`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    execute_insert(create_table_sql, ())

def get_user_ip_address() -> str:
    """Extract user IP address from request headers"""
    # Check for forwarded IP first (behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Fall back to direct remote address
    return request.remote_addr or ""

def save_prediction_to_history(form_data: Dict[str, Any], predicted_class: int, 
                              class_label: str, model_metrics: Dict[str, Any]) -> int:
    """Save a prediction to the history table"""
    
    # Extract request metadata
    user_agent = (request.headers.get("User-Agent", "") or "")[:512]
    client_ip = get_user_ip_address()[:64]
    
    insert_sql = f"""
    INSERT INTO `{HISTORY_TABLE_NAME}`
    (ip, user_agent, city, hotel, nights, stars, board, transport, room, month,
     pred_class, pred_label, k, weighted, acc, f1m)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    insert_params = (
        client_ip,
        user_agent,
        str(form_data.get("city") or "")[:128],
        str(form_data.get("hotel") or "")[:256],
        int(form_data.get("nights") or 0),
        int(form_data.get("stars") or 0),
        str(form_data.get("board") or "")[:64],
        str(form_data.get("transport") or "")[:64],
        str(form_data.get("room") or "")[:64],
        int(form_data.get("month") or 0),
        int(predicted_class),
        str(class_label)[:32],
        int(K_VALUE),
        1 if USE_DISTANCE_WEIGHTING else 0,
        float(model_metrics.get("acc") or 0.0),
        float(model_metrics.get("f1m") or 0.0)
    )
    
    return execute_insert(insert_sql, insert_params)

def build_history_filter_clause(request_args) -> Tuple[str, list]:
    """Build WHERE clause and parameters for history filtering"""
    where_conditions = []
    query_parameters = []
    
    # City filter
    if request_args.get("city"):
        where_conditions.append("city = %s")
        query_parameters.append(request_args.get("city"))
    
    # Month filter
    if request_args.get("month") and str(request_args.get("month")).isdigit():
        month_val = int(request_args.get("month"))
        if 1 <= month_val <= 12:
            where_conditions.append("month = %s")
            query_parameters.append(month_val)
    
    where_clause = ("WHERE " + " AND ".join(where_conditions)) if where_conditions else ""
    return where_clause, query_parameters

def query_prediction_history(request_args, page_number: int, records_per_page: int):
    """Query prediction history with filtering and pagination"""
    
    # Build filtering conditions
    where_clause, filter_params = build_history_filter_clause(request_args)
    
    # Count total records
    count_query = f"SELECT COUNT(*) FROM `{HISTORY_TABLE_NAME}` {where_clause}"
    total_records = execute_query(count_query, tuple(filter_params))[0][0]
    
    # Calculate pagination
    total_pages = max(1, (total_records + records_per_page - 1) // records_per_page)
    current_page = min(max(1, page_number), total_pages)
    offset = (current_page - 1) * records_per_page
    
    # Query records for current page
    data_query = f"""
        SELECT id, created_at, ip, user_agent, city, hotel, nights, stars, board, transport, room, month,
               pred_class, pred_label, k, weighted
        FROM `{HISTORY_TABLE_NAME}` {where_clause}
        ORDER BY id DESC
        LIMIT %s OFFSET %s
    """
    query_params = filter_params + [records_per_page, offset]
    records = execute_query_dict(data_query, tuple(query_params))
    
    return total_records, total_pages, current_page, records



app = Flask(__name__)

# Global variables to store trained model and preprocessing objects
TRAINED_MODEL: Optional[KNearestNeighbors] = None
CATEGORICAL_ENCODER: Optional[ManualOneHotEncoder] = None  
NUMERICAL_SCALER: Optional[ManualStandardScaler] = None
FEATURE_VOCABULARIES: Dict[str, List[str]] = {}
MODEL_METADATA = {"table": "?", "target": "?", "k": K_VALUE, "weighted": USE_DISTANCE_WEIGHTING}
PERFORMANCE_METRICS = {"acc": 0.0, "f1m": 0.0}
CONFUSION_MATRIX = []
CLASS_SUPPORT = {}
DATA_COUNTS = {"train": 0, "test": 0, "features": 0}



@app.route("/", methods=["GET"])
def home_page():
    """Main prediction interface"""
    return render_template_string(
        HTML_TPL,
        cities=FEATURE_VOCABULARIES.get("city", []),
        boards=FEATURE_VOCABULARIES.get("board_cat", []) or [
            "BED_BREAKFAST", "HALF_BOARD", "FULL_BOARD", "ALL_INCLUSIVE", 
            "ULTRA_AI", "AI_PREMIUM", "SELF_CATERING", "ROOM_ONLY", "BOARD_OTHER"
        ],
        transports=FEATURE_VOCABULARIES.get("transport_mode", []) or [],
        rooms=FEATURE_VOCABULARIES.get("room_group", []) or [],
        hotels=[hotel for hotel in FEATURE_VOCABULARIES.get("hotel_title", []) 
                if hotel != "OTHER_HOTEL"][:150],  # Limit hotel options for UI
        metrics=PERFORMANCE_METRICS,
        counts=DATA_COUNTS,
        support=CLASS_SUPPORT,
        cm=CONFUSION_MATRIX,
        prediction=None,
        label=None,
        meta=MODEL_METADATA
    )

@app.route("/predict", methods=["POST"])
def make_prediction():
    """Handle prediction requests"""
    
    # Extract form data
    form_data = {
        "city": request.form.get("city"),
        "hotel": request.form.get("hotel"),
        "nights": int(request.form.get("nights")),
        "stars": int(request.form.get("stars")),
        "board": request.form.get("board"),
        "transport": request.form.get("transport"),
        "room": request.form.get("room"),
        "month": int(request.form.get("month"))
    }
    
    # Create a data row for prediction (similar to training data format)
    prediction_row = {
        "title": str(form_data.get("hotel") or "ANY_HOTEL"),
        "hotel_title": str(form_data.get("hotel") or "ANY_HOTEL"),
        "city": str(form_data["city"]).strip(),
        "place": "OTHER_PLACE",  # Default since not in form
        "board_cat": str(form_data["board"]).strip(),
        "room_group": str(form_data["room"]).strip(),
        "transport_mode": str(form_data["transport"]).strip().lower(),
        "is_air": 1 if str(form_data["transport"]).strip().lower() == "air" else 0,
        "nights": int(form_data["nights"]),
        "stars": int(form_data["stars"]),
        "stars_unknown": 1 if int(form_data["stars"]) == 0 else 0,
        "month": int(form_data["month"]),
        "lead_time": 0  # Default for new predictions
    }
    
    # Handle unseen categorical values by mapping to 'OTHER' tokens
    category_mappings = [
        ("city", "OTHER_CITY"),
        ("place", "OTHER_PLACE"), 
        ("room_group", "ROOM_OTHER"),
        ("transport_mode", "other"),
        ("hotel_title", "OTHER_HOTEL")
    ]
    
    for column_name, fallback_value in category_mappings:
        if prediction_row[column_name] not in FEATURE_VOCABULARIES.get(column_name, []):
            prediction_row[column_name] = fallback_value
    
    # Add derived features (same as training)
    month_angle = 2 * math.pi * ((prediction_row["month"] % 12) / 12.0)
    prediction_row["month_sin"] = math.sin(month_angle)
    prediction_row["month_cos"] = math.cos(month_angle)
    prediction_row["is_peak"] = 1 if prediction_row["month"] in (7, 8) else 0
    prediction_row["nights_squared"] = prediction_row["nights"] ** 2
    prediction_row["stars_squared"] = prediction_row["stars"] ** 2
    prediction_row["nights_times_stars"] = prediction_row["nights"] * prediction_row["stars"]
    
    # Convert to feature vector
    categorical_cols = ["city", "place", "transport_mode", "board_cat", "room_group", "hotel_title"]
    numerical_cols = ["nights", "stars", "stars_unknown", "is_air", "lead_time", "is_peak", 
                     "month_sin", "month_cos", "nights_squared", "stars_squared", "nights_times_stars"]
    
    categorical_features = CATEGORICAL_ENCODER.transform([prediction_row], categorical_cols)
    numerical_features = NUMERICAL_SCALER.transform([prediction_row], numerical_cols)
    
    if categorical_features.size > 0:
        feature_vector = np.hstack([categorical_features, numerical_features])
    else:
        feature_vector = numerical_features
    
    # Make prediction
    predicted_class = int(TRAINED_MODEL.predict(feature_vector)[0])
    
    # Map class to label
    class_labels = ["≤500", "501-1500", "1501-3000", "≥3000"]
    predicted_label = class_labels[predicted_class]
    
    # Save to history database
    try:
        save_prediction_to_history(form_data, predicted_class, predicted_label, PERFORMANCE_METRICS)
    except Exception as e:
        print(f"[WARNING] Failed to save prediction to history: {e}")
    
    # Return result page
    return render_template_string(
        HTML_TPL,
        cities=FEATURE_VOCABULARIES.get("city", []),
        boards=FEATURE_VOCABULARIES.get("board_cat", []),
        transports=FEATURE_VOCABULARIES.get("transport_mode", []),
        rooms=FEATURE_VOCABULARIES.get("room_group", []),
        hotels=[hotel for hotel in FEATURE_VOCABULARIES.get("hotel_title", []) 
                if hotel != "OTHER_HOTEL"][:150],
        metrics=PERFORMANCE_METRICS,
        counts=DATA_COUNTS,
        support=CLASS_SUPPORT,
        cm=CONFUSION_MATRIX,
        prediction=predicted_class,
        label=predicted_label,
        meta=MODEL_METADATA
    )

@app.route("/history", methods=["GET"])
def history_page():
    """Display prediction history with filtering and pagination"""
    
    # Parse pagination parameters
    try:
        per_page = min(max(1, int(request.args.get("per") or HIST_DEFAULT_PER)), HIST_MAX_PER)
    except:
        per_page = HIST_DEFAULT_PER
    
    try:
        page_num = max(1, int(request.args.get("page") or "1"))
    except:
        page_num = 1
    
    # Query history data
    total, pages, current_page, raw_records = query_prediction_history(request.args, page_num, per_page)
    
    # Format records for display
    display_items = []
    for record in raw_records:
        user_agent_full = record.get("user_agent") or ""
        display_items.append({
            "id": record["id"],
            "created_at": record["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "ip": record.get("ip") or "",
            "user_agent_short": (user_agent_full[:60] + "…") if len(user_agent_full) > 60 else user_agent_full,
            "city": record.get("city") or "",
            "hotel": record.get("hotel") or "",
            "nights": record.get("nights") or 0,
            "stars": record.get("stars") or 0,
            "board": record.get("board") or "",
            "transport": record.get("transport") or "",
            "room": record.get("room") or "",
            "month": record.get("month") or 0,
            "pred_class": record.get("pred_class"),
            "pred_label": record.get("pred_label") or "",
            "k": record.get("k") or 0,
            "weighted": bool(record.get("weighted")),
        })
    
    # Get available cities for filter dropdown
    available_cities = FEATURE_VOCABULARIES.get("city", [])
    
    def build_pager_url(target_page):
        """Helper function to build pagination URLs"""
        url_args = request.args.to_dict(flat=True)
        url_args["page"] = str(target_page)
        url_args["per"] = str(per_page)
        query_string = "&".join([f"{key}={val}" for key, val in url_args.items() if val != ""])
        return f"/history?{query_string}" if query_string else "/history"
    
    return render_template_string(
        HTML_HIST,
        items=display_items,
        total=total,
        pages=pages,
        page=current_page,
        per=per_page,
        cities=available_cities,
        flt={
            "city": request.args.get("city") or "",
            "month": int(request.args.get("month")) if (request.args.get("month") and 
                     request.args.get("month").isdigit()) else None
        },
        pager_url=build_pager_url
    )

@app.route("/history.csv", methods=["GET"])
def export_history_csv():
    """Export prediction history as CSV file"""
    
    # Limit export size to prevent huge downloads
    try:
        max_records = min(int(request.args.get("limit") or 10000), 10000)
    except:
        max_records = 10000
    
    # Query data (using page 1 with high limit)
    total, pages, current_page, records = query_prediction_history(request.args, 1, max_records)
    
    # Generate CSV content
    csv_output = io.StringIO()
    csv_writer = csv.writer(csv_output)
    
    # Write header row
    csv_writer.writerow([
        "id", "created_at", "ip", "user_agent", "city", "hotel", "nights", "stars", 
        "board", "transport", "room", "month", "pred_class", "pred_label", 
        "k", "weighted", "acc", "f1m"
    ])
    
    # Write data rows
    for record in records:
        csv_writer.writerow([
            record["id"],
            record["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            record.get("ip") or "",
            record.get("user_agent") or "",
            record.get("city") or "",
            record.get("hotel") or "",
            record.get("nights") or 0,
            record.get("stars") or 0,
            record.get("board") or "",
            record.get("transport") or "",
            record.get("room") or "",
            record.get("month") or 0,
            record.get("pred_class"),
            record.get("pred_label") or "",
            record.get("k") or 0,
            int(record.get("weighted") or 0),
            PERFORMANCE_METRICS.get("acc", 0.0),
            PERFORMANCE_METRICS.get("f1m", 0.0)
        ])
    
    csv_content = csv_output.getvalue()
    csv_output.close()
    
    return Response(
        csv_content,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=prediction_history_knn.csv"}
    )

@app.route("/api/history", methods=["GET"])
def history_api():
    """JSON API for prediction history"""
    
    # Parse parameters
    try:
        per_page = min(max(1, int(request.args.get("limit") or HIST_DEFAULT_PER)), HIST_MAX_PER)
    except:
        per_page = HIST_DEFAULT_PER
    
    try:
        page_num = max(1, int(request.args.get("page") or "1"))
    except:
        page_num = 1
    
    # Query data
    total, pages, current_page, records = query_prediction_history(request.args, page_num, per_page)
    
    # Format response
    response_data = {
        "page": current_page,
        "pages": pages,
        "per_page": per_page,
        "total": total,
        "items": []
    }
    
    for record in records:
        response_data["items"].append({
            "id": record["id"],
            "created_at": record["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "ip": record.get("ip"),
            "user_agent": record.get("user_agent"),
            "city": record.get("city"),
            "hotel": record.get("hotel"),
            "nights": record.get("nights"),
            "stars": record.get("stars"),
            "board": record.get("board"),
            "transport": record.get("transport"),
            "room": record.get("room"),
            "month": record.get("month"),
            "pred_class": record.get("pred_class"),
            "pred_label": record.get("pred_label"),
            "k": record.get("k"),
            "weighted": int(record.get("weighted") or 0),
            "accuracy": PERFORMANCE_METRICS.get("acc", 0.0),
            "macro_f1": PERFORMANCE_METRICS.get("f1m", 0.0)
        })
    
    return jsonify(response_data)


def main():
    """Initialize and run the Flask application"""
    
    # Test database connection
    try:
        test_connection = get_db_connection()
        test_connection.close()
        print("[INFO] Database connection successful")
    except Error as e:
        print(f"[ERROR] Database connection failed: {e}")
        return
    
    # Initialize history table
    try:
        create_history_table_if_needed()
        print("[INFO] History table ready")
    except Exception as e:
        print(f"[WARNING] Could not initialize history table: {e}")
    
    # Find and load data
    try:
        table_name, target_column = find_best_table_and_target()
        print(f"[INFO] Using table: {table_name}, target: {target_column}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Load and preprocess data
    print("[INFO] Loading data from database...")
    data_rows = load_data_from_database(table_name, target_column)
    
    if not data_rows:
        print("[ERROR] No valid data rows found after loading")
        return
    
    print(f"[INFO] Loaded {len(data_rows)} rows")
    
    # Initialize feature pipeline
    set_random_seed(RANDOM_SEED)
    feature_pipeline = FeaturePipeline()
    
    # Preprocess categorical features
    feature_pipeline.preprocess_rare_categories(data_rows)
    feature_pipeline.add_derived_features(data_rows)
    
    # Split into train/test sets
    train_data, test_data = split_train_test(data_rows, test_size=TEST_RATIO, random_seed=RANDOM_SEED)
    print(f"[INFO] Split: {len(train_data)} train, {len(test_data)} test")
    
    # Fit preprocessing components on training data
    feature_pipeline.categorical_encoder.fit(train_data, feature_pipeline.categorical_columns)
    feature_pipeline.numerical_scaler.fit(train_data, feature_pipeline.numerical_columns)
    
    # Create feature matrices
    X_train, y_train = feature_pipeline.create_feature_matrix(train_data)
    X_test, y_test = feature_pipeline.create_feature_matrix(test_data)
    
    print(f"[INFO] Feature matrix shape: {X_train.shape}")
    
    # Train KNN model
    print("[INFO] Training KNN classifier...")
    knn_classifier = KNearestNeighbors(n_neighbors=K_VALUE, use_weighting=USE_DISTANCE_WEIGHTING)
    knn_classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    test_predictions = knn_classifier.predict(X_test)
    
    test_accuracy = calculate_accuracy(y_test, test_predictions)
    test_macro_f1 = calculate_macro_f1(y_test, test_predictions, num_classes=4)
    confusion_mat = create_confusion_matrix(y_test, test_predictions, num_classes=4)
    class_support = dict(Counter(y_test))
    
    print(f"[EVALUATION] KNN k={K_VALUE} weighted={USE_DISTANCE_WEIGHTING}")
    print(f"[EVALUATION] Accuracy: {test_accuracy:.4f}")
    print(f"[EVALUATION] Macro F1: {test_macro_f1:.4f}")
    print(f"[EVALUATION] Features: {X_train.shape[1]}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Store everything in global variables for Flask routes
    global TRAINED_MODEL, CATEGORICAL_ENCODER, NUMERICAL_SCALER, FEATURE_VOCABULARIES
    global MODEL_METADATA, PERFORMANCE_METRICS, CONFUSION_MATRIX, CLASS_SUPPORT, DATA_COUNTS
    
    TRAINED_MODEL = knn_classifier
    CATEGORICAL_ENCODER = feature_pipeline.categorical_encoder
    NUMERICAL_SCALER = feature_pipeline.numerical_scaler
    
    FEATURE_VOCABULARIES = {
        col: feature_pipeline.categorical_encoder.vocabularies.get(col, []) 
        for col in feature_pipeline.categorical_columns
    }
    
    MODEL_METADATA = {
        "table": table_name,
        "target": target_column,
        "k": K_VALUE,
        "weighted": USE_DISTANCE_WEIGHTING
    }
    
    PERFORMANCE_METRICS = {
        "acc": test_accuracy,
        "f1m": test_macro_f1
    }
    
    CONFUSION_MATRIX = confusion_mat
    CLASS_SUPPORT = class_support
    DATA_COUNTS = {
        "train": X_train.shape[0],
        "test": X_test.shape[0],
        "features": X_train.shape[1]
    }
    
    # Start Flask server
    port_number = int(os.environ.get("PORT", 5002))
    print(f"[INFO] Starting Flask server on port {port_number}")
    
    app.run(host="0.0.0.0", port=port_number, debug=False)

if __name__ == "__main__":
    main()