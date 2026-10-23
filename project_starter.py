import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import re
import threading
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from sqlalchemy import create_engine, Engine
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)
from dotenv import load_dotenv

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise
@tool
def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

@tool
def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

@tool
def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

@tool
def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


@tool  
def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


@tool
def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

@tool
def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
load_dotenv()

# The Vocareum proxy by default; set OPENAI_API_BASE (e.g. https://api.openai.com/v1)
# in .env to call OpenAI directly with your own key.
API_BASE = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base=API_BASE,
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
)

# The orchestrator makes the workflow decisions, so it uses a stronger model. It delegates
# to one agent per step: parallel calls would run the same agent concurrently and send
# work to sales before the quote is ready.
orchestrator_model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base=API_BASE,
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    parallel_tool_calls=False,
)


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# Catalog lookup by exact item name
CATALOG = {item["item_name"]: item for item in paper_supplies}

# Words that appear in most requests but do not identify a product
GENERIC_WORDS = {
    "a", "an", "the", "of", "and", "for", "in", "with", "per",
    "paper", "papers", "sheet", "sheets", "ream", "reams", "pack", "packs",
    "packet", "packets", "size", "sized", "inch", "inches", "high", "quality",
    "various", "assorted", "colors",
}

# Customer words for plain paper, mapped to the word the catalog uses
KEYWORD_SYNONYMS = {"printer": "copy", "printing": "copy"}

# Finishes and materials that make a different product, e.g. "A4 glossy paper" is not "A4 paper"
TYPE_WORDS = {
    "glossy", "matte", "colored", "recycled", "eco", "kraft", "construction", "photo",
    "cardstock", "glitter", "crepe", "butcher", "uncoated", "patterned", "letterhead", "wrapping",
}

# Dimensions that correspond to named paper sizes
SIZE_ALIASES = {"8.5x11": "letter", "8.5x14": "legal"}

# Inch marks include quotes and primes, since agents rewrite 8.5"x11" as 8.5'x11'
DIMENSION_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:"|”|″|\'|’|′|in\b|inch(?:es)?)?\s*[x×]\s*(\d+(?:\.\d+)?)')

# Units that hold several sheets or items. Only the ream has an established size;
# for the others the customer has to confirm how many the pack holds.
PACKAGING_UNITS = {"ream", "pack", "package", "packet", "box", "case", "carton", "bundle"}
PACK_SIZES = {"ream": 500}  # sheets per ream


def _extract_sizes(text: str) -> set:
    """Return the paper sizes mentioned in text, e.g. {'a4'}, {'letter'} or {'24x36'}."""
    text = text.lower()
    sizes = set(re.findall(r"\b(a[0-9])\b", text))
    sizes |= set(re.findall(r"\b(letter|legal)\b", text))
    for width, height in DIMENSION_PATTERN.findall(text):
        dimension = f"{width}x{height}"
        sizes.add(SIZE_ALIASES.get(dimension, dimension))
    return sizes


def _extract_keywords(text: str) -> List[str]:
    """Return the product-describing words in text, without sizes, numbers, or generic words."""
    text = DIMENSION_PATTERN.sub(" ", text.lower().replace("card stock", "cardstock"))
    sizes = _extract_sizes(text)
    words = [KEYWORD_SYNONYMS.get(word, word) for word in re.findall(r"[a-z0-9]+", text)]
    return [
        word for word in dict.fromkeys(words)
        if word not in GENERIC_WORDS and word not in sizes and not word.isdigit()
    ]


def _word_match_score(word: str, item_words: List[str]) -> int:
    """Score 2 for an exact word match, 1 for a shared 5-letter stem (colorful/colored), else 0."""
    if word in item_words:
        return 2
    if any(min(len(word), len(item_word)) >= 5 and word[:5] == item_word[:5] for item_word in item_words):
        return 1
    return 0


# Case-insensitive lookup so "Colored Paper" resolves to "Colored paper"
CATALOG_NAMES = {name.lower(): name for name in CATALOG}


def _catalog_name(item_name: str) -> Optional[str]:
    """
    Return the exact catalog name for item_name, ignoring case, or the catalog item that
    fits it (e.g. "Heavy Cardstock (White)" -> "Cardstock"). Return None if not carried.
    """
    exact_name = CATALOG_NAMES.get(" ".join(str(item_name).lower().split()))
    if exact_name:
        return exact_name
    candidates = _match_candidates(str(item_name))
    return candidates[0]["item_name"] if candidates and candidates[0]["fits"] else None


def _pricing_unit(item_name: str) -> str:
    """Return 'sheet' for papers priced per sheet and 'unit' for products priced per item."""
    return "sheet" if CATALOG[item_name]["category"] in {"paper", "specialty"} else "unit"


def _packaging_unit(text: str) -> Optional[str]:
    """Return the packaging unit named in text, e.g. 'reams' -> 'ream', or None."""
    for word in re.findall(r"[a-z]+", str(text).lower()):
        for singular in (word, word[:-1], word[:-2]):
            if singular in PACKAGING_UNITS:
                return singular
    return None


def _catalog_quantity(item_name: str, quantity: int, unit: str, description: str) -> Dict:
    """
    Convert a requested quantity into the unit the catalog prices item_name in.

    Packaging units are converted with an established pack size: 500 reams of a paper
    priced per sheet is 250,000 sheets. Other units (sheets, rolls, units, ...) count
    one to one. Returns the catalog quantity and the conversion, or the reason the
    customer has to confirm the pack size.
    """
    package = _packaging_unit(unit) or _packaging_unit(description)
    if package is None:
        return {"quantity": quantity}
    pricing_unit = _pricing_unit(item_name)
    if package not in PACK_SIZES or pricing_unit != "sheet":
        return {
            "reason": f"A {package} of {item_name} has no standard size. Please confirm how many "
                      f"{pricing_unit}s each {package} should hold.",
        }
    pack_size = PACK_SIZES[package]
    catalog_quantity = quantity * pack_size
    return {
        "quantity": catalog_quantity,
        "conversion": f"{quantity:,} {package}{'s' if quantity != 1 else ''} = {catalog_quantity:,} sheets "
                      f"({pack_size} sheets per {package})",
    }


def _unknown_item_error(item_name: str) -> Dict:
    """Error returned when a tool receives a name that no catalog item fits."""
    return {
        "status": "error",
        "reason": f"'{item_name}' is not a catalog item. "
                  f"{_not_carried_reason(item_name, _match_candidates(item_name))}",
    }


def _current_stock(item_name: str, as_of_date: str) -> int:
    """
    Return the units of a catalog item available as of a date.

    Transactions dated after as_of_date (e.g. a sale recorded on a later date) count
    too when they lower the stock, so selling now can never make a later stock negative.
    """
    stock_on_date = int(get_stock_level(item_name, as_of_date)["current_stock"].iloc[0])
    stock_overall = int(get_stock_level(item_name, "9999-12-31")["current_stock"].iloc[0])
    return min(stock_on_date, stock_overall)


# The customer request being handled, filled in by OrchestrationAgent.handle_request.
# Tools take the request date, the inventory check, and the recorded sale from here
# instead of trusting dates and quantities that agents copy to each other in text.
REQUEST_CONTEXT = {}
# Agents can call tools in parallel; tools that change REQUEST_CONTEXT hold this lock
REQUEST_LOCK = threading.Lock()


def _request_order(item_names: List[str], quantities: List[int]):
    """
    While a request is being handled, the order is every item the inventory check marked
    available, in the quantity the customer asked for (in catalog units). The customer
    accepts the whole quote, so an agent naming only some of the items, e.g. one call per
    item, still gets the whole order and its bulk discount.

    Returns (item_names, quantities), or an error dict if a named item is not available.
    """
    if "request_date" not in REQUEST_CONTEXT:
        return item_names, quantities
    if "inventory" not in REQUEST_CONTEXT:
        return {"status": "error", "reason": "Check inventory for this request first."}

    # Two requested products can match the same catalog item, so add up their quantities
    available = {}
    for row in REQUEST_CONTEXT["inventory"]:
        if row["status"] == "available":
            available[row["item_name"]] = available.get(row["item_name"], 0) + row["quantity"]
    names = list(dict.fromkeys(_catalog_name(name) or name for name in item_names))
    unavailable = [name for name in names if name not in available]
    if unavailable:
        available_text = ", ".join(f"{name} x {quantity}" for name, quantity in available.items())
        return {
            "status": "error",
            "reason": f"Not available for this request: {', '.join(unavailable)}. "
                      f"Available items: {available_text or 'none'}.",
        }
    return list(available), list(available.values())


def _quantities_not_in_request(descriptions: List[str], quantities: List[int], units: List[str]) -> Optional[str]:
    """
    While a request is being handled, return why a quantity or its pack unit is not one the
    customer wrote (e.g. 200 sheets passed as 1 ream), or is one already checked for another
    item (a substitute), or None when they all match.
    """
    request = REQUEST_CONTEXT.get("request_text", "").lower()
    if not request:
        return None
    # Each quantity the customer wrote can be checked once
    used = [
        (int(row["requested_quantity"]), _packaging_unit(row["requested_unit"]) or _packaging_unit(row["requested"]))
        for row in REQUEST_CONTEXT.get("inventory", [])
    ]
    not_written, already_checked = [], []
    for description, quantity, unit in zip(descriptions, quantities, units):
        package = _packaging_unit(unit) or _packaging_unit(description)
        number = rf"(?<![\d,.])(?:{int(quantity):,}|{int(quantity)})(?!,?\d)"
        # The pack unit (or None) written after each occurrence of the number
        written_units = [_packaging_unit(word) for word in re.findall(rf"{number}\s*([a-z]*)", request)]
        item = f"{quantity} {unit} of '{description}'"
        if package not in written_units:
            not_written.append(item)
        elif written_units.count(package) <= used.count((int(quantity), package)):
            already_checked.append(item)
        used.append((int(quantity), package))

    reasons = []
    if not_written:
        reasons.append(f"The customer did not write {', '.join(not_written)}. Pass each quantity "
                       "and unit exactly as written in the request, without converting them.")
    if already_checked:
        reasons.append(f"The quantity of {', '.join(already_checked)} was already checked for another "
                       "item. Do not check substitutes; report the rows already checked.")
    return " ".join(reasons) or None


def _available_rows(item_name: str) -> List[Dict]:
    """Rows of the current request's inventory check that made item_name available."""
    return [
        row for row in REQUEST_CONTEXT.get("inventory", [])
        if row["status"] == "available" and row["item_name"] == item_name
    ]


def _ordered_as(item_name: str) -> str:
    """How the customer ordered item_name, with any unit conversion, e.g. '200 sheets'."""
    return " + ".join(
        row.get("conversion") or f"{row['requested_quantity']:,} {row['requested_unit']}"
        for row in _available_rows(item_name)
    )


# Tools for inventory agent

def _match_candidates(description: str) -> List[Dict]:
    """
    Return catalog items that share words or a size with description, fitting and best matching first.

    A candidate fits when it has the size the customer asked for (if any) and its
    name covers every finish or material the customer asked for (glossy, matte,
    recycled, ...). For example, "A4 glossy paper" does not fit "A4 paper".
    """
    requested_sizes = _extract_sizes(description)
    keywords = _extract_keywords(description)

    candidates = []
    for item in paper_supplies:
        item_sizes = _extract_sizes(item["item_name"])
        item_words = re.findall(r"[a-z0-9]+", item["item_name"].lower())
        word_scores = {word: _word_match_score(word, item_words) for word in keywords}
        score = sum(word_scores.values()) + 2 * len(requested_sizes & item_sizes)
        if score == 0:
            continue
        size_match = requested_sizes <= item_sizes
        unmatched_words = [word for word, word_score in word_scores.items() if word_score == 0]
        candidates.append({
            "item_name": item["item_name"],
            "fits": size_match and not any(word in TYPE_WORDS for word in unmatched_words),
            "size_match": size_match,
            "unmatched_words": unmatched_words,
            "score": score,
        })

    candidates.sort(key=lambda candidate: (candidate["fits"], candidate["score"]), reverse=True)
    return candidates


def _not_carried_reason(description: str, candidates: List[Dict]) -> str:
    """Explain why no catalog item fits description, based on the closest candidate."""
    if not candidates:
        sizes = sorted(_extract_sizes(description))
        if sizes:
            return f"The catalog has no item in size {', '.join(sizes).upper()}."
        return "The catalog has no matching product."
    closest = candidates[0]
    if not closest["size_match"]:
        sizes = sorted(_extract_sizes(description) - _extract_sizes(closest["item_name"]))
        return f"The closest item, {closest['item_name']}, does not come in size {', '.join(sizes).upper()}."
    missing = [word for word in closest["unmatched_words"] if word in TYPE_WORDS]
    return f"The closest item, {closest['item_name']}, is not {' or '.join(missing)}."


@tool
def find_catalog_items(descriptions: List[str], quantities: List[int], units: List[str]) -> List[Dict]:
    """
    Match each requested product, as the customer described it, to the catalog item that fits it.

    An item fits when it has the size the customer asked for (if any) and its name
    covers every finish or material the customer asked for (glossy, matte, recycled, ...).
    For example, "A4 glossy paper" does not fit "A4 paper", so it is not carried.
    Quantities are converted to the unit the catalog prices the item in: reams of a paper
    priced per sheet hold 500 sheets each. Other packs (packets, boxes, cases, ...) have no
    standard size, so the customer must confirm how many sheets or items they hold.

    Args:
        descriptions (List[str]): Each product as the customer described it, e.g. "A4 glossy paper".
        quantities (List[int]): Requested quantity of each product, in the same order as descriptions.
        units (List[str]): The unit the customer gave each quantity in, as written, e.g. "sheets",
            "reams", "rolls", or "units" when no unit is given. Same order as descriptions.

    Returns:
        List[Dict]: One row per requested product with the customer's description, quantity,
        and unit, the status ('matched', 'needs clarification', or 'not carried'), the exact
        catalog item_name when matched, the quantity in catalog units with the conversion
        and pricing unit, or the reason the item cannot be matched or converted.
    """
    if not len(descriptions) == len(quantities) == len(units):
        return [{"status": "error", "reason": "descriptions, quantities, and units must have the same length."}]

    rows = []
    for description, quantity, unit in zip(descriptions, quantities, units):
        candidates = _match_candidates(description)
        row = {"requested": description, "requested_quantity": quantity, "requested_unit": unit}
        if not (candidates and candidates[0]["fits"]):
            row.update(status="not carried", quantity=quantity,
                       reason=_not_carried_reason(description, candidates))
            rows.append(row)
            continue

        item_name = candidates[0]["item_name"]
        row.update(item_name=item_name, pricing_unit=_pricing_unit(item_name))
        converted = _catalog_quantity(item_name, quantity, unit, description)
        if "reason" in converted:
            row.update(status="needs clarification", quantity=quantity, reason=converted["reason"])
        else:
            row.update(status="matched", **converted)
        rows.append(row)
    return rows


@tool
def check_stock_level(item_name: str, requested_quantity: int, as_of_date: str) -> Dict:
    """
    Check whether a catalog item has enough stock for a requested quantity as of a given date.

    Args:
        item_name (str): Exact catalog item name, as returned by find_catalog_items.
        requested_quantity (int): Quantity the customer wants to buy.
        as_of_date (str): The cutoff date in ISO format (YYYY-MM-DD).

    Returns:
        Dict: The item name, requested quantity, current stock, and whether the stock
        covers the request, or an error if the name is not in the catalog.
    """
    item_name = _catalog_name(item_name) or item_name
    if item_name not in CATALOG:
        return _unknown_item_error(item_name)
    as_of_date = REQUEST_CONTEXT.get("request_date", as_of_date)
    current_stock = _current_stock(item_name, as_of_date)
    return {
        "item_name": item_name,
        "requested_quantity": requested_quantity,
        "current_stock": current_stock,
        "enough_stock": current_stock >= requested_quantity,
    }


@tool
def restock_item(
    item_name: str,
    requested_quantity: int,
    request_date: str,
    needed_by_date: Optional[str] = None,
) -> Dict:
    """
    Order stock of a catalog item from the supplier so a customer order can be filled.

    Orders enough units to cover the requested quantity and keep the item at its
    minimum stock level. The order is placed only if the supplier can deliver by the
    needed-by date and the cash balance covers the cost.

    Args:
        item_name (str): Exact catalog item name, as returned by find_catalog_items.
        requested_quantity (int): Quantity the customer wants to buy.
        request_date (str): Date of the customer request in ISO format (YYYY-MM-DD).
        needed_by_date (str, optional): Date the customer needs the items by in ISO format (YYYY-MM-DD).

    Returns:
        Dict: The status ('ordered', 'not_needed', or 'rejected') and the customer quantity.
        For an order, also the units restocked (not the quantity to sell), the stock after
        the restock, cost, supplier delivery date, and transaction ID; otherwise the reason.
    """
    item_name = _catalog_name(item_name) or item_name
    if item_name not in CATALOG:
        return _unknown_item_error(item_name)

    current_stock = _current_stock(item_name, request_date)
    min_level = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": item_name},
    )
    min_stock_level = int(min_level["min_stock_level"].iloc[0]) if not min_level.empty else 0

    order_quantity = requested_quantity + min_stock_level - current_stock
    if order_quantity <= 0:
        return {
            "status": "not_needed",
            "item_name": item_name,
            "customer_quantity": requested_quantity,
            "current_stock": current_stock,
        }

    supplier_eta = get_supplier_delivery_date(request_date, order_quantity)
    if needed_by_date and pd.Timestamp(supplier_eta) > pd.Timestamp(needed_by_date):
        return {
            "status": "rejected",
            "item_name": item_name,
            "customer_quantity": requested_quantity,
            "reason": f"The supplier can deliver {order_quantity} units by {supplier_eta}, "
                      f"after the needed-by date {needed_by_date}.",
        }

    cost = round(CATALOG[item_name]["unit_price"] * order_quantity, 2)
    cash_balance = get_cash_balance(request_date)
    if cost > cash_balance:
        return {
            "status": "rejected",
            "item_name": item_name,
            "customer_quantity": requested_quantity,
            "reason": f"The stock order costs ${cost:.2f} but only ${cash_balance:.2f} cash is available.",
        }

    transaction_id = create_transaction(item_name, "stock_orders", order_quantity, cost, request_date)
    return {
        "status": "ordered",
        "item_name": item_name,
        "customer_quantity": requested_quantity,
        "restocked_units": order_quantity,
        "stock_after_restock": current_stock + order_quantity,
        "cost": cost,
        "supplier_eta": supplier_eta,
        "transaction_id": transaction_id,
    }


@tool
def check_inventory(
    descriptions: List[str],
    quantities: List[int],
    units: List[str],
    request_date: str,
    needed_by_date: Optional[str] = None,
) -> List[Dict]:
    """
    Check every requested product: match it to the catalog, check its stock as of the
    request date, and restock it from the supplier when needed.

    Products are matched and their quantities converted to catalog units like
    find_catalog_items. Each matched item is restocked like
    restock_item: up to the requested quantity plus its minimum stock level, only if the
    supplier can deliver by the needed-by date and the cash balance covers the cost.
    Each available item gets a ready date: the request date when it is in stock, or the
    supplier delivery date when it has to be restocked.
    While a customer request is being handled, quantities and units must be the ones the
    customer wrote (never converted), each item is checked only once (asking again returns
    its first result), and the rows of every item checked so far are returned.

    Args:
        descriptions (List[str]): Each product as the customer described it, e.g. "A4 glossy paper".
        quantities (List[int]): Requested quantity of each product, in the same order as descriptions.
        units (List[str]): The unit the customer gave each quantity in, as written, e.g. "sheets",
            "reams", "rolls", or "units" when no unit is given. Same order as descriptions.
        request_date (str): Date of the customer request in ISO format (YYYY-MM-DD).
        needed_by_date (str, optional): Date the customer needs the items by in ISO format (YYYY-MM-DD).

    Returns:
        List[Dict]: One row per requested product with the customer's description, quantity,
        and unit, the status ('available', 'not carried', 'needs clarification', or
        'insufficient stock'), the exact catalog item_name and the quantity in catalog units
        when matched, the ready date when available, and a note explaining the status.
    """
    # Agents may call this several times in parallel, e.g. one call per item
    with REQUEST_LOCK:
        if not len(descriptions) == len(quantities) == len(units):
            return [{"status": "error", "note": "descriptions, quantities, and units must have the same length."}]
        checked = REQUEST_CONTEXT.get("inventory", [])
        # An item asked about again keeps its first result, so it is never re-checked
        checked_descriptions = {row["requested"] for row in checked}
        new_items = [item for item in zip(descriptions, quantities, units) if item[0] not in checked_descriptions]
        if not new_items:
            return checked
        if "sale" in REQUEST_CONTEXT:
            return [{"status": "error", "note": "The order for this request is already recorded, "
                                                "so no more items can be checked."}] + checked
        new_descriptions, new_quantities, new_units = (list(values) for values in zip(*new_items))
        mismatch = _quantities_not_in_request(new_descriptions, new_quantities, new_units)
        if mismatch:
            return [{"status": "error", "note": mismatch}] + checked

        rows = _check_items(new_descriptions, new_quantities, new_units,
                            REQUEST_CONTEXT.get("request_date", request_date), needed_by_date)
        if "request_date" in REQUEST_CONTEXT and all(row["status"] != "error" for row in rows):
            REQUEST_CONTEXT["inventory"] = checked + rows
            REQUEST_CONTEXT["needed_by_date"] = REQUEST_CONTEXT.get("needed_by_date") or needed_by_date
            return REQUEST_CONTEXT["inventory"]
        return rows


def _check_items(
    descriptions: List[str],
    quantities: List[int],
    units: List[str],
    request_date: str,
    needed_by_date: Optional[str],
) -> List[Dict]:
    """Match, stock-check, and restock each product for check_inventory, one row per product."""
    rows = find_catalog_items(descriptions, quantities, units)
    for row in rows:
        if row["status"] != "matched":
            row["note"] = row.pop("reason")
            continue

        current_stock = _current_stock(row["item_name"], request_date)
        stock_text = f"{current_stock} {row['pricing_unit']}s in stock"
        restock = restock_item(row["item_name"], row["quantity"], request_date, needed_by_date)
        if current_stock >= row["quantity"]:
            # A restock that only tops up the minimum stock level does not delay this order
            row.update(status="available", ready_date=request_date, note=f"{stock_text}, ready {request_date}.")
        elif restock["status"] == "ordered":
            # The note leaves out the restocked units so they are not mistaken for the order quantity
            row.update(
                status="available",
                ready_date=restock["supplier_eta"],
                note=f"Ready {restock['supplier_eta']}, after a supplier restock.",
            )
        else:
            row.update(status="insufficient stock", note=f"Only {stock_text}. {restock['reason']}")
        if row.get("conversion"):
            row["note"] = f"{row['conversion']}. {row['note']}"
    return rows


class InventoryAgent(ToolCallingAgent):
    """Agent for inventory management and stock transaction processing."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                check_inventory,
                get_all_inventory,
            ],
            model=model,
            name="inventory_processor",
            description="""
            Matches requested items to exact catalog names, converts
            quantities to catalog units, checks stock as of a given date,
            and restocks items when stock is short. Send it each item as
            the customer wrote it, the quantities with the unit of each
            (sheets, reams, rolls, packets, ...), the request date, and
            the needed-by date.
            """,
            instructions="""
            You are an Inventory Management Agent responsible for
            matching customer requests to catalog items, checking
            stock levels, and restocking items when stock is short.

            Your responsibilities include:

            1. Inventory Check:
               - Call check_inventory once with every requested item
                 as the customer described it, its quantity, the unit
                 of that quantity as the customer wrote it (use "units"
                 if none is given), the request date, and the
                 needed-by date. For "200 sheets of colored paper" pass
                 200 and "sheets"; for "500 reams of printer paper" pass
                 500 and "reams".
               - check_inventory matches each item to the catalog,
                 converts the quantity to catalog units itself, checks
                 its stock, and restocks it when the supplier can
                 deliver in time and cash allows. It gives each
                 available item its ready date.
               - Put every item in that single call; never one call
                 per item.
               - If check_inventory returns an error, fix the call as
                 the error says and call it again.

            Decision-Making Rules:
               - Never guess item names, stock levels, or cash balances.
               - Never convert, round, or change quantities or units.
               - Never fabricate database records or tool results.
               - Never change a status, item name, quantity, ready date,
                 or note returned by check_inventory.

            Return one line per row from check_inventory, in this form:
               "<status>: <item_name> x <quantity> <pricing_unit>s
               (customer asked for <requested_quantity> <requested_unit>
               of '<requested>') - <note>"
            For a not carried row, use the customer's wording in place
            of item_name. Put these lines in the short version of your
            answer. Example:
               "available: Standard copy paper x 250000 sheets (customer
               asked for 500 reams of 'printer paper') - 500 reams =
               250,000 sheets (500 sheets per ream). Ready 2025-04-11,
               after a supplier restock."
            """
        )


# Tools for quoting agent

# Bulk discount tiers by total units in the order: (minimum units, discount rate)
DISCOUNT_TIERS = [(10000, 0.15), (5000, 0.10), (1000, 0.05), (0, 0.0)]


def _price_order(item_names: List[str], quantities: List[int]) -> Dict:
    """
    Price an order from catalog unit prices and apply the bulk discount tier for its total units.

    Returns the priced order, or a dict with status 'error' and the reason.
    """
    if len(item_names) != len(quantities):
        return {"status": "error", "reason": "item_names and quantities must have the same length."}
    item_names = [_catalog_name(name) or name for name in item_names]
    unknown_items = [name for name in item_names if name not in CATALOG]
    if unknown_items:
        return _unknown_item_error(", ".join(unknown_items))
    if any(quantity <= 0 for quantity in quantities):
        return {"status": "error", "reason": "Every quantity must be greater than zero."}

    total_units = sum(quantities)
    discount_rate = next(rate for min_units, rate in DISCOUNT_TIERS if total_units >= min_units)

    line_items = []
    for item_name, quantity in zip(item_names, quantities):
        unit_price = CATALOG[item_name]["unit_price"]
        subtotal = round(unit_price * quantity, 2)
        line_items.append({
            "item_name": item_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "pricing_unit": _pricing_unit(item_name),
            "subtotal": subtotal,
            "line_total": round(subtotal * (1 - discount_rate), 2),
        })

    subtotal = round(sum(line["subtotal"] for line in line_items), 2)
    total = round(sum(line["line_total"] for line in line_items), 2)
    return {
        "line_items": line_items,
        "total_units": total_units,
        "discount_rate": discount_rate,
        "subtotal": subtotal,
        "discount_amount": round(subtotal - total, 2),
        "total": total,
    }


@tool
def calculate_quote(item_names: List[str], quantities: List[int]) -> Dict:
    """
    Price an order from catalog unit prices and apply the bulk discount tier.

    Discount tiers by total units in the order: under 1,000 units 0%,
    1,000-4,999 units 5%, 5,000-9,999 units 10%, 10,000 units or more 15%.
    While a customer request is being handled, the quote always covers every item the
    inventory check marked available, in the catalog quantity the inventory check converted
    the customer's quantity to (e.g. 500 reams = 250,000 sheets).

    Args:
        item_names (List[str]): Exact catalog item names, one per line item.
        quantities (List[int]): Quantity of each line item, in the same order as item_names.

    Returns:
        Dict: Line items with quantity, unit price and the unit it is priced per (sheet
        or unit), subtotal, line total after discount, and how the customer ordered the
        item, plus the total units, discount rate, subtotal, discount amount, and total price.
    """
    order = _request_order(item_names, quantities)
    if isinstance(order, dict):
        return order
    quote = _price_order(*order)
    if "inventory" in REQUEST_CONTEXT:
        for line in quote.get("line_items", []):
            line["ordered_as"] = _ordered_as(line["item_name"])
    return quote


class QuotingAgent(ToolCallingAgent):
    """Agent for pricing orders with bulk discounts and explaining quotes using historical data."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                calculate_quote,
                search_quote_history,
            ],
            model=model,
            name="quote_generator",
            description="""
            Prices an order from catalog unit prices with bulk discounts
            and explains the price using past quotes. Send it the exact
            item names, quantities, request date, and customer context
            (job, event, order size).
            """,
            instructions="""
            You are a Sales Quotation Agent responsible for
            pricing customer orders and explaining the price.

            Your responsibilities include:

            1. Price Calculation:
               - Call calculate_quote once with all line items, using
                 exact catalog item names and quantities.
               - calculate_quote applies the catalog unit prices and
                 the bulk discount tier for the total order size.

            2. Quote History Research:
               - Search historical quotations for similar requests
                 to support the explanation.

            3. Quotation Proposal:
               - Explain the discount tier and the pricing rationale
                 in a friendly, customer-facing tone.

            Decision-Making Rules:
               - Use calculate_quote for every price. Never change its
                 unit prices, line totals, discount, or total.
               - Treat historical quotes as context for the
                 explanation, not as prices.
               - If calculate_quote rejects an item name, do not guess
                 another name. Report the rejected name to your manager.
               - Do not assume inventory availability.
               - Do not confirm order fulfillment or delivery dates.
               - Do not create or modify transactions.

            Return the quotation with:
               - Each line item: exact item name, how the customer
                 ordered it (ordered_as), quantity in catalog units, unit
                 price per pricing unit, and line total after discount,
                 e.g. "Standard copy paper: 500 reams (250,000 sheets) at
                 $0.04 per sheet"
               - Total units, discount rate, discount amount, and total
               - A short pricing rationale
            """
        )

# Tools for ordering agent

@tool
def fulfill_order(item_names: List[str], quantities: List[int], date: str) -> Dict:
    """
    Record an accepted order as sales, one transaction per line item.

    The order is priced exactly like calculate_quote (catalog unit prices and the
    bulk discount tier), so the recorded prices match the quote. Nothing is recorded
    unless every line item is in stock. While a customer request is being handled, the
    sale covers every item the inventory check marked available, uses the request date
    and the quantities from the inventory check, and is recorded only once. Each line item is ready on the date the inventory check gave it,
    and the order's delivery date is the latest of these. The order is not recorded if
    that date is after the customer's needed-by date.

    Args:
        item_names (List[str]): Exact catalog item names, one per line item.
        quantities (List[int]): Quantity of each line item, in the same order as item_names.
        date (str): Date of the sale in ISO format (YYYY-MM-DD).

    Returns:
        Dict: The status ('recorded' or 'rejected'), and for a sale each line item with
        how the customer ordered it, its line total, ready date, transaction ID, and
        remaining stock, plus the order total, delivery date, and needed-by date;
        otherwise the reason.
    """
    # Parallel calls must not record the same order twice
    with REQUEST_LOCK:
        if "sale" in REQUEST_CONTEXT:
            return REQUEST_CONTEXT["sale"]
        return _record_sale(item_names, quantities, date)


def _record_sale(item_names: List[str], quantities: List[int], date: str) -> Dict:
    """Price, stock-check, and record the order for fulfill_order."""
    request_order = _request_order(item_names, quantities)
    if isinstance(request_order, dict):
        return {"status": "rejected", "reason": request_order["reason"]}
    date = REQUEST_CONTEXT.get("request_date", date)

    order = _price_order(*request_order)
    if order.get("status") == "error":
        return {"status": "rejected", "reason": order["reason"]}

    needed = {}
    for line in order["line_items"]:
        needed[line["item_name"]] = needed.get(line["item_name"], 0) + line["quantity"]
    stock = {item_name: _current_stock(item_name, date) for item_name in needed}
    short_items = [
        f"only {stock[item_name]} units of {item_name} in stock"
        for item_name, quantity in needed.items() if stock[item_name] < quantity
    ]
    # Ready dates come from the inventory check, so the sale never promises a different date
    ready_dates = {
        item_name: max((row["ready_date"] for row in _available_rows(item_name)), default=date)
        for item_name in needed
    }
    delivery_date = max(ready_dates.values())
    needed_by_date = REQUEST_CONTEXT.get("needed_by_date")
    reason = None
    if short_items:
        reason = "Not enough stock: " + "; ".join(short_items) + "."
    elif needed_by_date and pd.Timestamp(delivery_date) > pd.Timestamp(needed_by_date):
        reason = (f"The order can be ready on {delivery_date}, after the needed-by date {needed_by_date}. "
                  f"Offer {delivery_date} to the customer as an alternative date.")
    if reason:
        if "request_date" in REQUEST_CONTEXT:
            REQUEST_CONTEXT["sale_rejected"] = reason
        return {"status": "rejected", "reason": reason}

    recorded = []
    for line in order["line_items"]:
        transaction_id = create_transaction(line["item_name"], "sales", line["quantity"], line["line_total"], date)
        recorded.append({
            "item_name": line["item_name"],
            "ordered_as": _ordered_as(line["item_name"]),
            "quantity": line["quantity"],
            "unit_price": line["unit_price"],
            "pricing_unit": line["pricing_unit"],
            "line_total": line["line_total"],
            "ready_date": ready_dates[line["item_name"]],
            "transaction_id": transaction_id,
            "remaining_stock": stock[line["item_name"]] - needed[line["item_name"]],
        })
    sale = {
        "status": "recorded",
        "line_items": recorded,
        "total": order["total"],
        "delivery_date": delivery_date,
        "needed_by_date": needed_by_date,
    }
    if "request_date" in REQUEST_CONTEXT:
        REQUEST_CONTEXT["sale"] = sale
    return sale


class SalesOrderingAgent(ToolCallingAgent):
    """Agent for processing sales orders and managing order fulfillment."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                check_stock_level,
                fulfill_order,
                generate_financial_report,
            ],
            model=model,
            name="sales_order_processor",
            description="""
            Records accepted orders as sales, reports the delivery date
            the inventory check confirmed, and reports financial results.
            Send it the quoted line items with exact item names and
            quantities, and the request date.
            """,
            instructions="""
            You are a Sales Order Processing Agent responsible for
            recording accepted orders and reporting their delivery dates.

            Your responsibilities include:

            1. Sales Order Validation:
               - Verify each line item has an exact catalog item name
                 and a quantity.
               - Check available stock with check_stock_level,
                 passing the line item quantity.

            2. Order Processing:
               - Call fulfill_order once with all line items and the
                 request date (never a delivery date). It prices the
                 order exactly like the quote and records one sale
                 per line item.
               - Never call fulfill_order twice for the same order.
               - If fulfill_order rejects the order, report its reason.

            3. Order Fulfillment:
               - Report the delivery_date returned by fulfill_order as
                 the order delivery date, and the ready_date of each
                 line item. These come from the inventory check.
               - Never estimate or change a delivery date yourself.

            4. Financial Reporting:
               - Generate financial reports when requested.
               - Never fabricate financial results.

            Decision-Making Rules:
               - The quotes you receive have been accepted by the
                 customer. Record them without asking for further
                 confirmation.
               - Never process sales exceeding available inventory.
               - Never invent product prices or transaction details.
               - If a tool rejects an item name, do not guess another
                 name. Report the rejected name to your manager.
               - Clearly distinguish between recorded sales and
                 line items that could not be recorded.

            Return a concise order processing summary including:
               - Product name, how the customer ordered it (ordered_as),
                 and quantity
               - Inventory availability
               - Order processing status
               - Transaction result, if applicable
               - Delivery date from fulfill_order and the needed-by date
               - Any fulfillment issues or recommendations
            """
        )


# Set up your agents and create an orchestration agent that will manage them.

def check_available_items_sold(final_answer, memory, agent=None) -> bool:
    """
    Final answer check for the orchestrator: when the inventory check found available
    items, their sale must be recorded (or rejected by fulfill_order) before the reply.
    """
    inventory = REQUEST_CONTEXT.get("inventory", [])
    available = [row["item_name"] for row in inventory if row["status"] == "available"]
    if available and "sale" not in REQUEST_CONTEXT and "sale_rejected" not in REQUEST_CONTEXT:
        raise ValueError(
            f"These items are available: {', '.join(available)}. Get a quote for them from "
            "quote_generator and record the sale with sales_order_processor before replying. "
            "Items that are not carried or out of stock do not stop the rest of the order."
        )
    return True


MONTH_DATE = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
              r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?")
DATE_IN_TEXT = rf"(?:{MONTH_DATE}|\d{{4}}-\d{{2}}-\d{{2}})"
# A date the reply calls the customer's deadline, e.g. "needed-by date of April 15", "needed by
# April 15", or "the April 15 deadline" (not "after the needed date (April 16)", a supplier date)
DEADLINE_DATE = re.compile(
    rf"(?:(?:deadline|needed[- ]by date|needed date|required (?:delivery )?date)(?:\s+(?:of|is|was)|\s*:)"
    rf"|needed[- ]by)[\s*_]*({DATE_IN_TEXT})|({DATE_IN_TEXT}),?\s+deadline",
    re.IGNORECASE,
)
# Unfilled template text such as "[Your Name]"
PLACEHOLDER = re.compile(r"\[[^\]\n]{0,40}\b(?:name|title|company|contact)\b[^\]\n]{0,40}\]", re.IGNORECASE)


def _parse_reply_date(text: str, year: int) -> Optional[datetime]:
    """Parse a date written in a reply, e.g. 'April 15th, 2025' or '2025-04-15', or return None."""
    text = " ".join(re.sub(r"(?<=\d)(?:st|nd|rd|th)", "", text.replace(".", "").replace(",", " ")).split())
    for date_format in ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        return parsed if "%Y" in date_format else parsed.replace(year=year)
    return None


def check_reply_matches_order(final_answer, memory, agent=None) -> bool:
    """
    Final answer check for the orchestrator: the reply must not call another date the
    customer's deadline or leave template placeholders, and a reply about a recorded sale
    must give the delivery date, total, quantities (e.g. 250,000 sheets for 500 reams), and
    unit prices fulfill_order recorded, so it cannot promise another date, price, or quantity.
    """
    reply = str(final_answer)
    problems = []

    needed_by_date = REQUEST_CONTEXT.get("needed_by_date")
    if needed_by_date:
        needed_by = datetime.fromisoformat(needed_by_date)
        for match in DEADLINE_DATE.finditer(reply):
            written = match.group(1) or match.group(2)
            if _parse_reply_date(written, needed_by.year) not in (None, needed_by):
                problems.append(f"The reply calls {written} the customer's deadline, but the needed-by "
                                f"date is {needed_by:%B} {needed_by.day}, {needed_by.year}.")

    placeholders = PLACEHOLDER.findall(reply)
    if placeholders:
        problems.append(f"The reply contains the placeholder {', '.join(placeholders)}; "
                        "sign it as Munder Difflin Paper Company.")

    sale = REQUEST_CONTEXT.get("sale")
    if sale:
        missing = []
        delivery = datetime.fromisoformat(sale["delivery_date"])
        month = rf"(?:{delivery:%B}|{delivery:%b})\.?"
        day = rf"{delivery.day}(?:st|nd|rd|th)?"
        if sale["delivery_date"] not in reply and not re.search(
            rf"\b{month}\s+{day}\b|\b{day}\s+{month}", reply
        ):
            missing.append(f"the delivery date {delivery:%B} {delivery.day}, {delivery.year}")

        if f"{sale['total']:,.2f}" not in reply and f"{sale['total']:.2f}" not in reply:
            missing.append(f"the total ${sale['total']:,.2f}")

        for line in sale["line_items"]:
            if f"{line['quantity']:,}" not in reply and str(line["quantity"]) not in reply:
                missing.append(f"the quantity of {line['item_name']} as \"{line['ordered_as']}\"")
            if f"{line['unit_price']:.2f}" not in reply:
                missing.append(f"the unit price of {line['item_name']} "
                               f"(${line['unit_price']:.2f} per {line['pricing_unit']})")
        if missing:
            problems.append(f"The reply must give {' and '.join(missing)}, as returned by fulfill_order. "
                            "Do not give the sold items any other delivery date, price, or quantity.")

    if problems:
        raise ValueError(" ".join(problems))
    return True


class OrchestrationAgent(ToolCallingAgent):
    """Agent that coordinates the Inventory, Quoting, and Sales agents to answer customer requests."""

    def __init__(self, model: OpenAIServerModel, orchestrator_model: OpenAIServerModel):
        self.inventory_manager = InventoryAgent(model)
        self.quotation_manager = QuotingAgent(model)
        self.order_manager = SalesOrderingAgent(model)

        super().__init__(
            tools=[],
            model=orchestrator_model,
            managed_agents=[
                self.inventory_manager,
                self.quotation_manager,
                self.order_manager,
            ],
            final_answer_checks=[check_available_items_sold, check_reply_matches_order],
            name="orchestrator",
            description="""
            Coordinates the Munder Difflin agents to turn a customer
            quote request into a quote, a recorded sale, and a reply.
            """,
            instructions="""
            You are the Orchestrator Agent for Munder Difflin Paper Company.
            You handle one customer request at a time by delegating to
            your team of agents. You have no database tools yourself.

            Your team:
               - inventory_processor: checks stock and restocks items.
               - quote_generator: prices items using past quotes.
               - sales_order_processor: records sales and delivery dates.

            Workflow:

            1. Parse the request:
               - List each requested item, its quantity, and the unit
                 of that quantity as the customer wrote it (sheets,
                 reams, rolls, packets, ...).
               - Note the request date and the date the customer
                 needs the items by, if given.

            2. Inventory check (inventory_processor):
               - Send each item exactly as the customer wrote it,
                 with quantities and their units, the request date,
                 and the needed-by date. Never convert units yourself.
               - It returns each item as available, not carried,
                 needs clarification (pack size unknown), or
                 insufficient stock, with the quantity in catalog
                 units and each available item's ready date.

            3. Quotation (quote_generator):
               - Skip this step and step 4 only if no item is available.
                 If at least one item is available, always continue
                 with the available items, even when other items are
                 not carried or out of stock.
               - Send only the available items with exact catalog
                 names, quantities, the request date, and the customer
                 context (job, event, order size).
               - Ask for line items, bulk discount, total price, and
                 a short pricing rationale.

            4. Sales order (sales_order_processor):
               - The customer accepts the quote immediately.
               - Send the quoted line items with exact catalog names
                 and quantities, and the request date.
               - Ask it to record one sale per line item and return
                 the transaction results and the delivery date from
                 fulfill_order.

            5. Reply to the customer:
               - Include the quote, discount, total price, and
                 delivery date, using the catalog name of each item.
               - For each sold item give its quantity, unit price per
                 sheet or unit, and line total.
               - Give the needed-by date exactly as the customer did.
               - Sign as Munder Difflin Paper Company. Never leave
                 placeholders such as [Your Name].
               - For an item ordered in packs, show the customer's
                 quantity, the converted quantity, and the pricing
                 unit, e.g. "500 reams (250,000 sheets) of Standard
                 copy paper at $0.04 per sheet".
               - The delivery date of the sold items is the
                 delivery_date from fulfill_order. Never give them
                 another date, and compare it with the needed-by date.
               - List items that are not carried (we do not sell that
                 product or size) separately from items that are out
                 of stock and cannot be restocked in time. A supplier
                 date after the needed-by date is only an alternative:
                 say it misses the deadline and ask whether the
                 customer wants the item on that date.
               - For items that need clarification, ask the customer
                 how many sheets or items each pack holds.
               - If no item is available, decline politely
                 with the reasons.
               - Only tell the customer an item is ordered if
                 sales_order_processor recorded its sale.

            Decision-Making Rules:
               - Always delegate in the order: inventory, quotation, sales.
               - Call one agent at a time and wait for its answer
                 before calling the next.
               - Send all requested items to inventory_processor
                 in a single call.
               - In tasks for quote_generator and sales_order_processor,
                 name each item by the exact catalog name from
                 inventory_processor (e.g. "Cardstock"), never by the
                 customer's wording.
               - Order each available item in the catalog quantity
                 inventory_processor returned (the customer's quantity
                 converted to sheets or units), never in a restocked
                 quantity.
               - Always pass the request date to every agent.
               - Only use exact item names returned by inventory_processor.
               - Never invent stock levels, prices, or transaction results.
               - Never reveal internal data such as cash balance,
                 inventory value, or profit margins to the customer.
            """
        )

    def handle_request(self, request: str, request_date: str) -> str:
        """Run the full workflow for one customer request and return the reply."""
        task = f"Customer request: {request}\nDate of request: {request_date}"
        REQUEST_CONTEXT.clear()
        REQUEST_CONTEXT["request_date"] = request_date
        REQUEST_CONTEXT["request_text"] = request
        try:
            return str(self.run(task))
        finally:
            REQUEST_CONTEXT.clear()



# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    orchestrator = OrchestrationAgent(model, orchestrator_model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_context = (
            f"{row['request']}\n"
            f"Customer context: {row['job']} organizing a {row['event']} ({row['need_size']} order)"
        )

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try:
            response = orchestrator.handle_request(request_with_context, request_date)
        except Exception as e:
            print(f"Error handling request {idx+1}: {e}")
            response = f"Error: {e}"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
