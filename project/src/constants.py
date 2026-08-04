import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# seed for reproducibility
SEED = int(os.getenv("SEED", 42))

# File Paths
DATA_PATH = os.getenv("DATA_PATH", "data.csv")
CODE_BOOK_PATH = os.getenv("CODE_BOOK_PATH", "code_book.json")

# Model Parameters
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", 0.3))
