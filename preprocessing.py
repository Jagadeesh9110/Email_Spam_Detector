import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Initialize PorterStemmer once
ps = PorterStemmer()

# Load stopwords once at module level
all_stopwords = set(stopwords.words('english'))

# Precompile regex patterns
remove_subject_pattern = re.compile(r'Subject: ')
keep_alpha_pattern = re.compile(r'[^a-zA-Z]')

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by removing 'Subject: ', keeping only alphabetic characters,
    converting to lowercase, removing stopwords, and stemming.
    """
    # Remove 'Subject: '
    text = remove_subject_pattern.sub('', text)
    
    # Keep only alphabetic characters, replace others with space
    text = keep_alpha_pattern.sub(' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Split into words
    words = text.split()
    
    # Remove stopwords and stem
    stemmed_words = [ps.stem(word) for word in words if word not in all_stopwords]
    
    # Join back into a string
    return " ".join(stemmed_words)
