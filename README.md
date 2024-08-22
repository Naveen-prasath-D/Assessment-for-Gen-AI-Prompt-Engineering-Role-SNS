# Problem Statement 1: Natural Language Processing (NLP)

## Problem Statement
Implement a function to preprocess and tokenize text data using Python. The function should handle edge cases such as punctuation, stop words, and different cases. The function should be implemented using libraries like NLTK or spaCy.

### Requirements:
- Implement the function in Python using NLTK or spaCy.
- Handle edge cases like punctuation, stop words, and different cases.

### Evaluation Criteria:
- Correctness of the preprocessing steps.
- Efficiency and readability of the code.
- Clean and structured code with appropriate comments.

## Solution

### Python Code

```python
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text, method='nltk'):
    """
    Preprocesses and tokenizes the input text using either NLTK or spaCy.
    
    Parameters:
        text (str): The input text to preprocess.
        method (str): The library to use for preprocessing ('nltk' or 'spacy').
    
    Returns:
        list: A list of processed tokens.
    """
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords using NLTK
    if method == 'nltk':
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Tokenize and remove stopwords using spaCy
    elif method == 'spacy':
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    else:
        raise ValueError("Method must be either 'nltk' or 'spacy'")
    
    return tokens

# Example usage
sample_text = "Hello world! This is a test sentence, for tokenizing with NLP."
tokens_nltk = preprocess_text(sample_text, method='nltk')
tokens_spacy = preprocess_text(sample_text, method='spacy')

print("NLTK tokens:", tokens_nltk)
print("spaCy tokens:", tokens_spacy)
```

## Explanation
- **nltk** and **spaCy** libraries are used to preprocess and tokenize text data.
- The text is first converted to lowercase, and punctuation is removed.
- Depending on the chosen method ('nltk' or 'spacy'), tokenization is done and stop words are removed.
- The function returns a list of processed tokens.

## Output
Output Using NLTK:
```python
NLTK tokens: ['hello', 'world', 'test', 'sentence', 'tokenizing', 'nlp']
```
Output Using spaCy:
```python
spaCy tokens: ['hello', 'world', 'test', 'sentence', 'tokenizing', 'nlp']
```
