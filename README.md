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


# Problem Statement 2: Text Generation

## Problem Statement
Create a basic text generation model using a pre-trained transformer (e.g., GPT-3). The model should generate coherent text based on a given prompt.

### Requirements:
- Use the Hugging Face Transformers library.
- Generate coherent text based on a given prompt.

### Evaluation Criteria:
- Ability to load and use pre-trained models.
- Quality and coherence of the generated text.
- Understanding and application of the transformer model.

## Solution

### Python Code

```python
from transformers import pipeline

# Load pre-trained GPT-3 model via Hugging Face's text generation pipeline
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, max_length=50, num_return_sequences=1):
    """
    Generates text based on a given prompt using a pre-trained transformer model.

    Parameters:
        prompt (str): The initial text to start generating from.
        max_length (int): Maximum length of the generated text.
        num_return_sequences (int): Number of different sequences to generate.

    Returns:
        list: A list containing the generated text sequences.
    """
    generated_texts = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return [text['generated_text'] for text in generated_texts]

# Example usage
prompt = "In a world where artificial intelligence"
generated_text = generate_text(prompt, max_length=50, num_return_sequences=1)

print("Generated Text:")
print(generated_text[0])
```

## Explanation
- The Hugging Face Transformers library is used to load a pre-trained model (GPT-2 in this case).
- The `pipeline` function is used to create a text generation pipeline with the model.
- The `generate_text` function takes a prompt and generates text based on it, with options for maximum length and the number of sequences to return.

## Example Usage
```python
prompt = "In a world where artificial intelligence"
generated_text = generate_text(prompt, max_length=50, num_return_sequences=1)

print("Generated Text:")
print(generated_text[0])
```

## Expected Output
Given the prompt "In a world where artificial intelligence", the model might generate something like:
```
Generated Text:
"In a world where artificial intelligence is ubiquitous, humanity must navigate the complex relationship between man and machine..."
```

Note: The actual generated text will vary depending on the model and settings.
