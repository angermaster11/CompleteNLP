# NLP - Natural Language Processing

## **1️⃣ Natural Language Processing (NLP)**

**Definition:**

NLP is the **umbrella field** that deals with how computers interact with human language. It includes **both understanding and generating language**.

- Everything from tokenizing text → classification → translation → summarization falls under NLP.

**What NLP actually does:**

- Breaking text into tokens
- Removing stop-words
- POS tagging
- Named entity recognition
- Sentiment analysis
- Text classification
- Topic modeling
- Summarization
- Machine translation

## **2️⃣ Natural Language Understanding (NLU)**

**Definition:**

NLU is a **subfield of NLP** focused on making machines **understand human language**. It’s about extracting **meaning and intent** from text.

**Goal:** “What does this text mean?”

**Key tasks in NLU:**

1. **Intent Recognition:** Understand what the user wants
    - Example: “Book me a flight to Paris” → intent = book_flight
2. **Entity Extraction:** Find relevant information
    - Example: “Book a flight to Paris next Monday” → entity = destination: Paris, date: next Monday
3. **Sentiment Analysis:** Detect emotions / opinions
4. **Semantic Parsing:** Convert text into structured meaning
5. **Co-reference Resolution:** Determine “he/she/it” refers to whom

**Example:**

Input: “Can you tell me the weather in New York tomorrow?”

NLU Output:

```json
{
  "intent": "get_weather",
  "entities": {
    "location": "New York",
    "date": "tomorrow"
  }
}

```

**Summary:** NLU = **understanding the meaning** behind text.

## **3️⃣ Natural Language Generation (NLG)**

**Definition:**

NLG is a **subfield of NLP** that focuses on **producing human-like language** from structured data or machine understanding.

**Goal:** “How do I express this in natural language?”

**Key tasks in NLG:**

1. **Text summarization:** Turning long text into short summary
2. **Report generation:** e.g., from sales data → generate report
3. **Chatbots / Conversational AI:** Generate natural responses
4. **Translation:** Converting text to another language (requires some understanding + generation)

**Example:**

Input (structured data):

```json
{
  "temperature": 28,
  "condition": "sunny",
  "city": "New York"
}

```

NLG Output:

```
"The weather in New York tomorrow will be sunny with a high of 28°C."

```

**Summary:** NLG = **converting data or model understanding into natural language.**

# 🔹 1. Heuristic-based NLP

### **Definition**

Rule-based approach where human experts design explicit rules (heuristics) for processing text.

→ Focuses on **linguistic knowledge** (grammar, dictionaries, regex, parsing rules).

### **Approach**

- Create rules manually for tokenization, stemming, POS tagging, etc.
- Use pattern-matching and symbolic methods.

### **Example**

- Detecting emails in text using regex:
    
    ```python
    import re
    text = "Contact me at test@example.com"
    re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    
    ```
    
- Chatbots with predefined rules:
    
    ```
    if "hello" in input: return "Hi! How can I help you?"
    
    ```
    

### ✅ Advantages

- Interpretable & easy to debug.
- No large dataset required.
- Works well for simple tasks (tokenization, regex-based entity extraction).

### ❌ Disadvantages

- Not scalable for complex tasks.
- Struggles with ambiguity, sarcasm, context.
- Rules need manual updates → poor adaptability.

# 🔹 2. Machine Learning (ML)-based NLP

### **Definition**

Statistical approach where **models learn patterns from labeled data** instead of hand-coded rules.

### **Approach**

- Feature extraction (Bag-of-Words, TF-IDF, n-grams).
- Classical ML algorithms (Naive Bayes, SVM, Logistic Regression).

### **Example**

- **Spam detection**: Train classifier using TF-IDF features + Logistic Regression.
    
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    X = ["Free money!!!", "Let’s meet tomorrow"]
    y = [1, 0]  # 1 = spam, 0 = not spam
    
    vec = TfidfVectorizer()
    X_vec = vec.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vec, y)
    
    print(model.predict(vec.transform(["Get free cash now!"])))
    
    ```
    

### ✅ Advantages

- Learns from data (less manual rule-writing).
- More adaptable than heuristics.
- Works well for structured tasks (classification, sentiment analysis).

### ❌ Disadvantages

- Requires labeled datasets.
- Needs feature engineering (manual design of features).
- Limited context understanding (bag-of-words ignores order/semantics).

# 🔹 3. Deep Learning (DL)-based NLP

### **Definition**

Neural network-based approach that **automatically learns features & representations** from raw text, often using embeddings and transformers.

### **Approach**

- Word embeddings (Word2Vec, GloVe) → context-aware embeddings (BERT, GPT).
- Sequence models: RNN, LSTM, GRU, Transformers.
- Pretrained models + fine-tuning.

### **Example**

- **Sentiment analysis using BERT (transformers)**:
    
    ```python
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    print(classifier("I love NLP, it's amazing!"))
    
    ```
    
    → Output: `{'label': 'POSITIVE', 'score': 0.99}`
    

### ✅ Advantages

- Handles context, semantics, polysemy.
- State-of-the-art performance in translation, summarization, Q&A, chatbots.
- Pretrained models reduce data requirement.

### ❌ Disadvantages

- Requires high computation (GPU/TPU).
- Harder to interpret (black-box).
- Needs large datasets for training (if not using pretrained models).

# ⚖️ **Comparison Table**

| Aspect | Heuristic-based | ML-based | DL-based |
| --- | --- | --- | --- |
| **Data need** | None / very low | Medium | High (but pretrained helps) |
| **Feature design** | Manual rules | Manual (TF-IDF, n-grams) | Automatic (embeddings) |
| **Interpretability** | High | Medium | Low |
| **Scalability** | Low | Medium | High |
| **Performance** | Good for simple | Good for medium | Best for complex |
| **Example task** | Regex email detection | Spam classification | ChatGPT, Google Translate |

# Lingustic Terminology

## **1️⃣ Morpheme**

**Definition:**

- A **morpheme** is the **smallest unit of meaning** in a language.
- It can be a word or a part of a word that has meaning.

**Types of Morphemes:**

1. **Free Morpheme:** Can stand alone as a word
    - Example: `book`, `run`, `happy`
2. **Bound Morpheme:** Cannot stand alone; attached to other morphemes
    - Example: `s` (plural), `ed` (past tense), `un-` (negation)

**Example:**

Word: **unhappiness**

- `un-` → bound morpheme (negation)
- `happy` → free morpheme (base word)
- `ness` → bound morpheme (noun suffix)

> So morphemes = meaning units, smallest building blocks of language.
> 

---

## **2️⃣ Grapheme**

**Definition:**

- A **grapheme** is the **smallest unit in writing** that represents a sound (phoneme).
- It’s essentially a **letter or combination of letters** in a written language.

**Examples:**

1. Single letters: `a`, `b`, `c`
2. Letter combinations representing one sound: `sh`, `th`, `ch`

**Example in a word:**

Word: **cat**

- Graphemes: `c`, `a`, `t`

Word: **ship**

- Graphemes: `sh`, `i`, `p`

> Think: Graphemes are visual symbols that represent sounds.
> 

---

## **3️⃣ Phoneme**

**Definition:**

- A **phoneme** is the **smallest unit of sound** in a language that can change the meaning of a word.
- Phonemes are **spoken sounds**, not letters.

**Examples:**

1. Word: **bat** → phonemes: `/b/`, `/æ/`, `/t/`
2. Word: **cat** → phonemes: `/k/`, `/æ/`, `/t/`

Notice: Changing one phoneme changes meaning:

- `/b/` → `/c/` → `bat` → `cat`

**Phoneme vs Grapheme:**

- Grapheme = written symbol (`c`)
- Phoneme = sound (`/k/`)

---

### **Summary Table**

| Term | Unit Type | Example (Word: “unhappiness”) |
| --- | --- | --- |
| Morpheme | Meaning unit | `un-`, `happy`, `-ness` |
| Grapheme | Written unit | `u`, `n`, `h`, `a`, `p`, `p`, `i`, `n`, `e`, `s`, `s` |
| Phoneme | Sound unit | `/ʌ/`, `/n/`, `/h/`, `/æ/`, `/p/`, `/i/`, `/n/`, `/ɪ/`, `/s/` |

# 🔹 1. **Morphology**

👉 Study of **word structure** (how words are formed from smaller units called **morphemes**).

- **Morpheme** = smallest meaningful unit of a word.
    - Example: *unhappiness* → **un-** (prefix, negation) + **happy** (root) + **ness** (suffix, state of).
- **Morphological analysis in NLP**:
    - Tokenization (breaking into words).
    - Lemmatization (reduce words to base form: *running* → *run*).
    - Stemming (remove affixes: *studies* → *studi*).

✅ Helps in **text normalization** for search, translation, and sentiment analysis.

---

# 🔹 2. **Syntax**

👉 Study of **sentence structure** (how words combine into phrases and sentences).

- Concerned with **grammar rules**.
- Example:
    - Correct: *The cat sat on the mat.*
    - Incorrect: *Cat the mat sat on.*
- **Syntactic analysis in NLP**:
    - POS tagging (noun, verb, adjective, etc.).
    - Parsing (dependency tree, constituency tree).
    - Example dependency parse:
        
        ```
        "Dogs chase cats"
          Dogs → subject
          chase → verb
          cats → object
        
        ```
        

✅ Important for **machine translation, question answering, grammar checkers**.

---

# 🔹 3. **Semantics**

👉 Study of **meaning** of words and sentences.

- Goes beyond structure to what is being **communicated**.
- Example:
    - *John bought a car* → meaning: John is the buyer, car is the item.
    - Synonyms: *big* ≈ *large*.
    - Ambiguity: *bank* → financial institution / riverbank.
- **Semantic analysis in NLP**:
    - Word embeddings (Word2Vec, GloVe, BERT).
    - Named Entity Recognition (NER): *Apple* → company vs. fruit.
    - Semantic role labeling: *Who did what to whom?*

✅ Needed for **search engines, chatbots, knowledge graphs**.

---

# 🔹 4. **Pragmatics**

👉 Study of **meaning in context** (how real-world knowledge, intent, and situation affect interpretation).

- Example:
    - *Can you pass the salt?* (not a question about ability, but a polite request).
    - *It’s cold in here* (could mean “close the window”).
- **Pragmatic analysis in NLP**:
    - Dialogue systems & chatbots (understanding intent).
    - Sentiment & sarcasm detection.
    - Contextual understanding in conversation.

✅ Essential for **human-like interaction, personal assistants, social media analysis**.

---

# ⚖️ **Summary Table**

| Level | Focus | Example | NLP Techniques |
| --- | --- | --- | --- |
| Morphology | Word formation | *un+happy+ness* | Stemming, Lemmatization |
| Syntax | Sentence structure | *The cat sat* | POS tagging, Parsing |
| Semantics | Word & sentence meaning | *Bank = river/finance* | Embeddings, NER |
| Pragmatics | Meaning in context | *Can you pass the salt?* | Dialogue management, Intent detection |

# 🔹 **1. Semantics**

👉 Focus: **Literal meaning** of words, phrases, and sentences.

- Independent of context or situation.
- Concerned with **dictionary meaning** and how meanings combine logically.

**Example:**

- Sentence: *“It’s cold in here.”*
- **Semantic meaning**: The temperature inside is low.

---

# 🔹 **2. Pragmatics**

👉 Focus: **Intended meaning** in a **specific context**.

- Concerned with **speaker’s intention**, social situation, shared knowledge, tone.
- Goes beyond literal meaning.

**Example:**

- Sentence: *“It’s cold in here.”*
- **Pragmatic meaning** (depending on context):
    - Request: “Please close the window.”
    - Complaint: “Turn on the heater.”
    - Invitation: “Come sit closer.”

## **1️⃣ What is Word Tokenization?**

Word tokenization is the process of **splitting a sentence or text into individual words**, which are called **tokens**. These tokens are the basic units for further NLP tasks like parsing, sentiment analysis, or machine learning.

**Example:**

Text:

```
"I love Python!"
```

Word tokenization output:

```
["I", "love", "Python", "!"]
```

Notice how punctuation is often treated as separate tokens.

---

## **2️⃣ Why is it Important?**

- NLP algorithms cannot understand raw text; they need text in a **structured form**.
- Tokenization helps in **counting words**, **building vocabularies**, and creating **features** for ML models.
- It’s the first step for tasks like:
    - Sentiment Analysis
    - Language Modeling
    - Machine Translation
    - Text Classification

---

## **3️⃣ Types of Tokenization**

1. **Whitespace Tokenization:**
    - Splits text simply by spaces.
    - **Limitation:** Doesn’t handle punctuation properly.
    
    ```python
    text = "I love Python!"
    tokens = text.split()
    print(tokens)
    # Output: ['I', 'love', 'Python!']
    
    ```
    
    Notice `Python!` still has punctuation.
    
2. **Punctuation-Aware Tokenization:**
    - Uses libraries to handle punctuation properly.
    
    **Example with NLTK:**
    
    ```python
    import nltk
    from nltk.tokenize import word_tokenize
    
    nltk.download('punkt')  # download tokenizer models
    text = "I love Python!"
    tokens = word_tokenize(text)
    print(tokens)
    # Output: ['I', 'love', 'Python', '!']
    
    ```
    
3. **Subword Tokenization:**
    - Breaks words into smaller units (like WordPiece or Byte Pair Encoding).
    - Used in **transformers (BERT, GPT, etc.)** to handle rare or unknown words.
    
    **Example:** `"unhappiness"` → `["un", "happi", "ness"]`
    

---

## **4️⃣ Practical Example in Python**

```python
from nltk.tokenize import word_tokenize

text = "Tokenization is important for NLP."
tokens = word_tokenize(text)
print(tokens)
# Output: ['Tokenization', 'is', 'important', 'for', 'NLP', '.']

```

---

## **5️⃣ Common Libraries for Word Tokenization**

| Library | Features |
| --- | --- |
| **NLTK** | Classic NLP, handles punctuation, multiple languages |
| **spaCy** | Fast, built-in pipelines, good for large texts |
| **HuggingFace Tokenizers** | Subword tokenization for transformers, extremely fast |
| **TextBlob** | Simple, beginner-friendly |

## **1️⃣ What is Word Normalization?**

Word normalization is the process of **reducing words to a standard or canonical form** so that words with the same meaning are treated the same way.

It’s a key **text preprocessing step** after tokenization.

**Example:**

Text:

```python
"Running runners ran easily"
```

Normalized form (after stemming/lemmatization):

```python
["run", "runner", "run", "easy"]
```

Notice how different forms of the word “run” are normalized to a base form.

---

## **2️⃣ Why is Word Normalization Important?**

- Reduces **vocabulary size** → fewer unique words for ML models.
- Helps in **matching words** with the same meaning (e.g., “run” = “running” = “ran”).
- Improves **text analysis tasks** like:
    - Information Retrieval (search engines)
    - Sentiment Analysis
    - Topic Modeling
    - Machine Translation

---

## **3️⃣ Common Techniques for Word Normalization**

### **a) Lowercasing**

- Converts all words to lowercase.

```python
text = "Running Runners Ran Easily"
normalized_text = text.lower()
print(normalized_text)
# Output: "running runners ran easily"
```

## **1️⃣ What is Stemming?**

👉 **Stemming** is the process of reducing a word to its **root/base form** (called the *stem*).

- The stem may not always be a valid dictionary word.
- The goal is to **normalize similar words** to the same stem, so they can be treated alike in NLP tasks.

**Example:**

```
"connection", "connected", "connecting" → "connect"
"studies", "studying" → "studi"

```

---

## **2️⃣ Why Use Stemming?**

- Reduces **vocabulary size** in text analysis.
- Improves **search engines**: searching “connect” will also match “connected” or “connection”.
- Helps **information retrieval** and **text mining**.

---

## **3️⃣ Types of Stemmer**

There are several stemming algorithms. Here are the main ones:

---

### **a) Porter Stemmer (Most Common)**

- Developed by **Martin Porter (1980)**.
- Rule-based, applies a sequence of rules to strip suffixes.
- Produces stems that may not be valid words.

**Example:**

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = ["connection", "connected", "connecting", "studies", "studying"]
print([ps.stem(w) for w in words])
# Output: ['connect', 'connect', 'connect', 'studi', 'studi']

```

---

### **b) Snowball Stemmer (Improved Porter)**

- Also developed by **Porter** (aka **Porter2 Stemmer**).
- More accurate, supports multiple languages (English, French, German, etc.).

**Example:**

```python
from nltk.stem import SnowballStemmer

ss = SnowballStemmer("english")
words = ["connection", "connected", "connecting", "studies", "studying"]
print([ss.stem(w) for w in words])
# Output: ['connect', 'connect', 'connect', 'studi', 'studi']

```

---

### **c) Lancaster Stemmer**

- Very **aggressive** (over-stemming, removes too much).
- Produces very short stems, sometimes inaccurate.

**Example:**

```python
from nltk.stem import LancasterStemmer

ls = LancasterStemmer()
words = ["connection", "connected", "connecting", "studies", "studying"]
print([ls.stem(w) for w in words])
# Output: ['connect', 'connect', 'connect', 'study', 'study']

```

Notice it makes "studies" → "study" instead of "studi" (better in some cases).

---

### **d) Lovins Stemmer**

- One of the earliest stemmers (1968).
- Removes the longest suffix in one step.
- Less commonly used now, but historically important.

---

### **e) Paice-Husk Stemmer**

- Iterative, rule-based stemmer.
- Very aggressive (like Lancaster).
- Used in some IR (information retrieval) systems.

---

## **4️⃣ Comparison of Stemmers**

| Stemmer | Features | Pros | Cons |
| --- | --- | --- | --- |
| **Porter** | Rule-based, classic | Simple, widely used | Sometimes inaccurate |
| **Snowball** | Porter2, multilingual | More accurate | Slightly slower |
| **Lancaster** | Aggressive | Short stems, fast | Over-stemming |
| **Lovins** | Oldest, longest suffix removal | Fast, simple | Outdated |
| **Paice-Husk** | Iterative rules | Flexible | Over-stemming |

---

✅ **Quick Tip:**

- For **general NLP tasks** → use **Snowball** (balanced).
- For **search engines/IR** → **Porter** or **Lancaster** may work.
- For **multilingual tasks** → **Snowball** is best.

## **1️⃣ What is Lemmatization?**

**Lemmatization** is the process of reducing a word to its **dictionary/base form**, called a **lemma**.

- Unlike stemming, the result is always a **valid word**.
- Takes **context and part-of-speech (POS)** into account.

**Example:**

| Word | Lemma (Verb) | Lemma (Noun) |
| --- | --- | --- |
| running | run | running |
| ran | run | ran |
| better | good | better |
| studies | study | study |

---

## **2️⃣ Why Lemmatization is Important**

- Produces **meaningful root words** → better for NLP tasks than stemming.
- Helps in:
    - **Text classification**
    - **Sentiment analysis**
    - **Information retrieval**
    - **Question answering**
- Reduces vocabulary size without losing meaning.

---

## **3️⃣ Stemming vs Lemmatization**

| Feature | Stemming | Lemmatization |
| --- | --- | --- |
| Output | May not be a valid word | Always a valid word |
| Accuracy | Less accurate | More accurate |
| Use of POS | No | Yes |
| Speed | Faster | Slower |
| Example | studies → studi | studies → study |

---

## **4️⃣ How to Do Lemmatization in Python**

### **a) Using NLTK**

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

words = ["running", "ran", "better", "studies"]

# Lemmatize as verbs
print([lemmatizer.lemmatize(w, pos='v') for w in words])
# Output: ['run', 'run', 'better', 'study']

# Lemmatize as adjectives
print([lemmatizer.lemmatize(w, pos='a') for w in words])
# Output: ['running', 'ran', 'good', 'studies']

```

---

### **b) Using spaCy (Recommended for Practical NLP)**

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("running ran better studies")
lemmas = [token.lemma_ for token in doc]
print(lemmas)
# Output: ['run', 'run', 'good', 'study']

```

**Advantages of spaCy:**

- Automatically detects **POS**.
- Works on **entire sentences**.
- Faster and production-ready.

---

## **5️⃣ Key Points**

- Lemmatization = **meaningful root word**
- Stemming = **rough root word (may not be valid)**
- Always use **POS tagging** for best results.
- Lemmatization is **slightly slower** than stemming but much more accurate.

## **1️⃣ What is Lemmatization?**

**Lemmatization** is the process of reducing a word to its **dictionary/base form**, called a **lemma**.

- Unlike stemming, the result is always a **valid word**.
- Takes **context and part-of-speech (POS)** into account.

**Example:**

| Word | Lemma (Verb) | Lemma (Noun) |
| --- | --- | --- |
| running | run | running |
| ran | run | ran |
| better | good | better |
| studies | study | study |

---

## **2️⃣ Why Lemmatization is Important**

- Produces **meaningful root words** → better for NLP tasks than stemming.
- Helps in:
    - **Text classification**
    - **Sentiment analysis**
    - **Information retrieval**
    - **Question answering**
- Reduces vocabulary size without losing meaning.

---

## **3️⃣ Stemming vs Lemmatization**

| Feature | Stemming | Lemmatization |
| --- | --- | --- |
| Output | May not be a valid word | Always a valid word |
| Accuracy | Less accurate | More accurate |
| Use of POS | No | Yes |
| Speed | Faster | Slower |
| Example | studies → studi | studies → study |

---

## **4️⃣ How to Do Lemmatization in Python**

### **a) Using NLTK**

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

words = ["running", "ran", "better", "studies"]

# Lemmatize as verbs
print([lemmatizer.lemmatize(w, pos='v') for w in words])
# Output: ['run', 'run', 'better', 'study']

# Lemmatize as adjectives
print([lemmatizer.lemmatize(w, pos='a') for w in words])
# Output: ['running', 'ran', 'good', 'studies']

```

---

### **b) Using spaCy (Recommended for Practical NLP)**

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("running ran better studies")
lemmas = [token.lemma_ for token in doc]
print(lemmas)
# Output: ['run', 'run', 'good', 'study']

```

**Advantages of spaCy:**

- Automatically detects **POS**.
- Works on **entire sentences**.
- Faster and production-ready.

---

## **5️⃣ Key Points**

- Lemmatization = **meaningful root word**
- Stemming = **rough root word (may not be valid)**
- Always use **POS tagging** for best results.
- Lemmatization is **slightly slower** than stemming but much more accurate.

---

If you want, I can make a **visual diagram showing Stemming vs Lemmatization vs Tokenization** so you can see **the full NLP preprocessing pipeline clearly**.

Do you want me to do that?

### You said:

Stop Words

### ChatGPT said:

Alright! Let’s go **deep into stop words** in NLP.

---

## **1️⃣ What are Stop Words?**

**Stop words** are **common words in a language** that usually do **not add significant meaning** to a sentence.

- Examples in English:
    
    ```
    a, an, the, is, are, in, on, at, for, of, and, but
    
    ```
    
- They are often **removed during text preprocessing** to reduce noise in NLP tasks.

**Example:**

Text:

```
"This is a simple example of stop words."

```

After removing stop words:

```
"simple example stop words"

```

Notice how the meaning is **still preserved**.

---

## **2️⃣ Why Remove Stop Words?**

1. **Reduce Vocabulary Size**
    - Makes text processing faster.
2. **Focus on Important Words**
    - Words like “Python”, “data”, “NLP” are more meaningful than “is” or “the”.
3. **Improve ML Models**
    - Less noise, better feature extraction, better accuracy.
4. **Faster Search/Indexing**
    - In search engines, stop words don’t add value.

---

## **3️⃣ Stop Words in Different Languages**

- Every language has its own set of stop words.
- Most NLP libraries provide **predefined lists**:
    - English: “the”, “is”, “a”, “and”
    - Hindi: “है”, “का”, “को”
    - Spanish: “el”, “la”, “y”

---

## **4️⃣ How to Remove Stop Words in Python**

### **a) Using NLTK**

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
text = "This is a simple example of stop words."
words = text.split()

filtered_words = [w for w in words if w.lower() not in stop_words]
print(filtered_words)
# Output: ['simple', 'example', 'stop', 'words.']

```

---

### **b) Using spaCy**

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a simple example of stop words.")

filtered_words = [token.text for token in doc if not token.is_stop]
print(filtered_words)
# Output: ['simple', 'example', 'stop', 'words', '.']

```

**Advantage of spaCy:**

- Automatically handles **case**, **punctuation**, and **POS-based stop words**.

---

## **5️⃣ Custom Stop Words**

- You can also define your **own stop words list** depending on the domain.
- Example: In a medical text, words like “patient” may not be meaningful and can be added to stop words.

```python
custom_stopwords = set(stopwords.words('english'))
custom_stopwords.update(["example", "stop"])

```

---

✅ **Summary**

| Feature | Description |
| --- | --- |
| What | Common words with little meaning |
| Why remove | Reduce noise, smaller vocabulary, better ML |
| Libraries | NLTK, spaCy, Gensim |
| Customizable | Yes, can add/remove domain-specific words |
