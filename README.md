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
