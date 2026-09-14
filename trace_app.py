from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

# 1. Register Phoenix Tracer Provider (preserves your explicit OTEL collector endpoint)
tracer_provider = register(endpoint="http://localhost:6006/v1/traces")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# 2. Configure OpenAI client pointing to local Ollama instance
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)

# 3. 10 Multi-step complex queries (designed to force >5s generation latency)
queries = [
    "Step 1: Explain quantum computing to a 5-year-old. Step 2: Translate that explanation into French. Step 3: Summarize the French text in exactly 3 bullet points.",
    "Write a Python script to scrape a website, then explain the time complexity of the parsing algorithm used, and finally write a unit test for it.",
    "Design a system architecture for a high-traffic e-commerce site. Break it down into frontend, backend, database, and caching layers. Provide a sequence diagram explanation.",
    "Analyze the pros and cons of microservices vs monolithic architecture. Then, write a step-by-step migration plan for a legacy monolith moving to microservices.",
    "Step 1: Write a 500-word short story about a cyberpunk detective. Step 2: Extract all the adjectives used. Step 3: Rewrite the story without using any adjectives.",
    "Act as an expert DBA. Design a normalized SQL database schema for a hospital management system. Include tables, foreign keys, and write a query to find the most visited doctor.",
    "Explain the mathematics behind the RSA encryption algorithm. Then, provide a step-by-step numerical example using small prime numbers.",
    "Compare and contrast the React, Angular, and Vue.js frameworks. Create a detailed feature comparison matrix in markdown format.",
    "Write a complete business plan for a sustainable coffee shop. Include an executive summary, market analysis, financial projections, and marketing strategy.",
    "Act as a Linux systems administrator. Write a bash script that monitors CPU and RAM usage, logs it to a file, and sends an email alert if it exceeds 80%. Explain each line in detail."
]

print("Executing 10 multi-step trace queries against Ollama (phi3)...")

# 4. Iterate through all 10 queries
for idx, query in enumerate(queries, 1):
    print(f"\n--- [Query {idx}/10] Sending prompt to phi3 ---")
    try:
        response = client.chat.completions.create(
            model="phi3",
            messages=[{"role": "user", "content": query}],
            temperature=0.5
        )
        print(f"Query {idx} completed successfully.")
        print(f"Output preview: {response.choices[0].message.content[:80]}...\n")
    except Exception as e:
        print(f"Error on Query {idx}: {e}")

print("\nAll 10 queries finished! Open Phoenix at http://localhost:6006 to filter and export.")