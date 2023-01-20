import openai

# Set the API key
openai.api_key = "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"

# Set the prompt
prompt = "What is the capital of France?"

# Use the Completion API to generate a response
completion = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=1024)

# Print the response
print(completion.text)