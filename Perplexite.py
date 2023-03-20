import openai
import re

openai.api_key = "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"

# Define a function to calculate perplexity
def get_perplexity(prompt, response, type_of_answer = ""):

    if type_of_answer == "chitchat" :
        eng = "text-davinci-002"
    elif type_of_answer == "Q&A": 
        eng = "text-curie-001"
    else :
        eng = "davinci"
    # Make a new API request to get the log probabilities of each token
    result = openai.Completion.create(
        model=eng,
        prompt=prompt + response,
        temperature=0,
        max_tokens=len(response),
        n=1,
        logprobs=100,
    )
    choices = result.choices[0]
    tokens = choices.logprobs.token_logprobs

    # Calculate the average log probability
    if len(tokens) > 0 :
        log_prob = sum([logprob for logprob in tokens]) / len(tokens)
    else : 
        log_prob = 10e10
    # Convert log probability to perplexity
    perplexity = 2 ** (-log_prob)
    return perplexity

'''
# Example usage
prompt = "The quick brown fox jumps over the lazy dog."
response = "This is a test sentence generated by GPT-2."
perplexity = get_perplexity(prompt, response)
print(f"Perplexity: {perplexity:.2f}")'''

def custom_split(sepr_list, str_to_split):
    # create regular expression dynamically
    regular_exp = '|'.join(map(re.escape, sepr_list))
    # find all occurrences of the separator regex in the string
    # and split the string using the separators while preserving them
    L = re.findall(f'[^{regular_exp}]+|[{regular_exp}]', str_to_split)
    #print(L)
    final_L = []
    for i in range(len(L)) :
        if L[i] in sepr_list and i>0:
            #print(L[i],L[i-1])
            final_L.append(L[i-1]+L[i])
    if len(final_L) == 0 :
        final_L.append(L[i])
    return final_L
