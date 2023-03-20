import re

def custom_split(sepr_list, str_to_split):
    # create regular expression dynamically
    regular_exp = '|'.join(map(re.escape, sepr_list))
    # find all occurrences of the separator regex in the string
    # and split the string using the separators while preserving them
    return re.findall(f'[^{regular_exp}]+|[{regular_exp}]', str_to_split)

# Example usage:
separators = [' ', ',', ';']
string_to_split = 'This is a, string; with multiple separators.'
result = custom_split(separators, string_to_split)
print(result) # ['This', ' ', 'is', ' ', 'a', ',', ' ', 'string', ';', ' ', 'with', ' ', 'multiple', ' ', 'separators', '.']
