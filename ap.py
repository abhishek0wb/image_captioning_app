import pickle

# Create a sample tokenizer dictionary
tokenizer = {
    0: "<PAD>",  # Padding token
    1: "<START>",  # Start-of-sequence token
    2: "<END>",  # End-of-sequence token
    3: "a",
    4: "cat",
    5: "dog",
    6: "is",
    7: "running",
    8: "on",
    9: "the",
    10: "grass",
    11: "man",
    12: "woman",
    13: "riding",
    14: "bicycle",
    15: "helmet",
    16: "on",
    17: "road"
}

# Save the tokenizer as a pickle file
tokenizer_path = "tokenizer.pkl"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"Tokenizer file saved as '{tokenizer_path}'.")
