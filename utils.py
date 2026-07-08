import re
from tqdm import tqdm


def generate_vocab(text):
    text = text.lower()
    data = re.findall(r"[^- :,\n.!?&]+|[\- :,\.!?\&]|\n", text)
    freq_table = {}

    print("generating vocab")
    for d in tqdm(data):
        if d in freq_table:
            freq_table[d] += 1
        else:
            freq_table[d] = 1

    freq_table = list(freq_table.items())
    freq_table.sort(  # at this point it becomes a list of tuples
        key=lambda word: word[1], reverse=True
    )  # order in descending order based on the number of apparitions

    # print(f"vocabulary size is {len(freq_table)}")
    freq_table = [("<pad>", 0)] + freq_table

    stoi = {word: i for i, (word, count) in enumerate(freq_table)}
    itos = {i: word for i, (word, count) in enumerate(freq_table)}

    print("generated vocab")
    return freq_table, stoi, itos


def encode(text, stoi, verbose=0, padding=False):
    if verbose == 1:
        print("encoding text")

    text = text.lower()
    data = re.findall(r"[^- :,\n.!?&]+|[\- :,\.!?\&]|\n", text)
    encoded_text = [stoi[word] for word in data]

    if padding:  # change upon inference to longer texts
        while len(encoded_text) < 100:
            encoded_text.append(stoi["<pad>"])
        if len(encoded_text) > 100:
            encoded_text = encoded_text[:100]

    if verbose == 1:
        print(f"first 20 elements of encoded_text: {encoded_text[:20]}")
    return encoded_text


def decode(code, itos):
    print("Decoding text")
    decoded_text = ""
    for c in code:
        decoded_text += itos[c]

    return decoded_text


if __name__ == "__main__":
    with open("sprawl.txt", "r") as f:
        sprawl = f.read()

    table, stoi, itos = generate_vocab(sprawl)

    code = encode("Hello, World!", stoi)
    print(code)
    print(decode(code, itos))
