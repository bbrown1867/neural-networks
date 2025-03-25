"""
Auto-regressive character-level language model.

Bigram language model:
    Working with 2 characters at time. Looking at one character and trying to
    predict the next one. Simple but weak.
"""

import os

import matplotlib.pyplot as plt
import torch


def read_names(fname: str) -> list:
    assert os.path.exists(fname)

    names = []
    with open(fname, "r") as fp:
        for line in fp:
            names.append(line.strip())

    return names


def visualize(data, itos):
    plt.figure(figsize=(16, 16))
    plt.imshow(data, cmap="Blues")

    for i in range(len(itos)):
        for j in range(len(itos)):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, data[i, j].item(), ha="center", va="top", color="gray")

    plt.axis("off")
    plt.show(block=True)


def bigram():
    names = read_names("names.txt")

    num_chars = 27
    N = torch.zeros((num_chars, num_chars), dtype=torch.int32)

    chars = sorted(list(set("".join(names))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0  # start/end sentinel
    itos = {i: s for s, i in stoi.items()}

    # Create the bigram model
    for name in names:
        chs = ["."] + list(name) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    # Convert bigram model to probabilities
    P = N.float()
    P /= P.sum(1, keepdim=True)

    # Sample new names from the bigram model probabilities
    num_new_names = 20
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(num_new_names):
        ix = 0
        new_name = []
        while True:
            p = P[ix]
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            if ix == 0:  # end sentinel
                print("".join(new_name))
                break
            else:
                new_name.append(itos[ix])

    # TODO: Paused video at 50:14 (loss function chapter)


if __name__ == "__main__":
    bigram()
