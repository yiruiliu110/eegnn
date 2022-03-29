import torch

def stirling_number(n, alpha):
    i = 0
    result = 0
    while i < n:
        p = torch.tensor([alpha / (i + alpha)])
        add_index = torch.bernoulli(p)
        result += add_index.item()
        i += 1

    return int(result)

if __name__ == "__main__":
    result = stirling_number(1, 1)
    print(result)

    result = stirling_number(0, 1)
    print(result)

