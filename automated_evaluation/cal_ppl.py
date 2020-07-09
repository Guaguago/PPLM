import torch
import statistics as stat
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import logging
import math

# logging.basicConfig(level=logging.INFO)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()


# calulate ppl of samples from src
def cal_ppl(samples):
    ppls = [score_py(s) for s in samples]
    ppl = stat.mean(ppls)
    return ppl


def score_py(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tokenize_input = ['<|endoftext|>'] + tokenize_input
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


def score_trans(sentence):
    indexed_tokens = tokenizer.encode(sentence)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model.forward(tokens_tensor, labels=tokens_tensor)
    loss = outputs[0]
    return math.exp(loss.item())


def samples(src):
    with open(pos_src, 'r') as f:
        return f.read().split('<|endoftext|>')[1:]


if __name__ == '__main__':
    pos_src = '../data/test/generated_samples/paper/bc_pos(2_150_10)'
    neg_src = '../data/test/generated_samples/paper/bc_neg(2_150_10)'

    pos_samples = samples(pos_src)

    o = cal_ppl(pos_samples)
    print(o)
