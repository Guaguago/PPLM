import math
import torch
import statistics
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import logging

logging.basicConfig(level=logging.INFO)

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()


#
def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


#
# def score(sentence):
#     indexed_tokens = tokenizer.encode(sentence)
#     tokens_tensor = torch.tensor([indexed_tokens])
#     with torch.no_grad():
#         outputs = model.forward(tokens_tensor, labels=tokens_tensor)
#     loss = outputs[0]
#     return math.exp(loss.item())


# tokenizer_LM = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
# model_LM = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

# def score(sent):
#     indexed_tokens = tokenizer.encode(sent)
#     tokens_tensor = torch.tensor([indexed_tokens])
#     with torch.no_grad():
#         outputs = model.forward(tokens_tensor, labels=tokens_tensor)
#     loss = outputs[0]
#     return math.exp(loss.item())

topic = 'religion'
labels = ['B', 'BR', 'BC', 'BCR']
locations = ['automated_evaluation/{}/baseline (B)'.format(topic),
             'automated_evaluation/{}/baseline+reranking (BR)'.format(topic),
             'automated_evaluation/{}/gradient (BC)'.format(topic),
             'automated_evaluation/{}/gradient+reranking (BCR)'.format(topic)]
print('{}:'.format(topic))
lines = []
for l in range(len(labels)):
    with open(locations[l]) as f:
        lines = f.read().splitlines()
        f.close()

    scores = []
    for s in lines:
        if s.strip() != '':
            r = score(s)
            if r < 200:
                scores.append(r)
    # print(scores)
    print('{}: {}'.format(labels[l], statistics.mean(scores)))

    # print('{}: {}'.format(labels[l], statistics.mean([score(s) for s in lines])))
