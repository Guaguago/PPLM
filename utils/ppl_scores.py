import math
# import torch
import statistics
# from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import logging

logging.basicConfig(level=logging.INFO)

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()


def score_py(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
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


# tokenizer_LM = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
# model_LM = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

# def score(sent):
#     indexed_tokens = tokenizer.encode(sent)
#     tokens_tensor = torch.tensor([indexed_tokens])
#     with torch.no_grad():
#         outputs = model.forward(tokens_tensor, labels=tokens_tensor)
#     loss = outputs[0]
#     return math.exp(loss.item())
def ppl_scores(topic, src_dir):
    labels = ['B', 'BR', 'BC', 'BCR']
    locations = ['{}/{}/baseline (B)'.format(src_dir, topic),
                 '{}/{}/baseline+reranking (BR)'.format(src_dir, topic),
                 '{}/{}/gradient (BC)'.format(src_dir, topic),
                 '{}/{}/gradient+reranking (BCR)'.format(src_dir, topic)]
    print('{}:'.format(topic))
    lines = []
    for l in range(len(labels)):
        with open(locations[l]) as f:
            lines = f.read().splitlines()
            f.close()

        scores = []
        for s in lines:
            if s.strip() != '':
                r = score_py(s)
                if r < 200:
                    scores.append(r)
        # print(scores)
        print('{}: {}'.format(labels[l], statistics.mean(scores)))

        # print('{}: {}'.format(labels[l], statistics.mean([score(s) for s in lines])))


if __name__ == '__main__':
    src_dir = '/Users/xuchen/core/pycharm/project/PPLM/automated_evaluation'
    file_info = [
        'computers.csv',
        # 'legal.csv',
        # 'military.csv',
        # 'politics.csv',
        # 'religion.csv',
        # 'science.csv',
        # 'space.csv',
        # 'negative.csv',
        # 'positive.csv',
        # 'clickbait.csv'
    ]
    topic = file_info[0][:-4]
    # tokenizer.special_tokens['sos'] = '<|endoftext|>'
    ppl_scores(topic, src_dir)
