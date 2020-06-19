import math
import torch
import statistics

# from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import logging

logging.basicConfig(level=logging.INFO)

# tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', bos_token='<|endoftext|>')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
# tokenizer.set_special_tokens({'bos_token': '<|endoftext|>'})
#                               'end_token': '<|endoftext|>',
#                               'unk_token': '<|endoftext|>'})
# print(tokenizer.special_tokens)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
# model.resize_token_embeddings(len(tokenizer))
model.eval()


def score_py(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    # indexed_tokens = tokenizer.encode(sentence, add_special_tokens=True)
    tokenize_input = ['<|endoftext|>'] + tokenize_input
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


# tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
# model.resize_token_embeddings(len(tokenizer))


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
    samples = []
    for l in range(len(labels)):
        with open(locations[l]) as f:
            samples = f.read().split('<|endoftext|>')[1:]
            f.close()

        scores = []
        for s in samples:
            scores.append(score_trans(s))
        # print(scores)
        print('{}: {}'.format(labels[l], statistics.mean(scores)))

        # print('{}: {}'.format(labels[l], statistics.mean([score(s) for s in lines])))


if __name__ == '__main__':
    src_dir = '../../automated_evaluation'
    file_info = [
        # 'computers.csv',
        # 'legal.csv',
        # 'military.csv',
        # 'politics.csv',
        # 'religion.csv',
        # 'science.csv',
        # 'space.csv',
        'positive.csv',
        # 'negative.csv',
        # 'clickbait.csv'
    ]
    topic = file_info[0][:-4]
    # tokenizer.special_tokens['sos'] = '<|endoftext|>'
    # ppl_scores(topic, src_dir)

    # file_pos = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/vad/samples/positive/BC'
    # file_pos = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/positive/BC'
    file_neg = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/vad_abs/samples/negative/BC'
    # file_neg = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/negative/BC'
    samples = None
    with open(file_neg, 'r') as f:
        samples = f.read().split('<|endoftext|>')
        samples = [s for s in samples if len(s.split()) > 20]
    scores = [score_py(s) for s in samples]
    import statistics as stat

    print(stat.mean(scores))
    # print(score_trans('<|endoftext|> I love it'))
