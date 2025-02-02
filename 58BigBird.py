# https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Evaluating_Big_Bird_on_TriviaQA.ipynb
# evaluate a pretrained BigBirdForQuestionAnswering model on the validation dataset of TriviaQA

import datasets
import torch
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering

# remove [:5%] to run on full validation set
validation_dataset = datasets.load_dataset("trivia_qa", "rc", split="validation[:5%]")

# print(validation_dataset)
# Dataset({
#     features: ['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer'],
#     num_rows: 18669
# })

# print(validation_dataset.info.features)
# {'answer': {'aliases': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
#   'matched_wiki_entity_name': Value(dtype='string', id=None),
#   'normalized_aliases': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
#   'normalized_matched_wiki_entity_name': Value(dtype='string', id=None),
#   'normalized_value': Value(dtype='string', id=None),
#   'type': Value(dtype='string', id=None),
#   'value': Value(dtype='string', id=None)},
#  'entity_pages': Sequence(feature={'doc_source': Value(dtype='string', id=None), 'filename': Value(dtype='string', id=None), 'title': Value(dtype='string', id=None), 'wiki_context': Value(dtype='string', id=None)}, length=-1, id=None),
#  'question': Value(dtype='string', id=None),
#  'question_id': Value(dtype='string', id=None),
#  'question_source': Value(dtype='string', id=None),
#  'search_results': Sequence(feature={'description': Value(dtype='string', id=None), 'filename': Value(dtype='string', id=None), 'rank': Value(dtype='int32', id=None), 'title': Value(dtype='string', id=None), 'url': Value(dtype='string', id=None), 'search_context': Value(dtype='string', id=None)}, length=-1, id=None)}


# 1. For Questions Answering, all we need is the question, the context and the answer.
# 2. The question is a single entry, so we keep it.
# 3. Because BigBird was trained on the Wikipedia part of TriviaQA, we will use validation_dataset["entity_pages"]["wiki_context"]
# as our context.
# 4. We can also see that there are multiple entries for the answer. In this use case, we define a correct output of the model
# as one that is one of the answer aliases validation_dataset["answer"]["aliases"]. Lastly, we also keep validation_dataset["answer"]["normalized_value"]. All other columns can be disregarded.

def format_dataset(example):
    # the context might be comprised of multiple contexts => me merge them here
    example["context"] = " ".join(("\n".join(example["entity_pages"]["wiki_context"])).split("\n"))
    example["targets"] = example["answer"]["aliases"]
    example["norm_target"] = example["answer"]["normalized_value"]
    return example
validation_dataset = validation_dataset.map(format_dataset, remove_columns=["search_results", "question_source", "entity_pages", "answer", "question_id"])


# 随机选取数据查看
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))
show_random_elements(validation_dataset, num_examples=3)


# 去除部分样本
validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)
# print(validation_dataset)
# Dataset({
#     features: ['context', 'norm_target', 'question', 'targets'],
#     num_rows: 16504
# })


# BigBird is able to process inputs of up to a length of 4096 tokens, has a vocab size of ca. 50K tokens, and makes use of the sentencepiece tokenizer.
# Given this information, we can assume that a single token of BigBird's vocabulary represents roughly 4 characters on average.

# 再次过滤数据集
# short_validation_dataset only consists of data samples that do not exceed 4 * 4096 characters
short_validation_dataset = validation_dataset.filter(lambda x: (len(x['question']) + len(x['context'])) < 4 * 4096)
# print(short_validation_dataset)
# Dataset({
#     features: ['context', 'norm_target', 'question', 'targets'],
#     num_rows: 4159
# })


# 加载模型和分词器
tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc").to("cuda")


# Google's official evaluation scripts for TriviaQA includes many improvements to make sure that:
# 1. multiple aliases of both the model's prediction and the label are created. This way it can be assured that a correct prediction
# that only differs in format to the label is indeed classified as being correct. For this, we will write an expand_to_aliases function
# which will normalise the targets and prediction.
# 2. the model returns a non-empty answer. This can be achieved by using the top_k scores returned by model to filter out scores
# which would lead to an emtpy answer.

PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"']))

def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]

# 为了确保模型的预测结果即使在格式上与标签（label）有所不同，但只要内容正确，也能被正确地分类为“正确预测”，我们需要为模型的预测结果和标签创建多个别名（aliases）
# 如New York和NY实际上可能表示的是同一个...
# 将答案扩展为多个别名，例如去掉标点符号、小写化等
def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        # if answers are longer than one word, make sure a predictions is correct if it corresponds to the complete 1: or :-1 sub word
        # *e.g.* if the correct answer contains a prefix such as "the", or "a"
        given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace('_', ' ').lower()
        alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
        answers.append(' '.join(alias.split()).strip())
    return set(answers)


# Now instead of just taking the most likely start and end index, we can compute the  top_k^2  best start and end index combinations
# and pick the most likely combination that is valid. We define a valid combination as one where the start index is smaller than
# the end index and where  end_index−start_index<max_size
def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    # 确保结束索引必须大于起始索引, 答案的长度不能超过最大限制
    # 将上述两个条件进行逻辑或操作, 生成一个布尔掩码, 如果某个位置的值为True则表示该起始和结束索引组合不满足条件
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


# 评估
def evaluate(example):
    # encode question and context so that they are seperated by a tokenizer.sep_token and cut at max_length
    encoding = tokenizer(example["question"], example["context"], return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
    input_ids = encoding.input_ids.to("cuda")

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids).to_tuple()

    start_score, end_score = get_best_valid_start_end_idx(start_scores[0], end_scores[0], top_k=8, max_size=16)

    # Let's convert the input ids back to actual tokens
    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
    answer_tokens = all_tokens[start_score: end_score + 1]

    example["output"] = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    # replace('"', '')  # remove space prepending space token and remove unnecessary '"'

    answers = expand_to_aliases(example["targets"], make_sub_answers=True)
    predictions = expand_to_aliases([example["output"]])

    # if there is a common element, it's a match
    example["match"] = len(list(answers & predictions)) > 0

    return example

results_short = short_validation_dataset.map(evaluate)
print("Exact Match (EM): {:.2f}".format(100 * sum(results_short['match'])/len(results_short)))

wrong_results = results_short.filter(lambda x: x['match'] is False)
print(f"\nWrong examples: ")
print_out = wrong_results.map(lambda x, i: print(f"{i} - Output: {x['output']} - Target: {x['norm_target']}"), with_indices=True)

results = validation_dataset.map(evaluate)
print("Exact Match (EM): {:.2f}".format(100 * sum(results['match'])/len(results)))
