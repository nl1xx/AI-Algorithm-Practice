import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# 加载文本数据
data = pd.read_csv('your_text_data.csv')
text_data = data['text_column'].tolist()

# 分词
tokenized_text = [word_tokenize(text.lower()) for text in text_data]

from collections import Counter
import numpy as np

# 统计词频
word_counts = Counter([word for text in tokenized_text for word in text])


# 生成共现矩阵
def build_cooccurrence_matrix(tokenized_text, window_size=5):
    vocab = list(word_counts.keys())
    cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))

    for text in tokenized_text:
        for i, word in enumerate(text):
            word_idx = vocab.index(word)
            start = max(0, i - window_size)
            end = min(len(text), i + window_size + 1)

            for j in range(start, end):
                if j != i:
                    co_word_idx = vocab.index(text[j])
                    cooccurrence_matrix[word_idx][co_word_idx] += 1

    return cooccurrence_matrix, vocab


co_occurrence_matrix, vocab = build_cooccurrence_matrix(tokenized_text)

from glove import Glove
glove = Glove().fit(co_occurrence_matrix, epochs=100, no_threads=4, verbose=True)
glove.save('glove_model.model')


glove = Glove().load('glove_model.model')
word_vector = glove.word_vectors[glove.dictionary['example']]
# 获取最相似的5个词
similar_words = glove.most_similar('example', number=5)
