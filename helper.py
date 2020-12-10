import nltk
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_count_vect(docs, max_features=None, ngram_range=(1,1)):
    count_vect = CountVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True)
    counts = count_vect.fit_transform(docs)

    return count_vect, counts

def get_tfidf_vect(docs, max_features=None, ngram_range=(1,1)):
    tfidf_vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True)
    tfidf = tfidf_vect.fit_transform(docs)

    return tfidf_vect, tfidf

def get_term_frequencies(matrix):
    term_freq = []

    for t in matrix.T:
        term_freq.append(t.toarray().sum())

    return term_freq

def plot_term_frequencies_sorted(seq, rank=-1, is_log=False):
    if rank is -1:
        rank = len(seq)
    index = np.arange(rank)

    if is_log:
        seq_sorted = np.sort([math.log(i) for i in seq])[::-1]
    else:
        seq_sorted = np.sort(seq)[::-1]

    plt.plot(index, seq_sorted[:rank])
    plt.show()

def plot_sparse_matrix(matrix, precision):
    plt.subplots(figsize=(20, 25))
    plt.spy(matrix, precision=precision, markersize=1)

def term_rank(freq, feature_names, rank, ascending=False):
    if ascending:
        return np.argsort(freq)[:rank]
    else:
        return np.argsort(freq)[::-1][:rank]

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)

def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens

emoji = {
    "😂": "<lolface>",
    "😇": "<smile>",
    "😀": "<smile>",
    "🎉": "party",
    "😳": "embarrassed",
    "😔": "<sadface>",
    "👀": "shifty",
    "🤷": "shrugging",
    "💔": "brokenhearted",
    "👻": "ghost",
    "😍": "<heart>",
    "🙄": "disdain",
    "💖": "<heart>",
    "✌": "victory",
    "🎶": "music",
    "😱": "shock",
    "😃": "<smile>",
    "😒": "unsatisfied",
    "👊": "brofist",
    "😄": "<smile>",
    "🌞": "<smile>",
    "🙌": "celebration",
    "😁": "<smile>",
    "🤗": "hugging",
    "🤣": "rofl",
    "🌈": "gaypride",
    "😉": "winking",
    "💞": "<heart>",
    "🙃": "irony",
    "😜": "winking",
    "😭": "bawling",
    "🤔": "thinker",
    "😎": "cool",
    "💛": "<heart>",
    "💚": "<heart>",
    "💃": "fun",
    "💗": "<heart>",
    "😬": "awkward",
    "😌": "relieved",
    "😅": "whew",
    "💋": "kiss",
    "🙈": "laugh",
    "😊": "^^",
    "👌": "okay",
    "😡": "angry",
    "😘": "kiss",
    "😩": "weary",
    "🔥": "excellent",
    "💙": "<heart>",
    "💕": "<heart>",
    "👏": "clapping",
    "👍": "thumbsup",
    "💯": "perfect",
    "💜": "<heart>",
    "🕘" : "late",
    "😡" : "angry",
    "😒" : "dissatisfied",
    "😤" : "angry",
    "😠" : "angry",
    "😑" : "annoy",
    "😡😡😡" : "angry",
    "😡😡" : "angry",
    "😰": "anxious",
    "😯": "surprise",
    "😨": "scared",
    "😲": "astonished",
    "💪": "strong",
    "🤦": "facepalm",
    "✨": "sparkle",
    "😢": "crying",
    "💓": "<heart>",
    "👑": "crown",
    "🤘": "rockon",
    "🌹": "rose",
    "😋": "delicious",
    "😏": "flirting",
    "😆": "XD",
    "😫": "exhausted",
    "😦": "frowning",
    "🙏": "please",
}

frequent_name_dict = {
    "@realdonaldtrump": "sadness",
    "@fifthharmony": "sadness",
    "@mostrequestlive": "sadness",
    "@onairromeo": "sadness",
    "@matthardybrand": "sadness",
}