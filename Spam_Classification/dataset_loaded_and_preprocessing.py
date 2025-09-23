import pandas as pd
import re


# Load the dataset
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)

# Display basic information about the dataset
print("-------------------- HEAD --------------------")
print(df.head())
print("-------------------- DESCRIBE --------------------")
print(df.describe())
print("-------------------- INFO --------------------")
print(df.info())

"""
-------------------- HEAD --------------------
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...
-------------------- DESCRIBE --------------------
       label                 message
count   5572                    5572
unique     2                    5169
top      ham  Sorry, I'll call later
freq    4825                      30
-------------------- INFO --------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   label    5572 non-null   object
 1   message  5572 non-null   object
dtypes: object(2)
memory usage: 87.2+ KB
None
"""

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# 加载过数据集后，需要对数据集进行预处理
# 依赖于 nltk 库，用于标记化、停用词删除和词干提取等任务

import nltk

# Download the necessary NLTK data files
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

print("=== BEFORE ANY PREPROCESSING ===") 
print(df.head(5))

# 文本小写化， 有效降维 并提高一致性
# Convert all message text to lowercase
df["message"] = df["message"].str.lower()
print("\n=== AFTER LOWERCASED ===")
print(df["message"].head(5))

# 删除除小写字母、空格、美元符号和感叹号之外的所有字符
# Remove non-essential punctuation and numbers, keep useful symbols like $ and !
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\n=== AFTER REMOVING PUNCTUATION & NUMBERS (except $ and !) ===")
print(df["message"].head(5))

# tokenizing the text

from nltk.tokenize import word_tokenize

# Split each message into individual tokens
# 数据集包含以单词列表表示的消息，可以进行进一步细化文本数据的附加预处理步骤
df["message"] = df["message"].apply(word_tokenize)
print("\n=== AFTER TOKENIZATION ===")
print(df["message"].head(5))

"""
=== AFTER TOKENIZATION ===
0    [go, until, jurong, point, crazy, available, o...
1                       [ok, lar, joking, wif, u, oni]
2    [free, entry, in, a, wkly, comp, to, win, fa, ...
3    [u, dun, say, so, early, hor, u, c, already, t...
4    [nah, i, dont, think, he, goes, to, usf, he, l...
"""
# Stop words 是像 and 、 the 或 is 这样的常见词，它们通常不会提供有意义的上下文信息。
# 删除它们可以减少噪音，并使模型专注于最有可能区分垃圾邮件和正常邮件的词语。通过减少无信息量的词条数量，可以帮助模型更高效地学习。

from nltk.corpus import stopwords

# Define a set of English stop words and remove them from the tokens
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n=== AFTER REMOVING STOP WORDS ===")
print(df["message"].head(5))

# 词干提取，Stemming 通过将单词简化为基本形式（例如， running 变为 run ）来规范化单词
# 这整合了同一词根的不同形式，有效地缩减了词汇量并平滑了文本表示
# 词干提取后，标记列表集中于词根形式，进一步简化文本并提高模型的泛化能力
from nltk.stem import PorterStemmer

# Stem each token to reduce words to their base form
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\n=== AFTER STEMMING ===")
print(df["message"].head(5))

# 将标记重新连接成单个字符串
# 虽然标记对于操作很有用，但许多机器学习算法和矢量化技术（例如 TF-IDF）最适合处理原始文本字符串。
# 将标记重新连接成以空格分隔的字符串可以恢复与这些方法兼容的格式，从而使数据集能够无缝进入特征提取阶段。

# Rejoin tokens into a single string for feature extraction
df["message"] = df["message"].apply(lambda x: " ".join(x))
print("\n=== AFTER JOINING TOKENS BACK INTO STRINGS ===")
print(df["message"].head(5))

# 至此，数据集预处理完成。
# 每条消息都是经过清理和规范化的字符串，可以进行向量化和后续的模型训练，最终提升分类器的性能。