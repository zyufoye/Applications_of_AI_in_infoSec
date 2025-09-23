# 使用 CountVectorizer 进行词袋方法的特征提取
# scikit-learn 库中的 CountVectorizer 可以高效地实现词袋模型


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# min_df=1 ：一个词条必须至少出现在一篇文档中才能被收录。
# max_df=0.9 ：排除在超过 90% 的文档中出现的术语，删除那些过于常见、区分度有限的词语。
# ngram_range=(1, 2) ：特征矩阵通过包含单元和二元来捕获单个单词和常用单词对，从而可能提高模型检测垃圾邮件模式的能力。

df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)
df["message"] = df["message"].str.lower()
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
df["message"] = df["message"].apply(word_tokenize)
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
df["message"] = df["message"].apply(lambda x: " ".join(x))
print(df["message"].head(5))

# Initialize CountVectorizer with bigrams, min_df, and max_df to focus on relevant terms
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))

# Fit and transform the message column
X = vectorizer.fit_transform(df["message"])

# Labels (target variable)
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)  # Converting labels to 1 and 0

#  到这里，X 就变成了一个数值特征矩阵，可以输入到分类器中，例如朴素贝叶斯。
# print("Feature matrix shape:", X.shape)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Build the pipeline by combining vectorization and classification
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
}

# Perform the grid search with 5-fold cross-validation and the F1-score as metric
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

# Fit the grid search on the full dataset
grid_search.fit(df["message"], y)

# Extract the best model identified by the grid search
best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_)

# Example SMS messages for evaluation
new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]

import numpy as np
import re

# Preprocess function that mirrors the training-time preprocessing
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Preprocess and vectorize messages
processed_messages = [preprocess_message(msg) for msg in new_messages]

# 对已处理的消息进行矢量化
# Transform preprocessed messages into feature vectors 特征提取
X_new = best_model.named_steps["vectorizer"].transform(processed_messages)

# 对已处理的消息进行预测
# Predict with the trained classifier
predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)

# 对每个测试结果显示预测结果和概率
# Display predictions and probabilities for each evaluated message
for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
    spam_probability = prediction_probabilities[i][1]  # Probability of being spam
    ham_probability = prediction_probabilities[i][0]   # Probability of being not spam
    
    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Not-Spam Probability: {ham_probability:.2f}")
    print("-" * 50)

# 模型保存
import joblib

# Save the trained model to a file for future use
model_filename = 'spam_detection_model.joblib'
joblib.dump(best_model, model_filename)

print(f"Model saved to {model_filename}")

#  模型加载
# loaded_model = joblib.load(model_filename)
# predictions = loaded_model.predict(new_messages)