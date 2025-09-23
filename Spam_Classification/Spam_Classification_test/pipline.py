from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# 定义一个 Pipeline
pipeline = Pipeline([
    ("vect", CountVectorizer()),        # 第一步：把文本转成词频向量
    ("tfidf", TfidfTransformer()),      # 第二步：用 TF-IDF 权重化
    ("classifier", MultinomialNB())     # 第三步：训练分类器（朴素贝叶斯）
])

# 假设我们有数据
X = ["I love Python", "Python is great", "I dislike bugs", "bugs are bad","python bugs are bad"]
y = [1, 1, 0, 0, 0]   # 1=正样本, 0=负样本

# 直接训练 Pipeline（内部会按顺序执行）
pipeline.fit(X, y)

# 预测
print(pipeline.predict(["Python bugs are annoying"]))

# 结合 GridSearchCV 进行超参数调优
from sklearn.model_selection import GridSearchCV

# 参数网格：调 classifier 的 alpha
param_grid = {"classifier__alpha": [0.1, 0.15, 0.25, 0.5, 1.0]}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1"
)

grid_search.fit(X, y)

print("最佳参数:", grid_search.best_params_)
print("最佳模型:", grid_search.best_estimator_)
