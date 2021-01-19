import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

train_df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\短信分词模型训练.xlsx', sheet_name='Sheet1')
# print(train_df.info())

clf = MultinomialNB()
clf.fit(train_df.iloc[:, 6:], train_df['target'])
train_df['classification_result'] = clf.predict(train_df.iloc[:, 6:])


def order_number_str(order_number):
    return '\'' + str(order_number)


train_df['order_num'] = train_df.apply(lambda x: order_number_str(x['order_number']), axis=1)
train_df.to_excel(r'C:\Users\jizeyuan\Desktop\短信分词\训练结果_MultinomialNB.xlsx', index=False)
joblib.dump(clf, r'C:\Users\jizeyuan\Desktop\短信分词\MultinomialNB.pkl')
# clf = joblib.load('D:\\xxx\\data.pkl')
