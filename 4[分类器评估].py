from sklearn.metrics import classification_report
import pandas as pd

# labels：分类报告中显示的类标签的索引列表
# target_names：显示与labels对应的名称
# digits：指定输出格式的精确度
# 精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
# 召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
# F1 = 2*精度*召回率/(精度+召回率)

# result_df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\训练结果.xlsx', sheet_name='Sheet1')
result_df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\训练结果_MultinomialNB.xlsx', sheet_name='Sheet1')
# result_df.info()


print(classification_report(result_df['target'], result_df['classification_result'], digits=4))
