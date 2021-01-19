import pandas as pd

result_df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\训练结果_MultinomialNB.xlsx', sheet_name='Sheet1')


def socre_depend_on_classification(classification_result):
    if classification_result == '其他':
        score = 0
    elif classification_result == '催收短信&信用卡套现':
        score = 50
    elif classification_result == '还款提示':
        score = 2
    elif classification_result == '扣还款成功&贷款信用卡分期申请成功':
        score = -20
    elif classification_result == '扣还款失败&贷款信用卡分期申请失败':
        score = 10
    elif classification_result == '交易支取短信':
        score = -1
    elif classification_result == '贷款信用卡广告':
        score = 1
    return score


def order_number_str(order_number):
    return '\'' + str(order_number)


result_df['socre_depend_on_classification'] = result_df.apply(
    lambda x: socre_depend_on_classification(x['classification_result']), axis=1)
result_df['order_num'] = result_df.apply(lambda x: order_number_str(x['order_number']), axis=1)

score_df = result_df['socre_depend_on_classification'].groupby([result_df['order_num']]).sum()
score_df.reset_index()
score_df.to_excel(r'C:\Users\jizeyuan\Desktop\短信分词\短信评分.xlsx')
