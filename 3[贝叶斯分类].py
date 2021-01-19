import pandas as pd

train_df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\短信分词模型训练.xlsx', sheet_name='Sheet1')
# print(train_df.info())

# 先验概率
probe_target_其他 = len(train_df[(train_df['target'] == '其他')]) / len(train_df)
probe_target_催收短信 = len(train_df[(train_df['target'] == '催收短信&信用卡套现')]) / len(train_df)
probe_target_还款提示 = len(train_df[(train_df['target'] == '还款提示')]) / len(train_df)
probe_target_扣还款成功 = len(train_df[(train_df['target'] == '扣还款成功&贷款信用卡分期申请成功')]) / len(train_df)
probe_target_扣还款失败 = len(train_df[(train_df['target'] == '扣还款失败&贷款信用卡分期申请失败')]) / len(train_df)
probe_target_交易支取短信 = len(train_df[(train_df['target'] == '交易支取短信')]) / len(train_df)
probe_target_贷款信用卡广告 = len(train_df[(train_df['target'] == '贷款信用卡广告')]) / len(train_df)

# 每个词语的概率
word_list = [i for i in train_df.iloc[:, 6:].columns]
word_probe_dict = {}
for i in word_list:
    word_probe_dict[i] = len(train_df[(train_df[i] == 1)]) / len(train_df)

# 每个词语在每个类别中的概率
word_target_probe_dict_0 = {}
for i in word_list:
    word_target_probe_dict_0[i] = len(train_df[(train_df['target'] == '其他') & (train_df[i] == 1)]) / len(
        train_df[(train_df['target'] == '其他')])

word_target_probe_dict_1 = {}
for i in word_list:
    word_target_probe_dict_1[i] = len(train_df[(train_df['target'] == '催收短信&信用卡套现') & (train_df[i] == 1)]) / len(
        train_df[(train_df['target'] == '催收短信&信用卡套现')])

word_target_probe_dict_2 = {}
for i in word_list:
    word_target_probe_dict_2[i] = len(train_df[(train_df['target'] == '还款提示') & (train_df[i] == 1)]) / len(
        train_df[(train_df['target'] == '还款提示')])

word_target_probe_dict_3 = {}
for i in word_list:
    word_target_probe_dict_3[i] = len(train_df[(train_df['target'] == '扣还款成功&贷款信用卡分期申请成功') & (train_df[i] == 1)]) / len(
        train_df[(train_df['target'] == '扣还款成功&贷款信用卡分期申请成功')])

word_target_probe_dict_4 = {}
for i in word_list:
    word_target_probe_dict_4[i] = len(train_df[(train_df['target'] == '扣还款失败&贷款信用卡分期申请失败') & (train_df[i] == 1)]) / len(
        train_df[(train_df['target'] == '扣还款失败&贷款信用卡分期申请失败')])

word_target_probe_dict_5 = {}
for i in word_list:
    word_target_probe_dict_5[i] = len(train_df[(train_df['target'] == '交易支取短信') & (train_df[i] == 1)]) / len(
        train_df[(train_df['target'] == '交易支取短信')])

word_target_probe_dict_6 = {}
for i in word_list:
    word_target_probe_dict_6[i] = len(train_df[(train_df['target'] == '贷款信用卡广告') & (train_df[i] == 1)]) / len(
        train_df[(train_df['target'] == '贷款信用卡广告')])


# 计算训练集每行每个分类对应概率
def mult_list(list):
    mult = 1
    for i in list:
        mult = mult * i
    return mult


def BY_classification(row):
    temp_分子_0 = []
    temp_分子_1 = []
    temp_分子_2 = []
    temp_分子_3 = []
    temp_分子_4 = []
    temp_分子_5 = []
    temp_分子_6 = []
    temp_分母 = []
    result_dict = {}

    for i in word_list:
        if row[i] == 1:
            temp_分子_0.append(word_target_probe_dict_0[i])
            temp_分子_1.append(word_target_probe_dict_1[i])
            temp_分子_2.append(word_target_probe_dict_2[i])
            temp_分子_3.append(word_target_probe_dict_3[i])
            temp_分子_4.append(word_target_probe_dict_4[i])
            temp_分子_5.append(word_target_probe_dict_5[i])
            temp_分子_6.append(word_target_probe_dict_6[i])
            temp_分母.append(word_probe_dict[i])

    result_dict['其他'] = (mult_list(temp_分子_0) * probe_target_其他) / mult_list(temp_分母)
    result_dict['催收短信&信用卡套现'] = (mult_list(temp_分子_1) * probe_target_催收短信) / mult_list(temp_分母)
    result_dict['还款提示'] = (mult_list(temp_分子_2) * probe_target_还款提示) / mult_list(temp_分母)
    result_dict['扣还款成功&贷款信用卡分期申请成功'] = (mult_list(temp_分子_3) * probe_target_扣还款成功) / mult_list(temp_分母)
    result_dict['扣还款失败&贷款信用卡分期申请失败'] = (mult_list(temp_分子_4) * probe_target_扣还款失败) / mult_list(temp_分母)
    result_dict['交易支取短信'] = (mult_list(temp_分子_5) * probe_target_交易支取短信) / mult_list(temp_分母)
    result_dict['贷款信用卡广告'] = (mult_list(temp_分子_6) * probe_target_贷款信用卡广告) / mult_list(temp_分母)

    return max(result_dict, key=result_dict.get)


def order_number_str(order_number):
    return '\'' + str(order_number)


train_df['order_num'] = train_df.apply(lambda x: order_number_str(x['order_number']), axis=1)
train_df['classification_result'] = train_df.apply(lambda x: BY_classification(x), axis=1)
train_df.to_excel(r'C:\Users\jizeyuan\Desktop\短信分词\训练结果.xlsx', index=False)
