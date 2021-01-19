import pandas as pd
import jieba
from collections import Counter
from scipy.stats import chi2_contingency
import stop_word

# jieba.enable_paddle()
jieba.load_userdict('userdict.txt')


def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff' and len(check_str) >= 2:
            return True
    return False


def most_word_split(sms_word, n):
    word_split = [i for i in jieba.cut(sms_word, cut_all=True) if check_contain_chinese(i) == True]
    most_word = Counter(word_split).most_common(n)
    return most_word


def save_most_word(most_word, path):
    with open(path, 'w+') as f:
        for x in most_word:
            f.write('{1},{0}\n'.format(x[1], x[0]))


def load_most_word(path):
    most_word = []
    with open(path, 'r') as f:
        for x in f.readlines():
            x = x.split(',')
            most_word.append(x[0])
    return most_word


def word_regexp(i, x):
    if i in x:
        return 1
    else:
        return 0


def Chi_Square_check(df, p_limit, stop_word):
    userful_word = []
    for i in df.iloc[:, 5:].columns:
        t1 = pd.crosstab(df['target'], df[i])
        g, p, dof, expctd = chi2_contingency(t1)

        rate_0 = round(len(df[(df['target'] == '其他') & (df[i] == 1)]) / len(df[((df[i] == 1))]), 4)
        rate_1 = round(len(df[(df['target'] == '催收短信&信用卡套现') & (df[i] == 1)]) / len(df[(df[i] == 1)]), 4)
        rate_2 = round(len(df[(df['target'] == '还款提示') & (df[i] == 1)]) / len(df[(df['target'] == '还款提示')]), 4)
        rate_3 = round(len(df[(df['target'] == '扣还款成功&贷款信用卡分期申请成功') & (df[i] == 1)]) / len(
            df[(df[i] == 1)]), 4)
        rate_4 = round(len(df[(df['target'] == '扣还款失败&贷款信用卡分期申请失败') & (df[i] == 1)]) / len(
            df[(df[i] == 1)]), 4)
        rate_5 = round(len(df[(df['target'] == '交易支取短信') & (df[i] == 1)]) / len(df[(df[i] == 1)]), 4)
        rate_6 = round(len(df[(df['target'] == '贷款信用卡广告') & (df[i] == 1)]) / len(df[(df[i] == 1)]), 4)

        if p <= p_limit and i not in stop_word:
            userful_word.append([i, p, rate_0, rate_1, rate_2, rate_3, rate_4, rate_5, rate_6])
    return userful_word


if __name__ == '__main__':
    '''
    totally cost 
    '''

    print('load split sample')
    sms_word = open(r'C:\Users\jizeyuan\Desktop\短信分词\sms_message.txt', encoding='utf-8').read()
    print(len(sms_word))

    print('split word')
    most_word = most_word_split(sms_word, 500)
    save_most_word(most_word, r'C:\Users\jizeyuan\Desktop\短信分词\most_word.txt')
    most_word = load_most_word(r'C:\Users\jizeyuan\Desktop\短信分词\most_word.txt')

    print('load train_data')
    df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\短信分类.xlsx', sheet_name='短信文本')
    for i in most_word:  # 生成词频哑变量
        df[i] = df['body'].apply(lambda x: word_regexp(i, x))

    print('Chi_Square check')
    useful_word = Chi_Square_check(df, 0.00001, stop_word.stop_word)

    print('save userful_word')
    useful_word_df = pd.DataFrame(useful_word,
                                  columns=['word', 'P_value', '其他占比', '催收短信&信用卡套现占比', '还款提示占比', '扣还款成功&贷款信用卡分期申请成功占比',
                                           '扣还款失败&贷款信用卡分期申请失败占比',
                                           '交易支取短信占比', '贷款信用卡广告占比'])
    useful_word_df.sort_values(by=['P_value'], ascending=False)

    useful_word_df.to_excel(r'C:\Users\jizeyuan\Desktop\短信分词\useful_word.xlsx', index=False)
    # df.to_excel(r'C:\Users\jizeyuan\Desktop\短信分词\df_var_dummy.xlsx')
