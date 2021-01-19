import pandas as pd
import stop_word
import jieba

jieba.load_userdict('userdict.txt')

def jieba_word_split(sms_message, useful_word_filter):
    seg_list = jieba.cut(sms_message, cut_all=True)
    seg_list_chinese = set([i for i in seg_list if i in useful_word_filter])
    return '|'.join(seg_list_chinese)


def word_regexp(i, x):
    if i in x:
        return 1
    else:
        return 0


if __name__ == '__main__':
    df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\useful_word.xlsx')
    # print(df.info())

    useful_word_filter = []  # 去除STOP_WORD
    for i in df['word']:
        if i not in stop_word.stop_word:
            useful_word_filter.append(i)

    # print(useful_word_filter)

    df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\短信分类.xlsx', sheet_name='短信文本')
    df['seg_list'] = df.apply(lambda x: jieba_word_split(x['body'], useful_word_filter), axis=1)

    for i in useful_word_filter:  # 生成有效哑变量
        df[i] = df['body'].apply(lambda x: word_regexp(i, x))

    df.to_excel(r'C:\Users\jizeyuan\Desktop\短信分词\短信分词模型训练.xlsx', index=False)
