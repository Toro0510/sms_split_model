import pandas as pd
import word_score

result_df = pd.read_excel(r'C:\Users\jizeyuan\Desktop\短信分词\训练结果.xlsx', sheet_name='Sheet1')


def socre_depend_on_word(seg_list):
    temp = str(seg_list).split('|')
    temp_2 = []
    #print(temp)
    for i in temp:
        if i in word_score.word_score:
            temp_2.append(word_score.word_score[i])
    #print(temp_2)
    score = sum(temp_2)
    return score


def order_number_str(order_number):
    return '\'' + str(order_number)


result_df['socre_depend_on_word'] = result_df.apply(lambda x: socre_depend_on_word(x['seg_list']), axis=1)
result_df['order_num'] = result_df.apply(lambda x: order_number_str(x['order_number']), axis=1)

score_df = result_df['socre_depend_on_word'].groupby([result_df['order_num']]).sum()
score_df.reset_index()
score_df.to_excel(r'C:\Users\jizeyuan\Desktop\短信分词\方案B_socre.xlsx')
