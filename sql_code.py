from sqlalchemy import create_engine
import pandas as pd


conn = create_engine('')

sql_code='''
SELECT
order_number,address,DATE,body
FROM `qxy_base`.`fund_order_user_sms` a
WHERE body!=''
AND body IS NOT NULL
AND LENGTH(body)>=70 #根据编码调整
AND body NOT REGEXP '验证码'
AND body NOT REGEXP '校验码'
AND body NOT REGEXP '激活码'
AND body NOT REGEXP '联通'
AND body NOT REGEXP '电信'
AND body NOT REGEXP '移动'
AND body NOT REGEXP '充值'
AND body NOT REGEXP '话费'
AND body NOT REGEXP '积分'
AND body NOT REGEXP '套餐'
AND body NOT REGEXP '流量'
AND body NOT REGEXP '天猫'
AND body NOT REGEXP '京东'
AND body NOT REGEXP '淘宝'
AND body NOT REGEXP '拼多多'
AND body NOT REGEXP '腾讯'
AND body NOT REGEXP '商城'
AND body NOT REGEXP '快件'
AND body NOT REGEXP 'EMS'
AND body NOT REGEXP '外卖'
AND body NOT REGEXP '送餐'
AND body NOT REGEXP '美团'
AND body NOT REGEXP '饿了么'
AND body NOT REGEXP '58同城'
AND body NOT REGEXP 'BOSS直聘'
AND body NOT REGEXP '滴滴'
AND body NOT REGEXP '包裹'
AND body NOT REGEXP '快递'
AND body NOT REGEXP '12306'
AND body NOT REGEXP '10086'
AND body NOT REGEXP '航班'
AND body NOT REGEXP '旗舰'
AND body NOT REGEXP '面试'
AND body NOT REGEXP '薪资'
AND body NOT REGEXP '消防'
AND body NOT REGEXP '救援'
AND body NOT REGEXP '防空'
AND body NOT REGEXP '国家'
AND body NOT REGEXP '扶贫'
AND body NOT REGEXP '食品'
AND body NOT REGEXP '人口'
AND body NOT REGEXP '教育'
AND body NOT REGEXP '医疗'
AND body NOT REGEXP '扫黑'
AND body NOT REGEXP '旅游'
AND body NOT REGEXP '打车'
AND body NOT REGEXP '旅客'
AND body NOT REGEXP '福利'
AND body NOT REGEXP '恭喜'
AND body NOT REGEXP '红包'
AND body NOT REGEXP '出租'
AND body NOT REGEXP '房源'
AND body NOT REGEXP '链接'
AND body NOT REGEXP '会员'
AND body NOT REGEXP '活动'
AND body NOT REGEXP '优惠'
AND body NOT REGEXP '满意'
AND body NOT REGEXP '元宝'
AND body NOT REGEXP '专营'
AND body NOT REGEXP '温馨'
AND body NOT REGEXP '免费'
AND body NOT REGEXP 'KTV'
AND body NOT REGEXP 'luckin coffee'
AND body NOT REGEXP '抖音'
AND body NOT REGEXP '快手'
AND body NOT REGEXP '探探'
AND body NOT REGEXP '脉脉'
AND body NOT REGEXP '防疫'
AND body NOT REGEXP '疫情'
AND body NOT REGEXP '物业'
AND body NOT REGEXP '公益'
AND body NOT REGEXP '简历'
AND body NOT REGEXP '抢票'
AND body NOT REGEXP '车票'
AND body NOT REGEXP '机票'
AND body NOT REGEXP '火车'
AND body NOT REGEXP '酒店'
AND body NOT REGEXP '旅行'
AND body NOT REGEXP '电力'
AND body NOT REGEXP '交警'
AND body NOT REGEXP '体检'
AND body NOT REGEXP '音乐'
AND body NOT REGEXP '招聘'
AND body NOT REGEXP '好友'
AND body NOT REGEXP '双11'
AND body NOT REGEXP '骑行易'
'''

sms_df=pd.read_sql(sql_code,conn)
print(sms_df)

