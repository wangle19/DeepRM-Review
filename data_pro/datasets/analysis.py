
import pandas as pd
# data = pd.read_csv('Beijing.csv')
# print(data.shape)
# print(data.columns)

# data = data.dropna(axis=0, how='any')
# data.columns = ['g', 'd', 'r', 'attraction_city','user_id', 'item_id', 'rating', 'reviews', 'review_time', 'helpful', 'travel_month','travel_year', 'travel_type']
# print(data.shape)
# data = data[['user_id', 'item_id', 'rating', 'reviews', 'review_time', 'helpful', 'travel_month','travel_year', 'travel_type']]
# data.drop_duplicates(subset=['user_id', 'item_id'],keep='first',inplace=True)
# print(data.shape)
# data_attraction_city_list = data['item_id'].unique()
# # print(attraction_city_list)
#
# df = pd.read_csv('TripAdvisor_simple.csv')
# print(df.shape)
# print(df.columns)
# #
# df = df.dropna(axis=0, how='any')
# df.columns = ['user_id', 'item_id', 'rating', 'reviews', 'review_time', 'helpful', 'travel_month','travel_year', 'travel_type']
# print(df.shape)
#
# df_beijing = df[df['item_id'].isin(data_attraction_city_list)]
# print(df_beijing)
# df_beijing_attraction_list = df_beijing['item_id'].unique()
# print(df_beijing_attraction_list)
#
# diff_list = list(set(data_attraction_city_list) - set(df_beijing_attraction_list))
# print(diff_list)
# print(len(diff_list))
#
# data_diff_beijing = data[data['item_id'].isin(diff_list)]
# print(data_diff_beijing.shape)
#
# total_data=pd.concat([df,data_diff_beijing],axis=0)
# print(total_data.shape)
# print(total_data.columns)
# total_data.drop_duplicates(subset=['user_id', 'item_id'],keep='first',inplace=True)
# print(total_data.shape)
# total_data.to_csv('total_simple_diff_beijing.csv',index=False)



#-------------------------------------------------------------------------------------------------
data = pd.read_csv('total.csv')
print(data.shape)
print(data.columns)
data.drop_duplicates(subset=['g','attraction_city'], keep='first',inplace=True)
print(data.shape)
data.to_csv('city.csv',index=False)
