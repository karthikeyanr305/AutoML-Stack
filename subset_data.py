import pandas as pd

# Assuming df is your DataFrame
# Load your dataset into df
df_train = pd.read_csv('./data/credit_card_transactions/fraudTrain.csv')
df_test = pd.read_csv('./data/credit_card_transactions/fraudTest.csv')
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

fraud_ratio = df[df['is_fraud'] == 1].shape[0]/df.shape[0]
non_fraud_ratio = df[df['is_fraud'] == 0].shape[0]/df.shape[0]

print(fraud_ratio)
print(non_fraud_ratio)

# Subset the DataFrame
fraud = df[df['is_fraud'] == 1].sample(n=int(round(fraud_ratio, 5)*10000), random_state=1)  # 1000 rows where isFraud is 1
non_fraud = df[df['is_fraud'] == 0].sample(n=int(round(non_fraud_ratio, 5)*10000), random_state=1)  # 100000 rows where isFraud is 0

# Concatenate the two subsets
final_df = pd.concat([fraud, non_fraud])

# Shuffle the final dataframe if needed
#final_df = final_df.sample(frac=1, random_state=1).reset_index(drop=True)


final_df.to_csv('fraud_mini.csv')
# Now final_df contains the desired subset
