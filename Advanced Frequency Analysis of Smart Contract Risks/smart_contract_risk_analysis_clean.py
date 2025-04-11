#!/usr/bin/env python


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 




drive.mount('/content/drive')





uploaded = files.upload()






import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')






data_set = pd.read_excel("risk_data.xlsx")


data_set.head(10)






data_set.info()
data_set.describe()




data_set = data_set.dropna()  # or data_set.fillna(value)

data_set = data_set.drop_duplicates()






data_set['incorrect_inheritance_order'].value_counts()




risk_columns = ['Is_closed_source', 'hidden_owner', 'anti_whale_modifiable',
       'Is_anti_whale', 'Is_honeypot', 'buy_tax', 'sell_tax',
       'slippage_modifiable', 'Is_blacklisted', 'can_take_back_ownership',
       'owner_change_balance', 'is_airdrop_scam', 'selfdestruct', 'trust_list',
       'is_whitelisted', 'is_fake_token', 'illegal_unicode', 'exploitation',
       'bad_contract', 'reusing_state_variable', 'encode_packed_collision',
       'encode_packed_parameters', 'centralized_risk_medium',
       'centralized_risk_high', 'centralized_risk_low', 'event_setter',
       'external_dependencies', 'immutable_states',
       'reentrancy_without_eth_transfer', 'incorrect_inheritance_order',
       'shadowing_local', 'events_maths']





frequencies = data_set[risk_columns].apply(lambda x: x.value_counts()).loc[True]

frequencies = frequencies.fillna(0)
frequencies





sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.barplot(x=frequencies.index, y=frequencies.values, palette='viridis')
plt.title('Frequency of True Values for Each Risk Tag')
plt.xlabel('Risk Tags')
plt.ylabel('Frequency of True')
plt.xticks(rotation=45)
plt.show()





frequencies = data_set[risk_columns].apply(lambda x: x.value_counts()).loc[False]

frequencies = frequencies.fillna(0)
frequencies





sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.barplot(x=frequencies.index, y=frequencies.values, palette='viridis')
plt.title('Frequency of True Values for Each Risk Tag')
plt.xlabel('Risk Tags')
plt.ylabel('Frequency of False')
plt.xticks(rotation=45)
plt.show()


# 



get_ipython().system('pip install scipy')
import scipy.stats # Import the scipy.stats module

def phi_coefficient(x, y):
    """Calculate the Phi coefficient for two binary variables."""
    contingency_table = pd.crosstab(x, y)
    chi2 = scipy.stats.chi2_contingency(contingency_table, correction=False)[0]
    n = np.sum(np.sum(contingency_table))
    phi = np.sqrt(chi2 / n)
    return phi

phi = phi_coefficient(data_set['is_airdrop_scam'], data_set['hidden_owner'])
print(f"Phi Coefficient between 'is_airdrop_scam' and 'hidden_owner': {phi}")






def phi_coefficient(x, y):
    """Calculate the Phi coefficient for two binary variables."""
    contingency_table = pd.crosstab(x, y)
    chi2 = scipy.stats.chi2_contingency(contingency_table, correction=False)[0]
    n = np.sum(np.sum(contingency_table))
    phi = np.sqrt(chi2 / n)
    return phi

phi = phi_coefficient(data_set['centralized_risk_high'], data_set['centralized_risk_medium'])
print(f"Phi Coefficient between 'centralized_risk_high' and 'centralized_risk_medium': {phi}")





risk_columns = ['Is_closed_source', 'hidden_owner', 'anti_whale_modifiable',
       'Is_anti_whale', 'Is_honeypot', 'buy_tax', 'sell_tax',
       'slippage_modifiable', 'Is_blacklisted', 'can_take_back_ownership',
       'owner_change_balance', 'is_airdrop_scam', 'selfdestruct', 'trust_list',
       'is_whitelisted', 'is_fake_token', 'illegal_unicode', 'exploitation',
       'bad_contract', 'reusing_state_variable', 'encode_packed_collision',
       'encode_packed_parameters', 'centralized_risk_medium',
       'centralized_risk_high', 'centralized_risk_low', 'event_setter',
       'external_dependencies', 'immutable_states',
       'reentrancy_without_eth_transfer', 'incorrect_inheritance_order',
       'shadowing_local', 'events_maths']




risk_data = data_set[risk_columns]

phi_matrix = pd.DataFrame(index=risk_data.columns, columns=risk_data.columns)

for var1 in risk_data.columns:
    for var2 in risk_data.columns:
        phi_matrix.loc[var1, var2] = phi_coefficient(risk_data[var1], risk_data[var2])

print("Phi coefficients calculated for all pairs of variables:")
phi_matrix


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 




plt.figure(figsize=(12, 10))

sns.heatmap(phi_matrix.astype(float), annot=False, fmt=".2f", cmap='binary', vmin=-1, vmax=1)
plt.title('Heatmap of Phi Coefficients Between Risk Tags')
plt.show()




phi = phi_coefficient(data_set['buy_tax'], data_set['sell_tax'])
print(f"Phi Coefficient between 'buy_tax' and 'sell_tax': {phi}")


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 