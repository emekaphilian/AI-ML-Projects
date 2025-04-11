# Smart Contract Risk Analysis Script
# Converted from Jupyter Notebook

# --- Code Cell ---
from google.colab import drive
drive.mount('/content/drive')


# --- Code Cell ---
from google.colab import files
uploaded = files.upload()


# --- Code Cell ---
# importing the necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Display plots inline
%matplotlib inline


# --- Code Cell ---
# Uploading the dataset

data_set = pd.read_excel("risk_data.xlsx")

# Display the first 10 rows

data_set.head(10)

# --- Code Cell ---
# Performing EDA

data_set.info()
data_set.describe()


# --- Code Cell ---
# Data Cleaning
# Handle missing values
data_set = data_set.dropna()  # or data_set.fillna(value)

# Check for duplicates
data_set = data_set.drop_duplicates()

# --- Code Cell ---
# Let's now look at the value counts of an individual risk tag: for example incorrect_inheritance_order

data_set['incorrect_inheritance_order'].value_counts()

# --- Code Cell ---
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

# --- Code Cell ---
# Calculating the frequency of 'TRUE' in each risk tag column

frequencies = data_set[risk_columns].apply(lambda x: x.value_counts()).loc[True]

# Replace NaN with 0 for any column that may not have True values
frequencies = frequencies.fillna(0)
frequencies

# --- Code Cell ---
# Visualizing the frequencies of "TRUE" using a bar chart

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.barplot(x=frequencies.index, y=frequencies.values, palette='viridis')
plt.title('Frequency of True Values for Each Risk Tag')
plt.xlabel('Risk Tags')
plt.ylabel('Frequency of True')
plt.xticks(rotation=45)
plt.show()

# --- Code Cell ---
# Calculating the frequency of 'FALSE' in each risk tag column

frequencies = data_set[risk_columns].apply(lambda x: x.value_counts()).loc[False]

# Replace NaN with 0 for any column that may not have True values
frequencies = frequencies.fillna(0)
frequencies

# --- Code Cell ---
# Visualizing the frequencies of "FALSE" using a bar chart

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.barplot(x=frequencies.index, y=frequencies.values, palette='viridis')
plt.title('Frequency of True Values for Each Risk Tag')
plt.xlabel('Risk Tags')
plt.ylabel('Frequency of False')
plt.xticks(rotation=45)
plt.show()

# --- Code Cell ---
# To calculate the Phi coefficient, which is suitable for pairs of binary variables,
# we first need to establish a function that can handle this calculation:
!pip install scipy
import scipy.stats # Import the scipy.stats module

def phi_coefficient(x, y):
    """Calculate the Phi coefficient for two binary variables."""
    # Create a contingency table
    contingency_table = pd.crosstab(x, y)
    # Calculate the phi coefficient from the contingency table
    chi2 = scipy.stats.chi2_contingency(contingency_table, correction=False)[0]
    n = np.sum(np.sum(contingency_table))
    phi = np.sqrt(chi2 / n)
    return phi

# Example calculation between two risk tags
phi = phi_coefficient(data_set['is_airdrop_scam'], data_set['hidden_owner'])
print(f"Phi Coefficient between 'is_airdrop_scam' and 'hidden_owner': {phi}")

# --- Code Cell ---
# To calculate the Phi coefficient, which is suitable for pairs of binary variables,
# we first need to establish a function that can handle this calculation:

def phi_coefficient(x, y):
    """Calculate the Phi coefficient for two binary variables."""
    # Create a contingency table
    contingency_table = pd.crosstab(x, y)
    # Calculate the phi coefficient from the contingency table
    chi2 = scipy.stats.chi2_contingency(contingency_table, correction=False)[0]
    n = np.sum(np.sum(contingency_table))
    phi = np.sqrt(chi2 / n)
    return phi

# Example calculation between two risk tags
phi = phi_coefficient(data_set['centralized_risk_high'], data_set['centralized_risk_medium'])
print(f"Phi Coefficient between 'centralized_risk_high' and 'centralized_risk_medium': {phi}")

# --- Code Cell ---
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

# --- Code Cell ---
risk_data = data_set[risk_columns]

# Create a DataFrame to store Phi coefficients
phi_matrix = pd.DataFrame(index=risk_data.columns, columns=risk_data.columns)

# Calculate Phi coefficient for each pair of binary variables
for var1 in risk_data.columns:
    for var2 in risk_data.columns:
        phi_matrix.loc[var1, var2] = phi_coefficient(risk_data[var1], risk_data[var2])

print("Phi coefficients calculated for all pairs of variables:")
phi_matrix

# --- Code Cell ---
# Setting the size of the plot
plt.figure(figsize=(12, 10))

# Creating a heatmap
sns.heatmap(phi_matrix.astype(float), annot=False, fmt=".2f", cmap='binary', vmin=-1, vmax=1)
plt.title('Heatmap of Phi Coefficients Between Risk Tags')
plt.show()

# --- Code Cell ---
# From the heatmap, there is a high Phi Coefficient value (0.633826) between "buy_tax" and "sell_tax".
# This indicates an extremely strong positive correlation between these two risk tags in the smart contract data set you analyzed.
phi = phi_coefficient(data_set['buy_tax'], data_set['sell_tax'])
print(f"Phi Coefficient between 'buy_tax' and 'sell_tax': {phi}")

