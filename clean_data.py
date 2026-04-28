import pandas as pd
import numpy as np

# Load datasets
polity_df = pd.read_csv('data/polity.csv')
gdp_df = pd.read_csv('data/gdp.csv')
conflict_df = pd.read_csv('data/conflicts.csv')

# Rename columns for consistency
polity_df = polity_df.rename(columns={'Democracy': 'polity_score'})
gdp_df = gdp_df.rename(columns={'GDP per capita': 'gdp_pc'})
conflict_df = conflict_df.rename(columns={'Deaths in armed conflicts based on where they occurred': 'conflict_deaths'})

# Standardize Entity names (briefly mapping known mismatches if needed)
# For now, we'll use 'Code' (ISO) as it's more reliable than names
# USSR Code in Polity is OWID_USS, in GDP it might be different. Let's check.

# Merge datasets on Code and Year
merged_df = pd.merge(gdp_df, polity_df, on=['Code', 'Year', 'Entity'], how='outer')
merged_df = pd.merge(merged_df, conflict_df, on=['Code', 'Year', 'Entity'], how='outer')

# Fill missing values for conflict deaths with 0 (assuming no record means no major conflict in that year/place)
merged_df['conflict_deaths'] = merged_df['conflict_deaths'].fillna(0)

# Calculate GDP growth rate
merged_df = merged_df.sort_values(['Entity', 'Year'])
merged_df['gdp_growth'] = merged_df.groupby('Entity')['gdp_pc'].pct_change()

# Define Indicators
# Economic Collapse: GDP drop > 5%
merged_df['econ_collapse'] = (merged_df['gdp_growth'] < -0.05).astype(int)

# Dictatorship Rise: Polity score <= -6 (Common threshold for autocracy)
merged_df['is_dictatorship'] = (merged_df['polity_score'] <= -6).astype(int)

# War Outbreak: Conflict deaths > threshold (e.g., 1000 for major war, but let's use > 25 as per UCDP minor conflict)
merged_df['is_war'] = (merged_df['conflict_deaths'] > 25).astype(int)

# Drop rows with too many missing values in key predictors for model training later,
# but keep them for historical exploration if needed.
# For now, let's just save the merged set.

# Targeted verification
target_entities = ['Sudan', 'Libya', 'Rwanda', 'Zimbabwe', 'Germany', 'Italy', 'USSR']
available_targets = merged_df[merged_df['Entity'].isin(target_entities)]

print("Available target entities in merged data:")
print(available_targets['Entity'].unique())

# Save cleaned data
merged_df.to_csv('data/cleaned_data.csv', index=False)
print("Cleaned data saved to data/cleaned_data.csv")
