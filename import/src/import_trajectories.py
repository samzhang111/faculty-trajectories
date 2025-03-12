import pandas as pd

df_pubs = pd.read_csv("input/adjusted_productivity.csv")

# Create a unique identifier (this is sufficient)
df_pubs['dblp_id'] = df_pubs['dblp'] + df_pubs['phd_year'].astype(str)
df_pubs = df_pubs.sort_values(by=['dblp_id', 'year'])

# Create some useful variables for downstream
df_pubs['pubs_adj_round'] = df_pubs.pubs_adj.round()
df_pubs['pubs_adj_next'] = df_pubs.groupby(['dblp_id']).pubs_adj.shift(periods=-1)
df_pubs['pubs_adj_next2'] = df_pubs.groupby(['dblp_id']).pubs_adj.shift(periods=-2)
df_pubs['q_adj_delta'] = df_pubs.pubs_adj_next - df_pubs.pubs_adj
df_pubs['q_adj_next_delta'] = df_pubs.pubs_adj_next2 - df_pubs.pubs_adj_next
df_pubs['cumpubs'] = df_pubs.groupby(['dblp_id']).pubs_adj.cumsum()
df_pubs['YearSinceDegree'] = df_pubs['year'] - df_pubs['phd_year']

# This drops the final years of people's careers, where there is no delta
df_pubs = df_pubs.dropna(subset=['q_adj_delta'])

# Restrict to people who have at least 3 pubs by career age 5
# (as described in the paper)
df_pubs_minimally_productive = df_pubs[~df_pubs.dblp_id.isin(
    set(df_pubs[(df_pubs.CareerAge == 5) & (df_pubs.cumpubs < 3)].dblp_id)
)]

# Also, only keep people with degrees at/after 1980
# and drop data before their first professorship or
# after 21 years of career age
df_prod_adj = df_pubs_minimally_productive[
    (df_pubs_minimally_productive.CareerAge <= 21) &
    (df_pubs_minimally_productive.phd_year >= 1980) &
    (df_pubs_minimally_productive.CareerAge > 0)
].sort_values(by=['dblp_id', 'year']) # restrict to 20 years after degree

# Zero-indexed version of variable
df_prod_adj['CareerAgeZero'] = df_prod_adj['CareerAge'] - 1

# The "full" trajectories as described in the paper
df_prod_survivors = df_prod_adj[df_prod_adj.groupby('dblp_id').dblp.transform('count') == 21]

df_prod_adj.to_csv("output/all-trajectories.csv", index=False)
df_prod_survivors.to_csv("output/full-trajectories.csv", index=False)
