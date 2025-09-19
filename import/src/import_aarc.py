from pyprojroot.here import here
import pandas as pd
import scieco

productivities = pd.read_csv(here("import/input/annual_productivity.csv"))
all_years = pd.DataFrame({"Year": range(2007, 2024)})

expanded = productivities[["PersonId"]].drop_duplicates().merge(all_years, how="cross")

df = expanded.merge(productivities, on=["PersonId", "Year"], how="left").sort_values(
    ["PersonId", "Year"]
)
df["NumberPublications"] = df.NumberPublications.fillna(0)

print(
    f"After filling in years with 0 publications, the publication table has {len(df)} rows"
)


df_aa = scieco.get_dataframe("aa-v3")
print(f"Loaded the AARC data: {len(df_aa)} people-year rows")

first_years = (
    df_aa[(df_aa.Rank == "Assistant Professor") & (df_aa.PrimaryAppointment)]
    .groupby(["PersonId"])
    .agg(
        {
            "Year": "min",
            "Field": "first",
        }
    )
    .reset_index()
)

print(
    f"Picking out the first years that assistant professors were appointed, we have {len(first_years)} individuals"
)

first_date = first_years.Year.min()
people_with_known_start_dates = first_years[first_years.Year > first_date].rename(
    columns={"Year": "StartYear"}
)
print(
    f"Filtering for people who started after {first_date} so we know their exact start year, we have {len(people_with_known_start_dates)} people"
)

df_pubs_with_startyears = pd.merge(df, people_with_known_start_dates, on="PersonId")
df_pubs_with_startyears = df_pubs_with_startyears[
    (df_pubs_with_startyears.Year >= df_pubs_with_startyears.StartYear)
]
df_pubs_with_startyears["CareerAge"] = (
    df_pubs_with_startyears.Year - df_pubs_with_startyears.StartYear
)
df_pubs_with_startyears["HasPublications"] = (
    df_pubs_with_startyears.NumberPublications > 0
)
print(
    f"After merging with annual productivity, we now have {len(df_pubs_with_startyears)} rows of people-years."
)

summaries_for_people = (
    df_pubs_with_startyears.groupby("PersonId")
    .agg(
        {
            "HasPublications": "sum",
            "Year": "count",
            "NumberPublications": "sum",
            "Field": "first",
        }
    )
    .rename(
        columns={
            "HasPublications": "YearsWithPubs",
            "Year": "NumberYears",
            "NumberPublications": "TotalPublications",
        }
    )
    .reset_index()
)

summaries_for_people["FractionsYearsWithPub"] = (
    summaries_for_people.YearsWithPubs / summaries_for_people.NumberYears
)
has_no_pubs = set(
    summaries_for_people[summaries_for_people.YearsWithPubs == 0].PersonId
)
df_pubs_with_startyears_nonempty = (
    df_pubs_with_startyears[~df_pubs_with_startyears.PersonId.isin(has_no_pubs)]
    .sort_values(by=["PersonId", "Year"])
    .rename(columns={"NumberPublications": "pubs_adj", "CareerAge": "CareerAgeZero"})
)  # rename for compatibility with fit/src/fit_data.py; a hack because we don't actually adjust the publication counts

# df_pubs_with_startyears_nonempty["pubs_adj"] = (
#    df_pubs_with_startyears_nonempty.pubs_adj.fillna(0)
# )

df_pubs_with_startyears_nonempty["pubs_adj_next"] = (
    df_pubs_with_startyears_nonempty.groupby(["PersonId"]).pubs_adj.shift(periods=-1)
)
df_pubs_with_startyears_nonempty["q_adj_delta"] = (
    df_pubs_with_startyears_nonempty.pubs_adj_next
    - df_pubs_with_startyears_nonempty.pubs_adj
)

df_pubs_with_startyears_nonempty = df_pubs_with_startyears_nonempty.dropna(
    subset=["pubs_adj", "pubs_adj_next", "q_adj_delta"]
)

print(
    f"Dropping people who have no publications, we are down to {len(df_pubs_with_startyears_nonempty)} rows."
)

df_pubs_with_startyears_nonempty_pre_2016 = df_pubs_with_startyears_nonempty[
    (df_pubs_with_startyears_nonempty.StartYear <= 2016)
]
print(
    f"If we create a more cohort-specific dataset with people who first appeared between {first_date} and 2016, we are down to {len(df_pubs_with_startyears_nonempty_pre_2016)} rows."
)

print(
    f"Writing out the datasets to {here('import/output/aarc_trajectories.csv')} and {here('import/output/aarc_trajectories_cohort.csv')}"
)


df_pubs_with_startyears_nonempty.to_csv(
    here("import/output/aarc_trajectories.csv"), index=False
)
df_pubs_with_startyears_nonempty_pre_2016.to_csv(
    here("import/output/aarc_cohort_trajectories.csv"), index=False
)

fields = [
    "Mechanical Engineering",
    "Chemistry",
    "Physics",
    "Economics",
    "Political Science",
    "Management",
    "Sociology",
    "Electrical Engineering",
    "Mathematics",
    "Biological Sciences",
    "History",
    "Psychology",
    "Computer Science",
]

# Write out subsetted dataframes for each field


def clean_field_name(field):
    return field.replace(" ", "_").replace("/", "_").replace("&", "and").lower()


for field in fields:
    df_field = df_pubs_with_startyears_nonempty[
        df_pubs_with_startyears_nonempty.Field == field
    ]
    df_field_cohort = df_pubs_with_startyears_nonempty_pre_2016[
        df_pubs_with_startyears_nonempty_pre_2016.Field == field
    ]

    field_clean = clean_field_name(field)

    print(
        f"For field {field}, we have {len(df_field)} rows, and {len(df_field_cohort)} in the cohort-restricted dataset."
    )
    df_field.to_csv(
        here(f"import/output/aarc_trajectories_{field_clean}.csv"), index=False
    )
    df_field_cohort.to_csv(
        here(f"import/output/aarc_cohort_trajectories_{field_clean}.csv"), index=False
    )
