
# ============================================================
# 1. File paths and settings
# ============================================================

import pandas as pd
import numpy as np

from common import (
    ROOT,
    RAW,
    CES_DIR,
    ANA,
    SEED,
)

CES_2024_FILE = CES_DIR / "CCES24_Common_OUTPUT_vv_topost_final.csv"
CES_2025_FILE = CES_DIR / "CES25_Common.dta"
ELECTIONS_FILE = RAW / "governor_races_2026.csv"

OUTPUT_DIR = ANA
TARGET_PER_STATE = 100
RANDOM_SEED = 2453

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Load the original datasets
# ============================================================

ces24 = pd.read_csv(CES_2024_FILE)
ces25 = pd.read_stata(CES_2025_FILE)

print("CES 2024 shape:", ces24.shape)
print("CES 2025 shape:", ces25.shape)

# ============================================================
# 3. Define the original columns we need
# ============================================================

profile_columns = [
    "birthyr",
    "gender4",
    "educ",
    "race",
    "hispanic",
    "votereg",
    "pid7",
    "ideo5",
    "faminc_new",
]

identifier_columns = [
    "caseid",
    "inputstate",
    "commonweight",
]

# Election-context columns are retained only if present
# in BOTH datasets.
election_columns = [
    "GovCand1Name",
    "GovCand1Party",
    "GovCand2Name",
    "GovCand2Party",
    "CurrentGovName",
    "CurrentGovParty",
]

required_columns = profile_columns + identifier_columns

for year, df in [(2024, ces24), (2025, ces25)]:
    missing = sorted(set(required_columns) - set(df.columns))

    if missing:
        raise ValueError(
            f"CES {year} is missing required columns: {missing}"
        )

common_election_columns = [
    col for col in election_columns
    if col in ces24.columns and col in ces25.columns
]

print("\nElection-context columns available in both datasets:")
print(common_election_columns)

# ============================================================
# 4. Map CES 2024 state codes to state names
#    Keep the original inputstate column unchanged.
# ============================================================

state_codes = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
}

# Internal helper column; inputstate itself is not renamed.
ces24["_state_name"] = (
    pd.to_numeric(ces24["inputstate"], errors="coerce")
    .map(state_codes)
)

# CES 2025 inputstate already contains readable state names.
ces25["_state_name"] = (
    ces25["inputstate"]
    .astype("string")
    .str.strip()
)

valid_state_names = set(state_codes.values())

# Check for unknown state names in both datasets.
for year, df in [(2024, ces24), (2025, ces25)]:
    unknown = sorted(
        set(df["_state_name"].dropna().unique())
        - valid_state_names
    )

    print(f"\nCES {year} missing state names:",
          df["_state_name"].isna().sum())
    print(f"CES {year} unrecognized state names:", unknown)

    if df["_state_name"].isna().any() or unknown:
        raise ValueError(
            f"Check state mapping for CES {year} before proceeding."
        )

# Standardize inputstate across both survey years.
ces24["inputstate"] = ces24["_state_name"]
ces25["inputstate"] = ces25["_state_name"]


# ============================================================
# 5. CES 2024 response-label mappings
#    CES 2024 CSV stores numeric codes for these variables.
# ============================================================

value_mappings_2024 = {
    "gender4": {
        1: "Man",
        2: "Woman",
        3: "Non-binary",
        4: "Other",
        8: "skipped",
        9: "not asked",
    },
    "educ": {
        1: "No HS",
        2: "High school graduate",
        3: "Some college",
        4: "2-year",
        5: "4-year",
        6: "Post-grad",
        8: "skipped",
        9: "not asked",
    },
    "race": {
        1: "White",
        2: "Black",
        3: "Hispanic",
        4: "Asian",
        5: "Native American",
        6: "Two or more races",
        7: "Other",
        8: "Middle Eastern",
        98: "skipped",
        99: "not asked",
    },
    "hispanic": {
        1: "Yes",
        2: "No",
        8: "skipped",
        9: "not asked",
    },
    "votereg": {
        1: "Yes",
        2: "No",
        3: "Don't know",
        8: "skipped",
        9: "not asked",
    },
    "pid7": {
        1: "Strong Democrat",
        2: "Not very strong Democrat",
        3: "Lean Democrat",
        4: "Independent",
        5: "Lean Republican",
        6: "Not very strong Republican",
        7: "Strong Republican",
        8: "Not sure",
        9: "Don't know",
        98: "skipped",
        99: "not asked",
    },
    "ideo5": {
        1: "Very liberal",
        2: "Liberal",
        3: "Moderate",
        4: "Conservative",
        5: "Very conservative",
        6: "Not sure",
        8: "skipped",
        9: "not asked",
    },
    "faminc_new": {
        1: "Less than $10,000",
        2: "$10,000 - $19,999",
        3: "$20,000 - $29,999",
        4: "$30,000 - $39,999",
        5: "$40,000 - $49,999",
        6: "$50,000 - $59,999",
        7: "$60,000 - $69,999",
        8: "$70,000 - $79,999",
        9: "$80,000 - $99,999",
        10: "$100,000 - $119,999",
        11: "$120,000 - $149,999",
        12: "$150,000 - $199,999",
        13: "$200,000 - $249,999",
        14: "$250,000 - $349,999",
        15: "$350,000 - $499,999",
        16: "$500,000 or more",
        97: "Prefer not to say",
        998: "skipped",
        999: "not asked",
    },
}

# ============================================================
# 6. Check CES 2024 codes, then apply the label mappings
# ============================================================

for col, mapping in value_mappings_2024.items():
    numeric_values = pd.to_numeric(ces24[col], errors="coerce")

    # Non-missing values that cannot be interpreted as numbers.
    nonnumeric = ces24[col].notna() & numeric_values.isna()

    # Numeric codes that are not documented in our mapping.
    unexpected = sorted(
        set(numeric_values.dropna().unique())
        - set(mapping.keys())
    )

    print(
        f"CES 2024 {col}: "
        f"nonnumeric values={int(nonnumeric.sum())}, "
        f"unexpected codes={unexpected}"
    )

    if nonnumeric.any() or unexpected:
        raise ValueError(
            f"Unexpected CES 2024 values in {col}. "
            "Check the codebook before continuing."
        )

    # Apply labels while preserving the original column name.
    ces24[col] = numeric_values.map(mapping)

    # Treat skipped/not-asked responses as missing.
    ces24[col] = ces24[col].replace(
        ["skipped", "not asked"],
        pd.NA,
    )

# ============================================================
# 7. Harmonize CES 2025 special responses
#    Its selected variables already use readable labels.
# ============================================================

for col in profile_columns:
    ces25[col] = ces25[col].astype("string").str.strip()

    ces25[col] = ces25[col].replace(
        {
            "skipped": pd.NA,
            "not asked": pd.NA,
        }
    )

# ============================================================
# 8. Create respondent IDs without changing caseid
# ============================================================


def make_respondent_id(series, year):
    numeric = pd.to_numeric(series, errors="raise")

    # Prevent IDs like 2553206150.0.
    if numeric.isna().any():
        raise ValueError(f"Missing caseid found in CES {year}.")

    if (numeric % 1 != 0).any():
        raise ValueError(
            f"Non-integer caseid found in CES {year}; inspect before use."
        )

    return (
        str(year) + "_" +
        numeric.astype("int64").astype("string")
    )


ces24["respondent_id"] = make_respondent_id(ces24["caseid"], 2024)
ces25["respondent_id"] = make_respondent_id(ces25["caseid"], 2025)

ces24["survey_year"] = 2024
ces25["survey_year"] = 2025


# ============================================================
# 9. Validate IDs and state coverage
# ============================================================

for year, df in [(2024, ces24), (2025, ces25)]:
    duplicate_count = df["respondent_id"].duplicated().sum()

    print(f"\nCES {year} duplicate respondent IDs:", duplicate_count)

    if duplicate_count:
        raise ValueError(
            f"Duplicate respondent IDs found in CES {year}."
        )

    print(
        f"CES {year} distinct locations:",
        df["_state_name"].nunique()
    )

# ============================================================
# 10. Compare labels across the two survey years
# ============================================================

print("\n--- Comparing profile categories across years ---")

for col in profile_columns:
    values24 = set(ces24[col].dropna().unique())
    values25 = set(ces25[col].dropna().unique())

    only24 = sorted(values24 - values25)
    only25 = sorted(values25 - values24)

    print(f"\n{col}")
    print("Only in CES 2024:", only24)
    print("Only in CES 2025:", only25)

# ============================================================
# 11. Load the 2026 gubernatorial races and select 36 states
# ============================================================

elections = pd.read_csv(ELECTIONS_FILE)

required_election_columns = {"race", "type"}
if not required_election_columns.issubset(elections.columns):
    raise ValueError("Election CSV must contain 'race' and 'type'.")

# Use only state-level gubernatorial races; exclude territories.
target_states = sorted(
    elections.loc[
        elections["type"].str.strip().str.lower() == "state",
        "race",
    ].dropna().str.strip().unique()
)

print("\nNumber of target states:", len(target_states))
print("Target states:", target_states)

if len(target_states) != 36:
    raise ValueError(
        f"Expected 36 state races, but found {len(target_states)}. "
        "Check governor_races_2026.csv."
    )

# Verify that every target state is present in both surveys.
for year, df in [(2024, ces24), (2025, ces25)]:
    missing_states = sorted(
        set(target_states) - set(df["_state_name"].unique())
    )
    if missing_states:
        raise ValueError(
            f"CES {year} is missing target states: {missing_states}"
        )


# ============================================================
# 12. Restrict datasets to target states and registered voters
# ============================================================

# Keep the original inputstate and profile-variable names.
ces24_target = ces24[
    ces24["_state_name"].isin(target_states)
    & ces24["votereg"].eq("Yes")
].copy()

ces25_target = ces25[
    ces25["_state_name"].isin(target_states)
    & ces25["votereg"].eq("Yes")
].copy()

print("\nRegistered-voter records available in target states:")
print("CES 2024:", len(ces24_target))
print("CES 2025:", len(ces25_target))

# Count eligible respondents by state and survey year.
eligible_counts = pd.concat([
    ces24_target.groupby("_state_name").size().rename("CES_2024"),
    ces25_target.groupby("_state_name").size().rename("CES_2025"),
], axis=1).reindex(target_states, fill_value=0).fillna(0).astype(int)

eligible_counts["Combined_available"] = (
    eligible_counts["CES_2024"] + eligible_counts["CES_2025"]
)

print("\nEligible registered voters by state:")
print(eligible_counts.to_string())


# ============================================================
# 13. Combine eligible records, preserving original columns
# ============================================================

columns_to_keep = (
    ["caseid", "inputstate", "commonweight"]
    + profile_columns
    + common_election_columns
    + ["respondent_id", "survey_year", "_state_name"]
)

columns_to_keep = list(dict.fromkeys(columns_to_keep))

combined = pd.concat(
    [
        ces24_target[columns_to_keep],
        ces25_target[columns_to_keep],
    ],
    ignore_index=True,
)

if combined["respondent_id"].duplicated().any():
    raise ValueError("Duplicate respondent IDs found in combined data.")


# ============================================================
# 14. Sample up to 100 registered voters per target state
#     Prioritize CES 2025, then supplement with CES 2024
# ============================================================


sampled_groups = []
sampling_report_rows = []

rng = np.random.default_rng(RANDOM_SEED)


def weighted_sample(df, n):
    """Draw n respondents without replacement, with probability proportional to
    their CES survey weight (commonweight), so the sample resembles the state's
    registered voters. Takes everyone if n covers the whole pool."""
    if n >= len(df):
        return df
    p = df["commonweight"].fillna(0).to_numpy(dtype=float)
    idx = rng.choice(len(df), size=n, replace=False, p=p / p.sum())
    return df.iloc[idx]

for state in target_states:
    state_data = combined[combined["_state_name"] == state]

    available25 = state_data[state_data["survey_year"] == 2025]
    available24 = state_data[state_data["survey_year"] == 2024]

    n25 = min(TARGET_PER_STATE, len(available25))
    sample25 = weighted_sample(available25, n25)

    remaining = TARGET_PER_STATE - len(sample25)
    
    n24 = min(remaining, len(available24))
    sample24 = weighted_sample(available24, n24)

    state_sample = pd.concat(
        [sample25, sample24],
        ignore_index=True,
    )
    sampled_groups.append(state_sample)

    sampling_report_rows.append({
        "state": state,
        "available_2025": len(available25),
        "available_2024": len(available24),
        "sampled_2025": len(sample25),
        "sampled_2024": len(sample24),
        "sampled_total": len(state_sample),
        "target_met": len(state_sample) == TARGET_PER_STATE,
    })

samples = pd.concat(sampled_groups, ignore_index=True)
sampling_report = pd.DataFrame(sampling_report_rows)

print("\n--- Sampling report ---")
print(sampling_report.to_string(index=False))

print("\nTotal sampled:", len(samples))
print("Unique respondent IDs:", samples["respondent_id"].nunique())
print("Duplicate IDs:", samples["respondent_id"].duplicated().sum())

print("\nStates where 100 eligible respondents were unavailable:")
print(
    sampling_report.loc[
        ~sampling_report["target_met"],
        ["state", "available_2025", "available_2024", "sampled_total"],
    ].to_string(index=False)
)

# Structural checks: these do not guarantee representativeness.
assert samples["respondent_id"].is_unique
assert samples["_state_name"].isin(target_states).all()
assert (sampling_report["sampled_total"] <= TARGET_PER_STATE).all()

# ============================================================
# QUALITY CHECKS FOR THE FINAL VOTER SAMPLE
# ============================================================

print("\n========== QUALITY CHECKS ==========")

# 1. Basic size and uniqueness checks
print("\n1. BASIC CHECKS")
print("Total records:", len(samples))
print("Unique respondent IDs:", samples["respondent_id"].nunique())
print("Duplicate respondent IDs:", samples["respondent_id"].duplicated().sum())
print("Number of states:", samples["_state_name"].nunique())

# 2. Check missing values in persona-related fields
print("\n2. MISSING VALUES IN PERSONA FIELDS")

persona_columns = [
    "birthyr",
    "gender4",
    "_state_name",
    "race",
    "hispanic",
    "educ",
    "faminc_new",
    "pid7",
    "ideo5",
    "votereg",
]

missing_count = samples[persona_columns].isna().sum()

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": (
        missing_count / len(samples) * 100
    ).round(2),
})

print(missing_report)

# 3. Check missing values separately for each survey year
print("\n3. MISSING VALUES BY SURVEY YEAR")

print(
    samples.groupby("survey_year")[persona_columns]
    .agg(lambda column: column.isna().sum())
)

# 4. Check how many records came from each survey year
print("\n4. RECORDS BY SURVEY YEAR")
print(samples["survey_year"].value_counts().sort_index())

# 5. Check the number of records from each state
print("\n5. RECORDS BY STATE")

state_counts = (
    samples["_state_name"]
    .value_counts()
    .sort_index()
)

print(state_counts)

# 6. Check the year composition within each state
print("\n6. SURVEY YEAR COMPOSITION BY STATE")

state_year_counts = pd.crosstab(
    samples["_state_name"],
    samples["survey_year"]
)

print(state_year_counts)

# 7. Check registration status
print("\n7. VOTER REGISTRATION STATUS")
print(samples["votereg"].value_counts(dropna=False))

# 8. Check birth years
print("\n8. BIRTH YEAR CHECK")

samples["birthyr"] = pd.to_numeric(
    samples["birthyr"],
    errors="coerce"
).astype("Int64")

print("Missing birth years:", samples["birthyr"].isna().sum())
print("Earliest birth year:", samples["birthyr"].min())
print("Latest birth year:", samples["birthyr"].max())

invalid_birth_years = (
    samples["birthyr"].notna()
    & ~samples["birthyr"].between(1900, 2010)
)

print("Birth years outside 1900-2010:", invalid_birth_years.sum())

# 9. Automated checks: stop if important conditions fail
print("\n9. AUTOMATED VALIDATION")

assert len(samples) == 3600, "Expected 3,600 records."
assert samples["respondent_id"].is_unique, "Duplicate respondent IDs found."
assert samples["_state_name"].nunique() == 36, "Expected 36 states."
assert (state_counts == 100).all(), "Some states do not have 100 records."
assert samples["_state_name"].isin(
    target_states).all(), "Unexpected state found."
assert samples["votereg"].eq("Yes").all(
), "Some records are not registered voters."
assert samples["survey_year"].isin(
    [2024, 2025]).all(), "Unexpected survey year found."
assert not invalid_birth_years.any(), "Invalid birth years found."

print("All automated validation checks passed!")


# ============================================================
# BUILD AND SAVE FINAL OUTPUTS
# ============================================================

# Calculate approximate age in the year the survey was conducted.
birth_year = pd.to_numeric(samples["birthyr"], errors="raise")

age = samples["survey_year"] - birth_year

# Combine race and ethnicity into one category.
race_ethnicity = samples["race"].copy()

is_hispanic = samples["hispanic"].eq("Yes")
race_ethnicity.loc[is_hispanic] = "Hispanic"

# Keep only the eight requested persona attributes.
personas = pd.DataFrame({
    "age": age.astype("Int64"),
    "gender": samples["gender4"],
    "state": samples["_state_name"],
    "race_ethnicity": race_ethnicity,
    "education": samples["educ"],
    "income": samples["faminc_new"],
    "party_alignment": samples["pid7"],
    "ideology": samples["ideo5"],
})

# Validate the final schema.
expected_columns = [
    "age",
    "gender",
    "state",
    "race_ethnicity",
    "education",
    "income",
    "party_alignment",
    "ideology",
]

assert personas.columns.tolist() == expected_columns
assert personas["state"].notna().all()
assert personas["state"].isin(target_states).all()
assert personas["age"].notna().all()
assert personas["age"].between(18, 120).all()

# Save the final persona dataset.
personas_file = OUTPUT_DIR / "personas.parquet"
personas.to_parquet(personas_file, index=False)

# Save supporting audit reports separately.
sampling_report_file = OUTPUT_DIR / "sampling_report.csv"
eligible_counts_file = OUTPUT_DIR / "eligible_counts.csv"

sampling_report.to_csv(sampling_report_file, index=False)
eligible_counts.to_csv(eligible_counts_file)

print("\nSaved outputs:")
print(f"Personas: {personas_file}")
print(f"Sampling report: {sampling_report_file}")
print(f"Eligible counts: {eligible_counts_file}")

print("\nFinal persona dataset shape:", personas.shape)
print("Final columns:", personas.columns.tolist())
print("\nFirst five personas:")
print(personas.head().to_string(index=False))
print("\nMissing values:")
print(personas.isna().sum())
