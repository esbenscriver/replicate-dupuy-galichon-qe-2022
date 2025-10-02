import jax
from jax import numpy as jnp

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET

from matching import MatchingModel, Data
from dupuy_galichon_2022 import (
    variables_to_describe,
    variable_names,
    covariate_names,
    dupuy_galichon_estimates,
)

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

include_transfer_constant = True
include_scale_parameters = True


def normalize_variables(variables):
    return (variables - np.mean(variables, axis=0, keepdims=True)) / np.std(
        variables, axis=0, keepdims=True
    )


def read_excel_xml(file_path):
    """Read Excel XML file and convert to pandas DataFrame"""
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define namespace
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}

    # Find the worksheet
    worksheet = root.find(".//ss:Worksheet", ns)
    table = worksheet.find("ss:Table", ns)

    # Extract data
    rows = []
    for row in table.findall("ss:Row", ns):
        row_data = []
        for cell in row.findall("ss:Cell", ns):
            data_elem = cell.find("ss:Data", ns)
            if data_elem is not None:
                row_data.append(data_elem.text)
            else:
                row_data.append("")
        rows.append(row_data)

    # Create DataFrame
    if rows:
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
    else:
        return pd.DataFrame()


# Read the XML file
file_path = "Repl_QE/Data/workingdataset_occind.xml"
df = read_excel_xml(file_path)
# print(df.head(10))

# Convert numeric columns to proper data types
numeric_columns = [
    "wage",
    "x_yrseduc",
    "x_exp",
    "x_ethn",
    "x_sex",
    "x_married",
    "x_lma",
    "x_region",
    "x_union",
    "y_hospital",
    "y_risk_rateh_occind_ave",
    "y_public",
    "x_white",
    "x_black",
    "x_asian",
]

# Convert numeric columns to proper data types first
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create summary statistics table
print("\n" + "=" * 80)
print("Summary Statistics Table")
print("=" * 80)

summary_stats = df[variables_to_describe].describe().T
summary_stats = summary_stats.round(2)
summary_stats = summary_stats[["mean", "std", "min", "max"]]
summary_stats = summary_stats.rename(index=variable_names)
print(summary_stats)
print("=" * 80)
print(f"Number of observations: {len(df)}")
print(f"Number of variables: {len(df.columns)}\n")

# Save summary statistics to Markdown files
summary_stats.to_markdown("output/summary_stats.md", floatfmt=".2f")

none_dummy_columns = ["x_yrseduc", "x_exp", "y_risk_rateh_occind_ave"]

array_none_dummy = df[none_dummy_columns].to_numpy()
df[none_dummy_columns] = normalize_variables(array_none_dummy)

df["x_exp_sq"] = df["x_exp"].to_numpy() ** 2

X_columns = [
    "x_yrseduc",
    "x_exp",
    "x_sex",
    "x_married",
    "x_white",
    "x_black",
    "x_asian",
    "x_exp_sq",
]
Y_columns = ["y_risk_rateh_occind_ave", "y_public"]
y_public_index = 1
x_yrseduc_index = 0

observed_wage = jnp.log(jnp.asarray(df["wage"].to_numpy()))

X_vars = jnp.asarray(df[X_columns].to_numpy())
Y_vars = jnp.asarray(df[Y_columns].to_numpy())

number_of_Xvars, number_of_Yvars = 3, 2
X_vars_interacted = jnp.zeros((X_vars.shape[0], number_of_Xvars * number_of_Yvars))

for x in range(number_of_Xvars):
    for y in range(number_of_Yvars):
        X_vars_interacted = X_vars_interacted.at[:, x * number_of_Yvars + y].set(
            X_vars[:, x] * Y_vars[:, y]
        )

covariates_Y = jnp.concatenate([X_vars, X_vars_interacted], axis=1)
covariates_X = jnp.concatenate(
    [
        Y_vars,
        X_vars[:, [x_yrseduc_index]] * Y_vars[:, [y_public_index]],
    ],
    axis=1,
)
print(f"{covariates_X.shape=}, {covariates_Y.shape=}")

covariates = jnp.concatenate([covariates_Y, covariates_X], axis=1)

model = MatchingModel(
    covariates_X=covariates_X[None, :, :],
    covariates_Y=covariates_Y[:, None, :],
    marginal_distribution_X=jnp.ones((1, covariates_X.shape[0]))
    / covariates_X.shape[0],
    marginal_distribution_Y=jnp.ones((covariates_Y.shape[0], 1))
    / covariates_Y.shape[0],
    continuous_distributed_attributes=True,
    include_transfer_constant=include_transfer_constant,
    include_scale_parameters=include_scale_parameters,
)

parameter_names = covariate_names.copy()
if model.include_transfer_constant is True:
    parameter_names += ["Salary constant"]

if model.include_scale_parameters is True:
    parameter_names += ["Scale parameter (workers)", "Scale parameter (firms)"]

df_covariate_stats = pd.DataFrame(
    {
        "name": covariate_names,
        "mean": jnp.mean(covariates, axis=0),
        "std": jnp.std(covariates, axis=0),
        "min": jnp.min(covariates, axis=0),
        "max": jnp.max(covariates, axis=0),
    }
).round(3).set_index("name").rename_axis(None)

df_covariate_stats.to_markdown("output/covariate_stats.md", floatfmt=".3f")
print("=" * 80)
print("Covariate Statistics Table")
print("=" * 80)
print(df_covariate_stats)
print("=" * 80)
print(f"Number of covariates: {len(df_covariate_stats)}\n")

data = Data(transfers=observed_wage, matches=jnp.ones_like(observed_wage, dtype=float))

guess = jnp.ones(len(covariate_names))
if model.include_transfer_constant is True:
    guess = jnp.concatenate([guess, jnp.array([0.0])], axis=0)

if model.include_scale_parameters is True:
    # guess = jnp.concatenate([guess, jnp.log(jnp.array([1.0, 1.0]))], axis=0)
    guess = jnp.concatenate([guess, jnp.array([1.0, 1.0])], axis=0)

print(f"\nlogL(guess)={-model.neg_log_likelihood(guess, data)}")

estimates_transformed = model.fit(guess, data, maxiter=1000, verbose=True)
mp = model.extract_model_parameters(estimates_transformed, transform=True)
estimates = model.class2vec(mp, transform=False)

print(f"\nlogL(estimates)={-model.neg_log_likelihood(estimates, data)}")

if include_transfer_constant is True and include_scale_parameters is True:
    dupuy_galichon_estimates = jnp.asarray(dupuy_galichon_estimates)
    print(
        f"\nlogL(dupuy_galichon_estimates)={-model.neg_log_likelihood(dupuy_galichon_estimates, data)}"
    )
    df_estimates = pd.DataFrame(
        {
            "name": parameter_names,
            "Dupuy and Galichon (2022)": dupuy_galichon_estimates,
            "Andersen (2025)": estimates,
        }
    ).round(3).set_index("name").rename_axis(None)

    print(
        f"R^2 (Dupuy and Galichon (2022): {model.R_squared(dupuy_galichon_estimates, data)}"
        + f", Andersen (2025): {model.R_squared(estimates, data)}\n"
    )
else:
    df_estimates = pd.DataFrame(
        {
            "name": parameter_names,
            "estimates": estimates,
        }
    ).round(3).set_index("name").rename_axis(None)
print("\n" + "=" * 80)
print("Parameter Estimates")
print("=" * 80)
print(df_estimates)
print("=" * 80)
print(f"Number of estimated parameters: {len(df_estimates)}\n")

variance, mean = model.compute_moments(estimates, data)

df_moments = pd.DataFrame(
    {
        "name": ["mean", "variance"],
        "estimates": jnp.asarray([mean, variance]),
    }
).round(3).set_index("name").rename_axis(None)
print("=" * 80)
print("Estimated Moments of Measurement Errors")
print("=" * 80)
print(df_moments)
print("=" * 80)

if include_transfer_constant is True and include_scale_parameters is True:
    df_estimates.to_markdown("output/estimated_parameters.md", floatfmt=".3f")
    df_moments.to_markdown("output/estimated_moments.md", floatfmt=".3f")
else:
    df_estimates.to_markdown(
        f"output/estimated_parameters_constant_{include_transfer_constant}_scale_{include_scale_parameters}.md",
        floatfmt=".3f",
    )
    df_moments.to_markdown(
        f"output/estimated_moments_constant_{include_transfer_constant}_scale_{include_scale_parameters}.md",
        floatfmt=".3f",
    )
