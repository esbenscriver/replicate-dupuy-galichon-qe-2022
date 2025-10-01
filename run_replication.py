import jax
from jax import numpy as jnp

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET

from tabulate import tabulate

from matching import MatchingModel, Data

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)


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
print("=" * 80)
print("Summary Statistics Table")
print("=" * 80)

summary_stats = df[numeric_columns].describe().T
summary_stats = summary_stats.round(4)

# Format the table nicely
print(f"{'Variable':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 80)

for var in summary_stats.index:
    mean_val = summary_stats.loc[var, "mean"]
    std_val = summary_stats.loc[var, "std"]
    min_val = summary_stats.loc[var, "min"]
    max_val = summary_stats.loc[var, "max"]

    print(
        f"{var:<25} {mean_val:<12.2f} {std_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f}"
    )
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print(f"Number of observations: {len(df)}")
print(f"Number of variables: {len(df.columns)}")
print("=" * 80)

none_dummy_columns = ["x_yrseduc", "x_exp", "y_risk_rateh_occind_ave"]

array_none_dummy = df[none_dummy_columns].to_numpy()
df[none_dummy_columns] = normalize_variables(array_none_dummy)

df["x_exp_sq"] = df["x_exp"].to_numpy() ** 2
print(df.columns)

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

covariates = jnp.concatenate([covariates_Y, covariates_X], axis=1)

covariate_names = [
    "Years of schooling",
    "Years of experience",
    "Female",
    "Married",
    "White",
    "Black",
    "Asian",
    "Years of experience (squared)",
    "Risk x Years of schooling",
    "Public x Years of schooling",
    "Risk x Years of experience",
    "Public x Years of experience",
    "Risk x Females",
    "Public x Females",
    "Risk",
    "Public",
    "Public x Years of schooling",
]

other_parameters = 3  # sigma1, sigma2, salary constants
guess = jnp.ones((covariates_Y.shape[1] + covariates_X.shape[1] + other_parameters,))

model = MatchingModel(
    covariates_X=covariates_X[None, :, :],
    covariates_Y=covariates_Y[:, None, :],
)

parameter_names = covariate_names.copy()
if model.centered_variance is False:
    parameter_names += ["Salary constant"]
parameter_names += ["scale (X)", "scale (Y)"]

print("Summary statistics of covariates:")
print(
    tabulate(
        list(zip(
            covariate_names,
            jnp.mean(covariates, axis=0),
            jnp.std(covariates, axis=0),
            jnp.min(covariates, axis=0),
            jnp.max(covariates, axis=0),
        )),
        headers=["mean", "std", "min", "max"],
        tablefmt="grid",
    )
)
data = Data(transfers=observed_wage, matches=jnp.ones_like(observed_wage, dtype=float))

neg_log_lik = model.neg_log_likelihood(guess, data)
print(f"{model.replication = }: number of observations: {3*observed_wage.size}")
print(f"{neg_log_lik = }")


estimates = model.fit(guess, data, maxiter=3, verbose=True)
mp = model.extract_model_parameters(estimates)

estimates_transformed = jnp.concatenate(
    [mp.beta_X, mp.beta_Y, jnp.asarray([mp.transfer_constant, mp.sigma_X, mp.sigma_Y])],
    axis=0,
)

# Print estimated parameters
print(
    tabulate(
        list(zip(parameter_names, guess, estimates, estimates_transformed)),
        headers=["Parameter guess", "Parameter estimates", "Transformed estimates"],
        tablefmt="grid",
    )
)

df_estimates = pd.DataFrame(
    {
        "name": parameter_names,
        "estimates": estimates,
        "transformed_estimates": estimates_transformed,
    }
)

df_estimates.to_csv("estimates.csv", index=False)

mean, variance = model.compute_moments(estimates, data)
moment_estimates = jnp.asarray([mean, variance])
moment_names = ['mean','variance']

# Print estimated mean and variance of measurement errors
print(
    tabulate(
        list(zip(moment_names, moment_estimates)),
        headers=["Parameter estimates"],
        tablefmt="grid",
    )
)
