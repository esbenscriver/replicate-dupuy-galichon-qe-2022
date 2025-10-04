import jax
from jax import numpy as jnp

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from scipy.io import loadmat

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
import_matlab_file = False
print("\n" + "=" * 80)
print("Replication Settings")
print("=" * 80)
print(f"Include transfer constant: {include_transfer_constant}")
print(f"Include scale parameters: {include_scale_parameters}")  
print(f"Import covariates and wages from Matlab .mat files: {import_matlab_file}")
print("=" * 80+"\n")

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

if import_matlab_file is True:
    covariates_X = jnp.nan_to_num(loadmat("Repl_QE/Data/BF_A.mat")["BF_A_tilda"], nan=0.0)
    covariates_Y = jnp.nan_to_num(loadmat("Repl_QE/Data/BF_G.mat")["BF_G_tilda"], nan=0.0)
    observed_wage = jnp.nan_to_num(loadmat("Repl_QE/Data/wage.mat")["w"], nan=0.0).squeeze()

    # Or specify a custom order
    new_order_X =  [1, 2, 0]
    covariates_X = covariates_X[:, :, new_order_X]
    new_order_Y = [6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5]
    covariates_Y = covariates_Y[:, :, new_order_Y]
else:
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

    X_vars = jnp.asarray(df[X_columns].to_numpy())[:,None,:]
    Y_vars = jnp.asarray(df[Y_columns].to_numpy())[None,:,:]

    X_columns_to_interact = ["x_yrseduc", "x_exp", "x_sex"]
    Y_columns_to_interact = ["y_risk_rateh_occind_ave", "y_public"]

    N = X_vars.shape[0]

    X_vars_interacted = jnp.zeros((N, N, 0))
    for y_col in Y_columns_to_interact:
        y_var = df[y_col].to_numpy()[None,:]
        for x_col in X_columns_to_interact:
            x_var = df[x_col].to_numpy()[:,None]

            X_vars_interacted = jnp.append(
                X_vars_interacted, 
                (x_var * y_var)[:,:,None], 
                axis=-1
            )

    x_axis, y_axis = 0, 1

    covariates_Y = jnp.concatenate(
        [
            jnp.repeat(X_vars, repeats=N, axis=y_axis), 
            X_vars_interacted
        ], 
        axis=-1,
    )
    covariates_X = jnp.concatenate(
        [
            jnp.repeat(Y_vars, repeats=N, axis=x_axis),
            X_vars[:, :, [x_yrseduc_index]] * Y_vars[:, :, [y_public_index]],
        ],
        axis=-1,
    )

covariates = jnp.concatenate([covariates_X, covariates_Y], axis=-1)
print(f"{covariates_X.shape = }, {covariates_Y.shape = }")
df_covariate_stats = pd.DataFrame(
    {
        "name": covariate_names,
        "mean": jnp.mean(covariates, axis=(0, 1)),
        "std": jnp.std(covariates, axis=(0, 1)),
        "min": jnp.min(covariates, axis=(0, 1)),
        "max": jnp.max(covariates, axis=(0, 1)),
    }
).round(4).set_index("name").rename_axis(None)

if import_matlab_file is True:
    df_covariate_stats.to_markdown("output/covariate_stats_matlab.md", floatfmt=".4f")
else:
    df_covariate_stats.to_markdown("output/covariate_stats.md", floatfmt=".4f")

print("=" * 80)
print("Covariate Statistics Table")
print("=" * 80)
print(df_covariate_stats)
print("=" * 80)
print(f"Number of covariates: {len(df_covariate_stats)}\n")

model = MatchingModel(
    covariates_X=covariates_X,
    covariates_Y=covariates_Y,
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

data = Data(transfers=observed_wage, matches=jnp.ones_like(observed_wage, dtype=float))

guess = jnp.zeros(len(covariate_names))
if model.include_transfer_constant is True:
    guess = jnp.concatenate([guess, jnp.array([0.0])], axis=0)
if model.include_scale_parameters is True:
    guess = jnp.concatenate([guess, jnp.array([1.0, 1.0])], axis=0)

# guess = jnp.asarray(dupuy_galichon_estimates)

I = 10
estimates_matrix = jnp.zeros((len(guess), I))
for i in range(I):
    print(f"\ni={i+1}: logL(guess)={-model.neg_log_likelihood(guess, data)}")
    par_est = model.fit(guess, data)
    estimates_matrix = estimates_matrix.at[:,i].set(par_est)
    guess = par_est

print(f"\nestimate:\n{estimates_matrix}")
estimates = estimates_matrix[:,-1]
print(f"\nlogL(estimates)={-model.neg_log_likelihood(estimates, data)}")

if include_transfer_constant is True and include_scale_parameters is True:
    dupuy_galichon_estimates = jnp.asarray(dupuy_galichon_estimates)
    df_estimates = pd.DataFrame(
        {
            "name": parameter_names,
            "Dupuy and Galichon (2022)": dupuy_galichon_estimates,
            "Andersen (2025)": estimates,
        }
    ).round(3).set_index("name").rename_axis(None)
    variance_DG, mean_DG = model.compute_moments(dupuy_galichon_estimates, data)
    variance, mean = model.compute_moments(estimates, data)

    df_moments = pd.DataFrame(
        {
            "name": ["mean", "variance"],
            "Dupuy and Galichon (2022)": jnp.asarray([mean_DG, variance_DG]),
            "Andersen (2025)": jnp.asarray([mean, variance]),
        }
    ).round(3).set_index("name").rename_axis(None)

    logL_DG = -model.neg_log_likelihood(dupuy_galichon_estimates, data)
    logL = -model.neg_log_likelihood(estimates, data)

    R2_DG = model.R_squared(dupuy_galichon_estimates, data)
    R2 = model.R_squared(estimates, data)

    df_objectives = pd.DataFrame(
        {
            "": ["Log-likelihood", "R-squared"],
            "Dupuy and Galichon (2022)": [logL_DG, R2_DG],
            "Andersen (2025)": [logL, R2],
        }
    ).round(3).set_index("").rename_axis(None)
else:
    df_estimates = pd.DataFrame(
        {
            "name": parameter_names,
            "estimates": estimates,
        }
    ).round(3).set_index("name").rename_axis(None)
    variance, mean = model.compute_moments(estimates, data)

    df_moments = pd.DataFrame(
        {
            "name": ["mean", "variance"],
            "estimates": jnp.asarray([mean, variance]),
        }
    ).round(3).set_index("name").rename_axis(None)

    logL = -model.neg_log_likelihood(estimates, data)
    R2 = model.R_squared(estimates, data)

    df_objectives = pd.DataFrame(
        {
            "": ["Log-likelihood", "R-squared"],
            "fit": [logL, R2],
        }
    ).round(3).set_index("").rename_axis(None)

print("\n" + "=" * 80)
print("Parameter Estimates")
print("=" * 80)
print(df_estimates)
print("=" * 80)
print(f"Number of estimated parameters: {len(df_estimates)}\n")

print("=" * 80)
print("Estimated Moments of Measurement Errors")
print("=" * 80)
print(df_moments)
print("=" * 80)

if include_transfer_constant is True and include_scale_parameters is True and import_matlab_file is True:
    df_estimates.to_markdown("output/estimated_parameters.md", floatfmt=".3f")
    df_moments.to_markdown("output/estimated_moments.md", floatfmt=".3f")
    df_objectives.to_markdown("output/objective.md", floatfmt=".3f")
else:
    df_estimates.to_markdown(
        f"output/estimated_parameters_constant_{include_transfer_constant}_scale_{include_scale_parameters}_MatLab_{import_matlab_file}.md",
        floatfmt=".3f",
    )
    df_moments.to_markdown(
        f"output/estimated_moments_constant_{include_transfer_constant}_scale_{include_scale_parameters}_MatLab_{import_matlab_file}.md",
        floatfmt=".3f",
    )
    df_objectives.to_markdown(
        f"output/objective_constant_{include_transfer_constant}_scale_{include_scale_parameters}_MatLab_{import_matlab_file}.md",
        floatfmt=".3f",
    )

pd.DataFrame(estimates_matrix, columns=[f"iter_{i}" for i in range(I)]).round(4).to_markdown(
    f"output/estimation_path_{include_transfer_constant}_scale_{include_scale_parameters}_MatLab_{import_matlab_file}.csv",
    floatfmt=".4f",
)