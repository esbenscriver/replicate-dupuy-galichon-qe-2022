import jax
from jax import numpy as jnp

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET

from module.matching import MatchingModel, Data
from module.dupuy_galichon_2022 import (
    variables_to_describe,
    variable_names,
    covariate_names,
    dupuy_galichon_estimates,
)

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

standardize = True
log_transform_scale = False

number_of_starting_values = 4
number_of_optimizations = 1
model_names = [f"({g})" for g in range(number_of_starting_values)]

specification_name = f"standardize_{standardize}"

print("\n" + "=" * 80)
print("Replication Settings")
print("=" * 80)
print(f"standardize numerical values: {standardize}")
print(f"log transform scale parameters: {log_transform_scale}")
print("=" * 80+"\n")

def standardize_variables(variables):
    return (variables - np.mean(variables, axis=0, keepdims=True)) / np.std(
        variables, axis=0, keepdims=True
    )


def normalize_variables(variables):
    return (variables - np.min(variables, axis=0, keepdims=True)) / (
        np.max(variables, axis=0, keepdims=True) - np.min(variables, axis=0, keepdims=True)
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
    if worksheet is None:
        return pd.DataFrame()

    table = worksheet.find("ss:Table", ns)
    if table is None:
        return pd.DataFrame()

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


def mle_estimation(guess: jax.Array, model: MatchingModel):
    estimates_raw = model.fit(guess, data)
    estimates = model.transform_parameters(estimates_raw)
    loglik = -model.neg_log_likelihood(estimates_raw, data)
    print(f"\n logL(estimates_raw)={loglik:.6f}\n")
    return estimates_raw, estimates, loglik

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
summary_stats = df[variables_to_describe].describe().T
summary_stats = summary_stats[["mean", "std", "min", "max"]]
summary_stats = summary_stats.rename(index=variable_names)

print("\n" + "=" * 80)
print("Summary Statistics Table")
print("=" * 80)
print(summary_stats.round(2))
print("=" * 80)
print(f"Number of observations: {len(df)}")
print(f"Number of variables: {len(df.columns)}\n")

# Save summary statistics to Markdown files
summary_stats.to_markdown("output/summary_stats.md", floatfmt=".2f")

none_dummy_columns = ["x_yrseduc", "x_exp", "y_risk_rateh_occind_ave"]

array_none_dummy = df[none_dummy_columns].to_numpy()
if standardize is True:
    df[none_dummy_columns] = standardize_variables(array_none_dummy)
else:
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

observed_wage = jnp.log(jnp.asarray(df["wage"].to_numpy()))

covariates = jnp.concatenate([covariates_X, covariates_Y], axis=-1)
print(f"{covariates_X.shape = }, {covariates_Y.shape = }")
df_covariate_stats = pd.DataFrame(
    {
        "": covariate_names,
        "mean": jnp.mean(covariates, axis=(0, 1)),
        "std": jnp.std(covariates, axis=(0, 1)),
        "min": jnp.min(covariates, axis=(0, 1)),
        "max": jnp.max(covariates, axis=(0, 1)),
    }
).set_index("")

df_covariate_stats.to_markdown(f"output/covariate_stats_standardize_{standardize}.md", floatfmt=".4f")

print("\n" + "=" * 80)
print("Covariate Statistics Table")
print("=" * 80)
print(df_covariate_stats.round(3))
print("=" * 80)
print(f"Number of covariates: {len(df_covariate_stats)}\n")

model = MatchingModel(
    covariates_X=covariates_X,
    covariates_Y=covariates_Y,
    marginal_distribution_X=jnp.ones((1, covariates_X.shape[0]))
    / covariates_X.shape[0],
    marginal_distribution_Y=jnp.ones((covariates_Y.shape[0], 1))
    / covariates_Y.shape[0],
    include_transfer_constant=False,
    log_transform_scale=log_transform_scale,
    reference=-1,
    continuous_distributed_attributes=True,
    include_scale_parameters=True,
)

parameter_names = covariate_names.copy()
parameter_names += ["Scale parameter (workers)", "Scale parameter (firms)"]

data = Data(transfers=observed_wage, matches=jnp.ones_like(observed_wage, dtype=float))

dupuy_galichon_estimates = jnp.asarray(dupuy_galichon_estimates)

guess = dupuy_galichon_estimates[:len(covariate_names)]
if model.log_transform_scale:
    guess = jnp.concatenate([guess, jnp.log(dupuy_galichon_estimates[-2:])], axis=0)
else:
    guess = jnp.concatenate([guess, dupuy_galichon_estimates[-2:]], axis=0)

max_loglik = -jnp.inf
max_loglik_idx = 0
estimates_raw_sensitivity = jnp.zeros((len(guess), 0))
for g in range(number_of_starting_values):
    if g >0:
        guess = jax.random.uniform(jax.random.PRNGKey(g), guess.shape)

    estimates_raw_g, estimates_g, loglik_g = mle_estimation(guess, model)
    loglik_h = -jnp.inf
    for h in range(number_of_optimizations):
        if h > 0:
            estimates_raw_h, estimates_h, loglik_h = mle_estimation(guess, model)
            
        if loglik_h > loglik_g:
            estimates_raw_g, estimates_g, loglik_g = estimates_raw_h, estimates_h, loglik_h
        else:
            print(f"{loglik_h > loglik_g = }")
            break
        
        guess = estimates_raw_g

    estimates_raw_sensitivity = jnp.concatenate(
        [estimates_raw_sensitivity, estimates_raw_g[:,None]], 
        axis=1,
    )
    print(f"estimates_raw_sensitivity:\n{estimates_raw_sensitivity.round(4)}")
    
    max_loglik_idx = jnp.where(max_loglik > loglik_g, max_loglik_idx, g)
    max_loglik = jnp.maximum(max_loglik, loglik_g)
    print(f"g={g}: max_loglik_idx={max_loglik_idx}, max_loglik={max_loglik:.6f}.")

estimates_raw = estimates_raw_sensitivity[:,max_loglik_idx]
estimates = model.transform_parameters(estimates_raw)

estimates_sensitivity = jnp.zeros((len(estimates_raw), 0))
for g in range(estimates_raw_sensitivity.shape[1]):
    estimates_sensitivity = jnp.concatenate(
        [   estimates_sensitivity,
            model.transform_parameters(estimates_raw_sensitivity[:,g])[:,None]
        ],
        axis=1,
    )
    if g == 0:
        loglik_sensitivity = -model.neg_log_likelihood(estimates_raw_sensitivity[:,g], data)[None]
    else:
        loglik_sensitivity = jnp.concatenate(
            [   loglik_sensitivity,
                -model.neg_log_likelihood(estimates_raw_sensitivity[:,g], data)[None]
            ],
            axis=0,
    )

df_estimates = pd.DataFrame(
    {
        "": parameter_names,
        "Dupuy and Galichon (2022)": dupuy_galichon_estimates,
        "Our estimates": estimates,
        "differences": dupuy_galichon_estimates - estimates,
    }
).set_index("")
variance_DG, mean_DG, R2_DG = model.compute_moments(dupuy_galichon_estimates, data)
variance, mean, R2 = model.compute_moments(estimates_raw, data)

df_moments = pd.DataFrame(
    {
        "": ["mean, $m$", "variance, $s^2$"],
        "Dupuy and Galichon (2022)": jnp.asarray([mean_DG, variance_DG]),
        "Our estimates": jnp.asarray([mean, variance]),
    }
).set_index("")

logL_DG = -model.neg_log_likelihood(dupuy_galichon_estimates, data)
logL = -model.neg_log_likelihood(estimates_raw, data)

df_objectives = pd.DataFrame(
    {
        "": ["Log-likelihood", "R-squared"],
        "Dupuy and Galichon (2022)": jnp.asarray([logL_DG, R2_DG]),
        "Our results": jnp.asarray([logL, R2]),
    }
).set_index("")
df_sensitivity = pd.concat(
    [
        pd.DataFrame({"": parameter_names}),
        pd.DataFrame(estimates_sensitivity, columns=model_names),
    ],
    axis=1,
).set_index("")
df_loglik = pd.concat(
    [
        pd.DataFrame({"": ['log-likelihood']}),
        pd.DataFrame(loglik_sensitivity[None,:], columns=model_names),
    ],
    axis=1,
).set_index("")

# Print tables with estimation results
print("\n" + "=" * 80)
print("Parameter Estimates")
print("=" * 80)
print(df_estimates.round(3))
print("=" * 80)
print(f"Number of estimated parameters: {len(df_estimates)}")

print("\n" + "=" * 80)
print("Estimated Moments of Measurement Errors")
print("=" * 80)
print(df_moments.round(3))
print("=" * 80)

print("\n" + "=" * 80)
print("Model fit")
print("=" * 80)
print(df_objectives.round(3))
print("=" * 80)

print("\n" + "=" * 80)
print("Sensitivity analysis")
print("=" * 80)
print(df_sensitivity.round(3))
print("=" * 80)
print(f"loglik (0-3):\n{loglik_sensitivity.round(6)}")

# Save results to Markdown files
df_estimates.to_markdown(
    f"output/estimated_parameters_{specification_name}.md",
    floatfmt=".3f",
)
df_moments.to_markdown(
    f"output/estimated_moments_{specification_name}.md",
    floatfmt=".3f",
)
df_objectives.to_markdown(
    f"output/objective_{specification_name}.md",
    floatfmt=".3f",
)
df_sensitivity.to_markdown(
    f"output/sensitivity_{specification_name}.md",
    floatfmt=".3f",
)
df_loglik.to_markdown(
    f"output/loglik_{specification_name}.md",
    floatfmt=".6f",
)

print(f"{observed_wage.mean() = :.3f}")