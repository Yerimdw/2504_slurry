import streamlit as st
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt

import numpy as np
st.set_page_config(page_title="Bayesian Optimization for Anode Slurry", layout="wide")
st.title("Bayesian Optimization for Anode Slurry(GA)")
import pandas as pd
file_path = "./BO_slurry_Data.xlsx"
df = pd.read_excel(file_path, usecols='B:F', skiprows=[0,1,2], header=0, engine='openpyxl')
raw_data=df.values
df = pd.read_excel(file_path, usecols='B:E', skiprows=[0], nrows=2, header=None, engine='openpyxl')
bounds=df.values

input_data=raw_data[:,0:4]
score=raw_data[:,4:]

if st.button("Genetic Algorithm을 통해 Initial Candidate를 추천받으려면 버튼을 누르세요."):
    scaler_input = MinMaxScaler()
    input_data_scaled = scaler_input.fit_transform(input_data)
    input_data_scaled=torch.from_numpy(input_data_scaled).double()
    score=torch.from_numpy(score).double()
    bounds=torch.from_numpy(bounds).double()

    train_x = torch.tensor(input_data_scaled, dtype=torch.float64)
    train_y = torch.tensor(score, dtype=torch.float64)


    def expected_improvement(x_tensor, model, best_f):
        ei = ExpectedImprovement(model=model, best_f=best_f)
        return ei(x_tensor.unsqueeze(0))  # shape: [1, d]


    def run_GA_for_initial_candidates(model, bounds_tensor, best_f, pop_size=20, generations=30):
        dim = bounds_tensor.shape[1]
        pop = torch.rand(pop_size, dim, dtype=torch.float64)

        for _ in range(generations):
            # Evaluate EI
            ei_vals = torch.tensor([expected_improvement(x, model, best_f) for x in pop], dtype=torch.float64).squeeze()
            # Select top
            topk = torch.topk(ei_vals, k=pop_size // 2)
            parents = pop[topk.indices]
            # Crossover & Mutation
            children = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[(i + 1) % len(parents)]
                alpha = torch.rand(1).item()
                child = alpha * p1 + (1 - alpha) * p2
                # Mutation
                mutation = 0.05 * torch.randn(dim, dtype=torch.float64)
                child += mutation
                child = torch.clamp(child, 0.0, 1.0)
                children.append(child)
            pop = torch.vstack((parents, torch.stack(children)))

        # Return top-N candidates (normalized)
        final_ei = torch.tensor([expected_improvement(x, model, best_f) for x in pop], dtype=torch.float64).squeeze()
        best_indices = torch.topk(final_ei, k=3).indices
        return pop[best_indices]


    # ---------------------- Fit Initial GP ----------------------
    gp_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)

    # ---------------------- GA-based Initial Candidates ----------------------
    normalized_bounds = torch.tensor([[0.0] * 4, [1.0] * 4], dtype=torch.float64)
    best_y = train_y.max().item()
    init_candidates = run_GA_for_initial_candidates(gp_model, normalized_bounds, best_y, pop_size=20, generations=50)

    # 역정규화하여 실제 조성 출력
    init_candidates_denorm = scaler_input.inverse_transform(init_candidates.numpy())
    print("Initial candidate:")
    print(init_candidates_denorm)

    candidate_values = init_candidates_denorm[0]
    columns = ["Carbon Black", "Binder", "Solvent", "Graphite"]
    table_data = [
        ["Composition"] + columns,
        ["(g)"] + candidate_values.tolist()
    ]

    df_table = pd.DataFrame(table_data)
    for col in df_table.columns[1:]:
        df_table.iloc[1, col] = "{:.4f}".format(df_table.iloc[1, col])

    styled_df = (
        df_table.style
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {"selector": "th", "props": [("min-width", "120px"), ("max-width", "120px"), ("text-align", "center")]},
            {"selector": "td", "props": [("min-width", "120px"), ("max-width", "120px"), ("text-align", "center")]}
        ])
    )

    st.title("Candidate for Slurry Composition(GA)")
    st.table(styled_df)

    cb = pd.read_excel(file_path, usecols='B', skiprows=[0,1,2], header=0, engine='openpyxl')
    ys = pd.read_excel(file_path, usecols='F', skiprows=[0,1,2], header=0, engine='openpyxl')

    cb = cb.to_numpy().flatten()
    ys = ys.to_numpy().flatten()

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(cb, ys, color="blue")
    ax.scatter(cb, ys, color="black", label="Experimental data", s=20)

    candidate_cb = init_candidates_denorm[0, 0].item()
    candidate_ys = np.interp(candidate_cb, cb, ys)
    ax.scatter(candidate_cb, candidate_ys, color="red", s=20, label="Candidate(GA)", zorder=10)

    ax.set_title("Carbon Black vs Yields stress")
    ax.set_xlabel("Carbon Black [wt%]")
    ax.set_ylabel("Yield stress [Pa]")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
