import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.optim import optimize_acqf

def run_GA_BO():
    st.title("Genetic Algorithm + BO 최적화 조성 추천 (3가지)")

    url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
    df = pd.read_csv(url)
    x_cols = ["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]
    y_cols = ["yield stress", "viscosity"]

    X = df[x_cols]
    Y = df[y_cols]

    scaler_input = MinMaxScaler()
    input_data_scaled = scaler_input.fit_transform(X)
    train_x = torch.tensor(input_data_scaled, dtype=torch.float64)
    train_y = torch.tensor(Y.values, dtype=torch.float64)

    gpr_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gpr_model.likelihood, gpr_model)
    fit_gpytorch_mll(mll)

    train_y_yield = torch.tensor(Y["yield stress"].values, dtype=torch.float64).unsqueeze(-1)

    def expected_improvement(x_tensor, model, best_f):
        ei = ExpectedImprovement(model=model, best_f=best_f)
        return ei(x_tensor.unsqueeze(0)).item()

    def fitness(x_tensor, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx):
        ei = ExpectedImprovement(model=model, best_f=best_f)
        ei_val = ei(x_tensor.unsqueeze(0)).item()

        x_denorm = scaler.inverse_transform(x_tensor.unsqueeze(0).numpy()).squeeze()
        graphite = x_denorm[graphite_idx]
        viscosity = x_denorm[viscosity_idx]

        graphite_min = 20.0
        graphite_max = 40.0
        graphite_norm = (graphite - graphite_min) / (graphite_max - graphite_min)

        if viscosity < 0.1 or viscosity > 1.5:
            viscosity_penalty = -1.0
        else:
            viscosity_target = 0.5
            viscosity_penalty = -abs(viscosity - viscosity_target)

        score = (1.0 * ei_val + 1.0 * viscosity_penalty + 0.5 * graphite_norm)
        return score

    def run_GA_for_initial_candidates(model, bounds_tensor, best_f, scaler, graphite_idx, viscosity_idx, pop_size=20,
                                      generations=30):
        dim = bounds_tensor.shape[1]
        pop = torch.rand(pop_size, dim, dtype=torch.float64)

        for gen in range(generations):
            fitness_vals = torch.tensor([
                fitness(x, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx)
                for x in pop
            ], dtype=torch.float64).squeeze()

            topk = torch.topk(fitness_vals, k=pop_size // 2)
            parents = pop[topk.indices]

            children = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[(i + 1) % len(parents)]
                alpha = torch.rand(1).item()
                child = alpha * p1 + (1 - alpha) * p2
                mutation = 0.05 * torch.randn(dim, dtype=torch.float64)
                child += mutation
                child = torch.clamp(child, 0.0, 1.0)
                children.append(child)

            pop = torch.vstack((parents, torch.stack(children)))

        final_fitness = torch.tensor([
            fitness(x, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx)
            for x in pop
        ], dtype=torch.float64).squeeze()
        best_indices = torch.topk(final_fitness, k=10).indices
        return pop[best_indices]

    graphite_idx = 0
    viscosity_idx = 1

    normalized_bounds = torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], dtype=torch.float64)
    best_y = train_y_yield[:, 0].max().item()

    init_candidates = run_GA_for_initial_candidates(
        gpr_model,
        normalized_bounds,
        best_y,
        scaler_input,
        graphite_idx,
        viscosity_idx,
        pop_size=20,
        generations=50
    )

    init_candidates_denorm = scaler_input.inverse_transform(init_candidates.numpy())
    st.subheader("GA 추천 조성")
    st.dataframe(pd.DataFrame(init_candidates_denorm, columns=X.columns))

    model_BO = SingleTaskGP(train_x, torch.tensor(np.hstack([train_y, X[["Graphite"]].values]), dtype=torch.double))
    mll_BO = ExactMarginalLogLikelihood(model_BO.likelihood, model_BO)
    fit_gpytorch_mll(mll_BO)

    train_y_hv = model_BO.train_targets.clone()
    train_y_hv[:, 1] = -train_y_hv[:, 1]

    ref_point = [0.0, -15.0, 20.0]
    partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=train_y_hv)
    acq_func = qExpectedHypervolumeImprovement(model=model_BO, ref_point=ref_point, partitioning=partitioning)

    initial_conditions = init_candidates.to(dtype=torch.double)
    candidate_scaled, _ = optimize_acqf(acq_func, bounds=normalized_bounds, q=1, num_restarts=len(initial_conditions),
                                        raw_samples=512, options={"batch_initial_conditions": initial_conditions})

    candidate_wt = scaler_input.inverse_transform(candidate_scaled.detach().cpu().numpy())[0]
    candidate_wt = candidate_wt / np.sum(candidate_wt) * 100

    st.subheader("최종 추천 조성 (qEHVI + GA)")
    for i, col in enumerate(X.columns):
        st.write(f"{col}: **{candidate_wt[i]:.2f} wt%**")
    st.write(f"**총합**: {np.sum(candidate_wt):.2f} wt%")

    pred_tensor = torch.tensor(scaler_input.transform(candidate_wt.reshape(1, -1)), dtype=torch.double)
    posterior = model_BO.posterior(pred_tensor)
    pred_mean = posterior.mean.detach().cpu().numpy()[0]

    st.write(f"**예측 Yield Stress**: {pred_mean[0]:.2f} Pa")
    st.write(f"**예측 Viscosity**: {-pred_mean[1]:.3f} Pa.s")
    st.write(f"**예측 Graphite wt%**: {pred_mean[2]:.2f} wt%")




    # 3D 파레토 시각화
    pareto_mask = is_non_dominated(train_y_hv)
    train_y_vis_plot = train_y_hv.clone()
    train_y_vis_plot[:, 1] = -train_y_vis_plot[:, 1]
    pareto_points = train_y_vis_plot[pareto_mask].numpy()

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(train_y_vis_plot[:, 1], train_y_vis_plot[:, 0], train_y_vis_plot[:, 2],
               color='gray', alpha=0.7, label='Data', s=30, depthshade=True)
    ax.scatter(pareto_points[:, 1], pareto_points[:, 0], pareto_points[:, 2],
               color='red', edgecolors='black', s=90, marker='o', depthshade=True, label='Pareto Front')
    ax.scatter(visc_pred, yield_pred, graphite_pred,
               color='blue', edgecolors='black', s=200, marker='^', label='Candidate')

    if len(pareto_points) >= 4:
        try:
            hull = ConvexHull(pareto_points)
            for simplex in hull.simplices:
                tri = pareto_points[simplex]
                ax.plot_trisurf(tri[:, 1], tri[:, 0], tri[:, 2],
                                color='pink', alpha=0.4, edgecolor='gray', linewidth=1.2)
        except Exception as e:
            st.warning(f"Convex Hull 실패: {e}")

    ax.set_xlabel("Viscosity [Pa.s] (↓)", fontsize=12, labelpad=10)
    ax.set_ylabel("Yield Stress [Pa] (↑)", fontsize=12, labelpad=10)
    ax.set_zlabel("Graphite wt% (↑)", fontsize=12, labelpad=15)
    ax.set_zlim(20, 40)
    ax.zaxis.set_ticks(np.arange(20, 45, 5))
    ax.view_init(elev=25, azim=135)
    ax.legend()
    ax.grid(True)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    st.pyplot(fig)

    # 5-Fold Cross-Validation
    st.subheader("\U0001F4CA 5-Fold Cross Validation (RMSE)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_dict = {"yield_stress": [], "viscosity": [], "graphite_wt%": []}
    X_tensor = torch.tensor(X_scaled, dtype=torch.double)
    Y_tensor = torch.tensor(Y_raw_extended, dtype=torch.double)

    for train_idx, test_idx in kf.split(X_tensor):
        X_train, Y_train = X_tensor[train_idx], Y_tensor[train_idx]
        X_test, Y_test = X_tensor[test_idx], Y_tensor[test_idx]
        Y_train_mod = Y_train.clone()
        Y_train_mod[:, 1] = -Y_train_mod[:, 1]
        model_cv = SingleTaskGP(X_train, Y_train_mod)
        mll_cv = ExactMarginalLogLikelihood(model_cv.likelihood, model_cv)
        fit_gpytorch_mll(mll_cv)
        preds = model_cv.posterior(X_test).mean.detach().numpy()
        preds[:, 1] = -preds[:, 1]
        for i, target in enumerate(["yield_stress", "viscosity", "graphite_wt%"]):
            rmse = np.sqrt(mean_squared_error(Y_test[:, i], preds[:, i]))
            rmse_dict[target].append(rmse)

    rmse_df = pd.DataFrame({
        "Target": list(rmse_dict.keys()),
        "RMSE Mean": [np.mean(v) for v in rmse_dict.values()],
        "RMSE Std": [np.std(v) for v in rmse_dict.values()]
    })
    st.dataframe(rmse_df)

    # 하이퍼볼륨 추적
    hv_log_path = "hv_tracking_3obj.csv"
    hv_list = []
    ref_point_fixed = torch.tensor([0.0, -15.0, 20.0], dtype=torch.double)

    for i in range(1, len(train_y_hv) + 1):
        current_Y = train_y_hv[:i].clone()
        try:
            bd = DominatedPartitioning(ref_point=ref_point_fixed, Y=current_Y.clone().detach())
            hv = bd.compute_hypervolume().item()
        except Exception as e:
            hv = float('nan')
            st.warning(f"{i}번째 계산 중 에러: {e}")
        hv_list.append({"iteration": i, "hv": hv})

    hv_df = pd.DataFrame(hv_list)
    hv_df.to_csv(hv_log_path, index=False)

    fig_hv, ax_hv = plt.subplots(figsize=(8, 4))
    ax_hv.plot(hv_df["iteration"], hv_df["hv"], marker='o')
    ax_hv.set_xlabel("Iteration")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("3D Hypervolume Progress Over Iterations")
    ax_hv.set_xticks(np.arange(1, hv_df["iteration"].max() + 1, 3))
    ax_hv.grid(True)
    st.pyplot(fig_hv)