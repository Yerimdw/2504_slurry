# C:/YR/PycharmProjects/250507_framework_git_mod.py

import streamlit as st
import pandas as pd
import numpy as np
from pyDOE import lhs

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement

# Sidebar 구성
st.set_page_config(page_title="Data-driven optimization of anode slurry", layout="wide")
st.sidebar.title("Optimization steps")
step = st.sidebar.radio("Select a step", [
    "A. DOE(LHS)",
    "B. Feature importance",
    "C. Initial candidates (GA)",
    "D. Bayesian optimization (GPR)",
    "E. Optimal composition for coating"
])

# Data lodaing 함수 정의

# def data_load


# step 별 함수 정의
def run_LHS():
    st.title("LHS를 통한 초기 실험 조성 추천")
    left_col, right_col = st.columns([1, 3])  # 1:3 비율

    with left_col:
        st.markdown("### Bound (wt%)") # ###는 bold체

        def input_range(label, key_min, key_max, default_min, default_max, step=0.1):
            col1, col2 = st.columns(2)
            min_val = col1.number_input(f"{label} 최소", key=key_min, value=default_min, step=step)
            max_val = col2.number_input(f"{label} 최대", key=key_max, value=default_max, step=step)
            return min_val, max_val

        graphite_min, graphite_max = input_range("Graphite", "graphite_min", "graphite_max", 20.0, 40.0, step=0.5)
        cb_min, cb_max = input_range("Carbon Black", "cb_min", "cb_max", 1.0, 5.0)
        cmc_min, cmc_max = input_range("CMC", "cmc_min", "cmc_max", 0.5, 2.0)
        sbr_min, sbr_max = input_range("SBR", "sbr_min", "sbr_max", 1.0, 4.0)
        solvent_min, solvent_max = input_range("Solvent", "solvent_min", "solvent_max", 60.0, 80.0)

        n_samples = st.number_input("몇가지 조성을 추천할까요?", min_value=1, max_value=100, value=15, step=1)
        st.markdown("총 Binder는 2~5 wt% CMC:SBR <= 1:3")

    with right_col:
        st.markdown(f"### 추천 조성 {n_samples}개 (wt%)")

        bounds = {
            'carbon_black': (cb_min, cb_max),
            'cmc': (cmc_min, cmc_max),
            'sbr': (sbr_min, sbr_max),
            'solvent': (solvent_min, solvent_max)
        }
        keys = list(bounds.keys())

        valid_samples = []
        max_trials = 1000
        trial_count = 0

        while len(valid_samples) < n_samples and trial_count < max_trials:
            trial_count += 1
            lhs_sample = lhs(len(bounds), samples=1)
            sample_dict = {}

            for i, key in enumerate(keys):
                min_val, max_val = bounds[key]
                val = lhs_sample[0, i] * (max_val - min_val) + min_val
                sample_dict[key + "_wt%"] = val
            # binder 조건 확인
            cmc = sample_dict["cmc_wt%"]
            sbr = sample_dict["sbr_wt%"]
            total_binder = cmc + sbr
            cmc_to_sbr = sbr / cmc if cmc != 0 else 0

            if not (sbr > cmc and sbr <= 3 * cmc and 2 <= total_binder <= 5 and cmc_to_sbr <= 3):
                continue

            graphite = (
                    100
                    - sample_dict["carbon_black_wt%"]
                    - sample_dict["cmc_wt%"]
                    - sample_dict["sbr_wt%"]
                    - sample_dict["solvent_wt%"]
            )

            if 0 < graphite <= 100 and graphite_min <= graphite <= graphite_max:
                sample_dict["graphite_wt%"] = graphite
                sample_dict["cmc:sbr"] = f"1:{cmc_to_sbr:.2f}"
                valid_samples.append(sample_dict)

        if len(valid_samples) < n_samples:
            st.warning(f"설정한 범위를 만족하는 조성이 {len(valid_samples)}개입니다.범위를 넓혀보세요.")
        else:
            st.success(f"{n_samples}개의 조건 만족 조성을 성공적으로 생성했습니다.")

        if valid_samples:
            df = pd.DataFrame(valid_samples)
            df.rename(columns={
                "carbon_black_wt%": "Carbon Black",
                "cmc_wt%": "CMC",
                "sbr_wt%": "SBR",
                "solvent_wt%": "Solvent",
                "graphite_wt%": "Graphite",
                "cmc:sbr": "CMC:SBR Ratio"
            }, inplace=True)

            column_order = [
                "Graphite",
                "Carbon Black",
                "CMC",
                "SBR",
                "Solvent",
                "CMC:SBR Ratio"
            ]
            df = df[column_order]
            df.index = np.arange(1, len(df) + 1)
            float_cols = df.select_dtypes(include=['float']).columns
            st.dataframe(df.style.format({col: "{:.4f}" for col in float_cols}), height=570)
        else:
            st.error("조건을 만족하는 조성이 없습니다.")


def run_RF():
    st.title("Random forest를 통한 feature importance 분석 (Summary plots)")
    url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
    df=pd.read_csv(url)
    X = df[["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]]
    Y = df[["yield stress", "n", "K", "viscosity"]]

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    results = {col: [] for col in Y.columns}

    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        for col in Y.columns:
            y_train = Y[col].iloc[train_idx]
            y_test = Y[col].iloc[test_idx]

            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            results[col].append(mae)

    cols = st.columns(2)

    for idx, col_name in enumerate(Y.columns):
        # 줄 바꿔야 할 때(짝수 인덱스일 때) 새로운 cols 선언
        if idx % 2 == 0:
            cols = st.columns(2)

        with cols[idx % 2]:  # 0번 또는 1번 column에 배치
            st.markdown(f"#### **[{col_name}]**")
            model = RandomForestRegressor(random_state=42)
            model.fit(X, Y[col_name])

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            fig, ax = plt.subplots(figsize=(3, 3))
            shap.summary_plot(shap_values, X, show=False)
            # plt.title(f"{col_name}", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

    if idx % 2 == 1:
        st.markdown("---")

    st.subheader("Parameter에 따른 상대 오차")
    for col in results:
        mae = np.mean(results[col])
        mean_actual = np.mean(Y[col])
        relative_error = mae / mean_actual * 100
        st.write(f"**{col}**: {relative_error:.2f}%")


def run_GA():
    st.title("Genetic Algorithm을 통한 추가 실험 조성 추천 (3가지)")

    # 데이터 불러오기 - GitHub용
    url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
    df=pd.read_csv(url)
    X = df[["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]]
    Y = df[["yield stress", "viscosity"]]  #일단 yield stress만

    # # 데이터 불러오기 - Streamlit용
    # df = pd.read_excel("C:/YR/1_Experiment/LHS_slurry_data.xlsx", engine="openpyxl")
    # X = df[["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]]
    # Y = df[["yield stress", "n", "K", "viscosity"]]

    # 전처리
    scaler_input = MinMaxScaler()
    input_data_scaled = scaler_input.fit_transform(X)
    train_x = torch.tensor(input_data_scaled, dtype=torch.float64)
    train_y = torch.tensor(Y.values, dtype=torch.float64)
    
    # SingleTaskGP에 yields stress 하나만 전달 (단일출력밖에 못하니까)
    train_y = torch.tensor(Y["yield stress"].values, dtype=torch.float64).unsqueeze(-1)

    # GP 모델 학습
    gpr_model = SingleTaskGP(train_x, train_y) # SingleTaskGP : yield stress만 학습
    mll = ExactMarginalLogLikelihood(gpr_model.likelihood, gpr_model)
    fit_gpytorch_mll(mll)

    # EI 계산 함수
    def expected_improvement(x_tensor, model, best_f):
        ei = ExpectedImprovement(model=model, best_f=best_f)
        return ei(x_tensor.unsqueeze(0)).item()

    ## 여기 수정
    def fitness(x_tensor, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx):
        ei = ExpectedImprovement(model=model, best_f=best_f)
        ei_val = ei(x_tensor.unsqueeze(0)).item()

        # 역정규화된 조성 (wt%)
        x_denorm = scaler.inverse_transform(x_tensor.unsqueeze(0).numpy()).squeeze()
        graphite = x_denorm[graphite_idx]
        viscosity = x_denorm[viscosity_idx]

        # 📌 graphite 정규화 (예: 20~40 wt%)
        graphite_min = 20.0
        graphite_max = 40.0
        graphite_norm = (graphite - graphite_min) / (graphite_max - graphite_min)

        # 📌 viscosity penalty (0.5 Pa.s에 가까울수록 좋음)
        if viscosity < 0.1 or viscosity > 1.5:
            viscosity_penalty = -1.0
        else:
            viscosity_target = 0.5
            viscosity_penalty = -abs(viscosity - viscosity_target)

        # 최종 가중합 점수 계산
        score = ( 1.0 * ei_val + 1.0 * viscosity_penalty + 0.5 * graphite_norm )
        return score


    # GA를 통한 초기 후보 추천
    def run_GA_for_initial_candidates(model, bounds_tensor, best_f, scaler, graphite_idx, viscosity_idx, pop_size=20,
                                      generations=30):
        dim = bounds_tensor.shape[1]
        pop = torch.rand(pop_size, dim, dtype=torch.float64)

        for gen in range(generations):
            # fitness 기반 평가
            fitness_vals = torch.tensor([
                fitness(x, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx)
                for x in pop
            ], dtype=torch.float64).squeeze()

            # 상위 절반 선택
            topk = torch.topk(fitness_vals, k=pop_size // 2)
            parents = pop[topk.indices]

            # 자식 생성
            children = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[(i + 1) % len(parents)]
                alpha = torch.rand(1).item()
                child = alpha * p1 + (1 - alpha) * p2
                mutation = 0.05 * torch.randn(dim, dtype=torch.float64)
                child += mutation
                child = torch.clamp(child, 0.0, 1.0)
                children.append(child)

            # 다음 세대 업데이트
            pop = torch.vstack((parents, torch.stack(children)))

        # 최종 평가 후 상위 3개 선택
        final_fitness = torch.tensor([
            fitness(x, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx)
            for x in pop
        ], dtype=torch.float64).squeeze()
        best_indices = torch.topk(final_fitness, k=3).indices
        return pop[best_indices]

    # Graphite와 viscosity 인덱스 지정
    graphite_idx = 0  # Graphite 열 인덱스
    viscosity_idx = 1  # viscosity 인덱스 (Y의 두 번째 열)

    # GA 실행
    normalized_bounds = torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], dtype=torch.float64)
    best_y = train_y[:, 0].max().item()  # yield stress 최대값

    init_candidates = run_GA_for_initial_candidates(
        gpr_model,
        normalized_bounds,
        best_y,
        scaler_input,
        graphite_idx,
        viscosity_idx,
        pop_size=20,
        generations=50  # 테스트 시 5로 줄여도 됨
    )

    # 역정규화 및 출력
    init_candidates_denorm = scaler_input.inverse_transform(init_candidates.numpy())
    st.dataframe(pd.DataFrame(init_candidates_denorm, columns=X.columns))
    
    st.write("추천 기준: Yield stress의 EI 값이 높고, Viscosity가 0.5 Pa.s 와 가장 가깝고, Graphite가 최대한 많이 함량")



def run_BO():
    st.title("Slurry 조성 최적화: qEHVI 기반 3목적 Bayesian Optimization")

    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import ConvexHull

    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.utils.multi_objective.box_decompositions import (
        NondominatedPartitioning,
        DominatedPartitioning
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.optim import optimize_acqf
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    # 데이터 불러오기 - GitHub용
    url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
    df=pd.read_csv(url)
    x_cols = ["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]
    y_cols = ["yield stress", "viscosity"]

    X_raw = df[x_cols].values
    Y_raw = df[y_cols].values
    graphite_idx = x_cols.index("Graphite")
    graphite_wt_values = X_raw[:, graphite_idx].reshape(-1, 1)
    Y_raw_extended = np.hstack([Y_raw, graphite_wt_values])

    x_scaler = MinMaxScaler()
    x_scaler.fit(X_raw)
    X_scaled = x_scaler.transform(X_raw)

    train_x = torch.tensor(X_scaled, dtype=torch.double)
    train_y = torch.tensor(Y_raw_extended, dtype=torch.double)

    train_y_hv = train_y.clone()
    train_y_hv[:, 1] = -train_y_hv[:, 1]  # viscosity는 낮을수록 좋음

    ref_point = [0.0, -15.0, 20.0]
    partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=train_y_hv)

    model = SingleTaskGP(train_x, train_y_hv)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    acq_func = qExpectedHypervolumeImprovement(model=model, ref_point=ref_point, partitioning=partitioning)

    # 최적화
    bounds = torch.tensor([[0.0] * len(x_cols), [1.0] * len(x_cols)], dtype=torch.double)
    candidate_scaled, _ = optimize_acqf(acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=128)
    candidate_wt = x_scaler.inverse_transform(candidate_scaled.detach().cpu().numpy())[0]
    candidate_wt = candidate_wt / np.sum(candidate_wt) * 100

    # 수치 및 예측치 표시
    st.subheader("최적 조성 (qEHVI 최적화, 3목적)")
    for col in x_cols:
        idx = x_cols.index(col)
        st.write(f"{col}: **{candidate_wt[idx]:.2f} wt%**")
    st.write(f"**총합**: {np.sum(candidate_wt):.2f} wt%")

    X_predict = x_scaler.transform(candidate_wt.reshape(1, -1))
    X_tensor = torch.tensor(X_predict, dtype=torch.double)
    posterior = model.posterior(X_tensor)
    pred_mean = posterior.mean.detach().cpu().numpy()[0]
    yield_pred = pred_mean[0]
    visc_pred = -pred_mean[1]
    graphite_pred = pred_mean[2]

    st.write(f"**예측 Yield Stress**: {yield_pred:.2f} Pa")
    st.write(f"**예측 Viscosity**: {visc_pred:.3f} Pa.s")
    st.write(f"**예측 Graphite wt%**: {graphite_pred:.2f} wt%")

    # 3D 파레토 시각화
    pareto_mask = is_non_dominated(train_y_hv)
    train_y_vis_plot = train_y_hv.clone()
    train_y_vis_plot[:, 1] = -train_y_vis_plot[:, 1]
    pareto_points = train_y_vis_plot[pareto_mask].numpy()

    fig = plt.figure(figsize=(8, 6))
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
    # plt.tight_layout()

    # tight_layout 제거하고 수동 조정
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    st.pyplot(fig)

    # 5-Fold Cross-Validation
    st.subheader("📊 5-Fold Cross Validation (RMSE)")
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

    # 하이퍼볼루머 추적
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
    ax_hv.set_xticks(np.arange(1, hv_df["iteration"].max() + 1, 1))
    ax_hv.grid(True)
    st.pyplot(fig_hv)


def run_ID():
    st.title("Inverse design을 통한 coating에 적합한 조성 추천")

    # 데이터 불러오기
    url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
    df=pd.read_csv(url)
    X = df[["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]]
    Y = df[["yield stress", "viscosity"]]  # 두 output 모두 사용

    # 입력 데이터 정규화
    scaler_input = MinMaxScaler()
    input_data_scaled = scaler_input.fit_transform(X)

    # output은 정규화하지 않음 (yield stress, viscosity 직접 예측할 것)

    train_x = torch.tensor(input_data_scaled, dtype=torch.float64)
    train_y = torch.tensor(Y.values, dtype=torch.float64)

    # GPR 모델 학습
    gpr_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gpr_model.likelihood, gpr_model)
    fit_gpytorch_mll(mll)

    # Bound 사전 정의
    bounds = {
        "Graphite": (20.0, 40.0),
        "Carbon Black": (1.0, 5.0),
        "CMC": (0.5, 2.0),
        "SBR": (1.0, 4.0),
        "Solvent": (60.0, 80.0)
    }
    bound_keys = list(bounds.keys())

    # 후보 생성 (n_samples개)
    n_samples = 5000
    lhs_samples = np.random.rand(n_samples, len(bounds))  # 0~1 랜덤

    # bound 적용
    random_samples = np.zeros_like(lhs_samples)
    for i, key in enumerate(bound_keys):
        min_val, max_val = bounds[key]
        random_samples[:, i] = lhs_samples[:, i] * (max_val - min_val) + min_val

    # 조성 합 100%로 정규화 (Graphite까지 포함)
    total = np.sum(random_samples, axis=1).reshape(-1, 1)
    normalized_samples = random_samples / total * 100

    # 후보를 원래 조성 범위로 역변환
    random_samples_denorm = scaler_input.inverse_transform(random_samples)

    # 각 조성의 합이 100%에 가깝게 만들기 (비율 맞추기)
    random_samples_denorm = (random_samples_denorm.T / random_samples_denorm.sum(axis=1)).T * 100

    # Tensor로 변환 후 모델 예측
    candidate_x = torch.tensor(scaler_input.transform(random_samples_denorm), dtype=torch.float64)
    pred_y = gpr_model.posterior(candidate_x).mean.detach().numpy()

    pred_yield_stress = pred_y[:, 0]
    pred_viscosity = pred_y[:, 1]

    # 조건 필터링: viscosity 0.5 ~ 1.5
    valid_idx = (pred_viscosity >= 0.5) & (pred_viscosity <= 1.5)

    if np.sum(valid_idx) == 0:
        st.error("조건을 만족하는 후보가 없습니다. 후보 수를 늘리거나 조건을 완화해보세요.")
        return

    filtered_candidates = random_samples_denorm[valid_idx]
    filtered_yield_stress = pred_yield_stress[valid_idx]

    # yield stress가 가장 높은 조성 선택
    best_idx = np.argmax(filtered_yield_stress)
    best_composition = filtered_candidates[best_idx]

    # 결과 출력
    result_df = pd.DataFrame([best_composition], columns=["Graphite", "Carbon Black", "CMC", "SBR", "Solvent"])
    result_df.index = ["추천 조성"]

    st.subheader("Inverse design 결과")
    st.dataframe(result_df.style.format("{:.2f}"))

    st.write("- viscosity 0.5~1.5 사이")
    st.write("- yield stress 최대")

    st.write("- 만들어둔 Surrogate model(GPR)에 Random Input data 5000개를 학습시켜서 예측값 중 조건에 맞는 것을 추천함)")





# 페이지마다 함수 실행
if step == "A. DOE(LHS)":
    run_LHS()

elif step == "B. Feature importance":
    run_RF()

elif step == "C. Initial candidates (GA)":
    run_GA()

elif step == "D. Bayesian optimization (GPR)":
    run_BO()

elif step == "E. Optimal composition for coating":
    run_ID()

