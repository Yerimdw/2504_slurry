# C:/YR/PycharmProjects/2504_slurry/250424_framework.py
import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import lhs

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
    "C. Initial candidates(GA)",
    "D. Bayesian optimization (GPR)",
    "E. Optimal composition for coating"
])


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
    df = pd.read_excel("C:/YR/1_Experiment/LHS_slurry_data.xlsx", engine="openpyxl")
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

    # 데이터 불러오기
    df = pd.read_excel("C:/YR/1_Experiment/LHS_slurry_data.xlsx", engine="openpyxl")
    X = df[["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]]
    Y = df[["yield stress"]]  #일단 yield stress만

    # 전처리
    scaler_input = MinMaxScaler()
    input_data_scaled = scaler_input.fit_transform(X)
    train_x = torch.tensor(input_data_scaled, dtype=torch.float64)
    train_y = torch.tensor(Y.values, dtype=torch.float64)

    # GP 모델 학습
    gp_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)

    # EI 계산 함수
    def expected_improvement(x_tensor, model, best_f):
        ei = ExpectedImprovement(model=model, best_f=best_f)
        return ei(x_tensor.unsqueeze(0))

    # GA를 통한 초기 후보 추천
    def run_GA_for_initial_candidates(model, bounds_tensor, best_f, pop_size=20, generations=30):
        dim = bounds_tensor.shape[1]
        pop = torch.rand(pop_size, dim, dtype=torch.float64)

        for _ in range(generations):
            ei_vals = torch.tensor([expected_improvement(x, model, best_f) for x in pop], dtype=torch.float64).squeeze()
            topk = torch.topk(ei_vals, k=pop_size // 2)
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

        final_ei = torch.tensor([expected_improvement(x, model, best_f) for x in pop], dtype=torch.float64).squeeze()
        best_indices = torch.topk(final_ei, k=3).indices
        return pop[best_indices]

    # GA 실행
    normalized_bounds = torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], dtype=torch.float64)
    best_y = train_y.max().item()
    init_candidates = run_GA_for_initial_candidates(gp_model, normalized_bounds, best_y, pop_size=20, generations=50)

    # 역정규화하여 출력
    init_candidates_denorm = scaler_input.inverse_transform(init_candidates.numpy())

    st.dataframe(pd.DataFrame(init_candidates_denorm, columns=X.columns))

    st.write("- 현재 단일 output 최적화(yield stress가 가장 클 것 같은 조성을 추천)")
    st.write("- NSGA-II라는 다목적 최적화 알고리즘")


# def run_BO():


def run_ID():
    st.title("Inverse design을 통한 coating에 적합한 조성 추천")

    # 데이터 불러오기
    df = pd.read_excel("C:/YR/1_Experiment/LHS_slurry_data.xlsx", engine="openpyxl")
    X = df[["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]]
    Y = df[["yield stress", "viscosity"]]  # 두 output 모두 사용

    # 입력 데이터 정규화
    scaler_input = MinMaxScaler()
    input_data_scaled = scaler_input.fit_transform(X)

    # output은 정규화하지 않음 (yield stress, viscosity 직접 예측할 것)

    train_x = torch.tensor(input_data_scaled, dtype=torch.float64)
    train_y = torch.tensor(Y.values, dtype=torch.float64)

    # GPR 모델 학습
    gp_model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
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
    pred_y = gp_model.posterior(candidate_x).mean.detach().numpy()

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

elif step == "C. Initial candidates(GA)":
    run_GA()

elif step == "D. Bayesian optimization (GPR)":
    run_BO()

elif step == "E. Optimal composition for coating":
    run_ID()

