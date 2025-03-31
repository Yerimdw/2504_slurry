#%%
import streamlit as st
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
    # Normalization
from sklearn.preprocessing import MinMaxScaler
    # define GPR model&function
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
    # learning
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
    # 기대함수 정의 & 최적화
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
    #시각화
import matplotlib.pyplot as plt
import numpy as np
#%%
st.set_page_config(page_title="Bayesian Optimization for Anode Slurry",page_icon="C:/Users/ludia/PycharmProjects/250320_Bayesian_Optimization/image/battery.png", layout="wide")
st.title("Bayesian Optimization for Anode Slurry")
import pandas as pd
file_path = "C:/Users/ludia/PycharmProjects/250320_Bayesian_Optimization/BO_slurry_Data.xlsx"
df = pd.read_excel(file_path, usecols='B:F', skiprows=[0,1,2], header=0, engine='openpyxl')
raw_data=df.values
# print(raw_data)
df = pd.read_excel(file_path, usecols='B:E', skiprows=[0], nrows=2, header=None, engine='openpyxl')
bounds=df.values
# print(bounds)

input_data=raw_data[:,0:4]
score=raw_data[:,4:]
#%%

# 버튼 클릭 시 실행
if st.button("Candidate를 추천받으려면 버튼을 누르세요."):


    scaler_input = MinMaxScaler()
    input_data_scaled = scaler_input.fit_transform(input_data)
    # scaler_input = StandardScaler()
    # input_data_scaled = scaler_input.fit_transform(input_data)

    # print(input_data_scaled)
    #%%
    input_data_scaled=torch.from_numpy(input_data_scaled).double()
    score=torch.from_numpy(score).double()
    bounds=torch.from_numpy(bounds).double()

    X_train=input_data_scaled
    Y_train=score
    #%%
    likelihood=GaussianLikelihood()
    model=SingleTaskGP(X_train, Y_train,likelihood=likelihood)
    mll=ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)
    #%%
    ei = ExpectedImprovement(model=model, best_f=Y_train.max(), maximize=True)
    candidate, _ = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20
    )
    # print(candidate)
    #%%
    candidate_values = candidate.numpy()[0]
    columns = ["Carbon Black", "Binder", "Solvent", "Graphite"]
    table_data = [
        ["Composition"] + columns,
        ["(g)"] + candidate_values.tolist()
    ]

    # 위의 2차원 리스트를 DataFrame으로 만들기
    df_table = pd.DataFrame(table_data)

    # 2행의 숫자들(두 번째 행, 열 인덱스 1~끝)을 소수점 아래 4자리로 포맷팅
    for col in df_table.columns[1:]:
        # df_table.iloc[1, col]는 두 번째 행의 각 셀
        df_table.iloc[1, col] = "{:.4f}".format(df_table.iloc[1, col])

    # Pandas Styler를 사용하여 텍스트 가운데 정렬 및 열 너비 고정
    styled_df = (
        df_table.style
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {"selector": "th", "props": [("min-width", "120px"), ("max-width", "120px"), ("text-align", "center")]},
            {"selector": "td", "props": [("min-width", "120px"), ("max-width", "120px"), ("text-align", "center")]}
        ])
    )

    st.title("Candidate for Slurry Composition")
    st.table(styled_df)


    cb = pd.read_excel(file_path, usecols='B', skiprows=[0,1,2], header=0, engine='openpyxl')
    ys = pd.read_excel(file_path, usecols='F', skiprows=[0,1,2], header=0, engine='openpyxl')

    # 열을 NumPy 배열로 변환 (1차원)
    cb = cb.to_numpy().flatten()
    ys = ys.to_numpy().flatten()

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(cb, ys, color="blue")
    ax.scatter(cb, ys, color="black", label="Experimental data", s=20)

    candidate_cb = candidate[0, 0].item()
    candidate_ys = np.interp(candidate_cb, cb, ys)
    ax.scatter(candidate_cb, candidate_ys, color="red", s=20, label="Candidate", zorder=10)

    ax.set_title("Carbon Black vs Yields stress")
    ax.set_xlabel("Carbon Black [g]")
    ax.set_ylabel("Yield stress [Pa]")
    ax.legend()
    ax.grid(True)
    # plt.show()
    st.pyplot(fig)