# C:/YR/PycharmProjects/250512_PINN.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

st.title("PINN을 통한 1차원 비정상 열전도 방정식 계산")

# -----------------------------
# 사용자 입력
# -----------------------------
L = st.number_input("두께 L (cm)", value=10)
T_max = st.number_input("시뮬레이션 시간 (s)", value=180)
alpha = st.number_input("열확산계수 α (m²/s)", value=0.0000117, format="%.8f") # 구리

T_left = st.number_input("왼쪽 경계조건 온도 (℃)", value=25.0)
T_right = st.number_input("오른쪽 경계조건 온도 (℃)", value=25.0)
initial_expr = st.text_input("초기 온도 분포 함수 (x에 대한 표현)", value="100 * torch.ones_like(x)")

# 사용자 입력: 중앙 목표 온도
target_temperature = st.number_input("중앙 목표 온도 (℃)", value=50.0)

# 초기 온도 함수 생성
def initial_temperature(x):
    return eval(initial_expr)

# -----------------------------
# PINN 정의
# -----------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, xt):
        return self.net(xt)

def pde_residual(model, xt):
    xt.requires_grad_(True)
    T_pred = model(xt)
    grads = torch.autograd.grad(T_pred, xt, torch.ones_like(T_pred), create_graph=True)[0]
    T_x = grads[:, 0:1]
    T_t = grads[:, 1:2]
    grads2 = torch.autograd.grad(T_x, xt, torch.ones_like(T_x), create_graph=True)[0]
    T_xx = grads2[:, 0:1]
    return T_t - alpha * T_xx

# -----------------------------
# 학습 데이터 생성
# -----------------------------
n_collocation, n_boundary, n_initial = 5000, 500, 500
x_collocation = torch.rand(n_collocation, 1) * L
t_collocation = torch.rand(n_collocation, 1) * T_max
xt_collocation = torch.cat([x_collocation, t_collocation], dim=1)

x_initial = torch.rand(n_initial, 1) * L
t_initial = torch.zeros_like(x_initial)
T_initial = initial_temperature(x_initial)
xt_initial = torch.cat([x_initial, t_initial], dim=1)

x_left = torch.zeros(n_boundary, 1)
t_left = torch.rand(n_boundary, 1) * T_max
T_left_tensor = torch.full_like(x_left, T_left)
xt_left = torch.cat([x_left, t_left], dim=1)

x_right = torch.full((n_boundary, 1), L)
t_right = torch.rand(n_boundary, 1) * T_max
T_right_tensor = torch.full_like(x_right, T_right)
xt_right = torch.cat([x_right, t_right], dim=1)

# -----------------------------
# 학습 시작 버튼
# -----------------------------
if st.button("PINN 학습 시작"):
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3000):
        optimizer.zero_grad()
        residual = pde_residual(model, xt_collocation)
        loss_pde = torch.mean(residual**2)
        loss_ic = torch.mean((model(xt_initial) - T_initial) ** 2)
        loss_bc = torch.mean((model(xt_left) - T_left_tensor) ** 2) + torch.mean((model(xt_right) - T_right_tensor) ** 2)
        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            st.write(f'Epoch {epoch}: Loss = {loss.item():.5e}')


    # 예측 및 결과 출력
    x_plot = torch.linspace(0, L, 100).reshape(-1, 1)
    t_plot = torch.full_like(x_plot, T_max)
    xt_plot = torch.cat([x_plot, t_plot], dim=1)
    T_plot = model(xt_plot).detach().numpy()

    import io
    import PIL.Image

    fig, ax = plt.subplots()
    ax.plot(x_plot.numpy(), T_plot)
    ax.set_xlabel("x (m)")
    ax.set_ylabel(f"T at t={T_max} s")
    ax.set_title("1D Heat Conduction Prediction by PINN")
    ax.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(PIL.Image.open(buf), caption="PINN 해석 결과")
# if st.button("슬릿 중앙 온도 도달 시간 계산"):
#     st.write("슬릿 중앙(T center) 온도가 목표 온도에 도달하는 시간 찾기...")

    # 찾을 시간 범위 (0 ~ T_max 사이 100개 시점 탐색)
    t_test = torch.linspace(0, T_max, 100).reshape(-1, 1)
    x_center = torch.full_like(t_test, L / 2)  # 중앙 위치 고정
    xt_test = torch.cat([x_center, t_test], dim=1)

    # PINN으로 예측
    T_pred_center = model(xt_test).detach().numpy().flatten()

    # 목표 온도와의 차이 계산
    diff = np.abs(T_pred_center - target_temperature)

    # 가장 가까운 시간 찾기
    min_index = np.argmin(diff)
    best_time = t_test[min_index].item()
    best_temperature = T_pred_center[min_index]

    st.write(f"중앙 온도가 약 {target_temperature:.1f}℃에 도달하는 시간: {best_time:.2f} s (예측 온도: {best_temperature:.2f} ℃)")

    # 결과 시각화

    fig, ax = plt.subplots()
    ax.plot(t_test.numpy().flatten(), T_pred_center, label="T center")
    ax.axhline(y=target_temperature, color='r', linestyle='--', label=f"T target {target_temperature}℃")
    ax.axvline(x=best_time, color='g', linestyle='--', label=f"predicted time {best_time:.2f} s")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("T center (℃)")
    ax.set_title("Temperature of center")
    ax.legend()
    st.pyplot(fig)


