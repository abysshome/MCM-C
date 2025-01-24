# predict_with_variance_plot.py

import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt
import sys
import random

# 配置 Matplotlib 以支持中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用 SimHei 字体显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# 定义滚动预测函数
def recursive_forecast(model_path, input_data, steps):
    """
    使用保存的模型进行滚动预测。
    """
    # 加载模型
    model = load_model(model_path)

    predictions = []
    current_input = input_data.copy()

    for _ in range(steps):
        # 预测下一个时间步
        pred = model.predict(current_input, verbose=0)

        # 将预测值保存
        predictions.append(pred[0, 0])

        # 使用当前预测值更新输入
        current_input = np.roll(current_input, -1)
        current_input[0, -1, 0] = pred

    return np.array(predictions)


# 定义处理每个起始点和预测长度的函数
def process_forecast(args):
    """
    处理单个起始点和预测长度的预测任务。
    """
    (model_path, start_index, X_test, y_test, time_step, pred_length) = args

    initial_input = X_test[start_index].reshape(1, time_step, 1)
    y_test_actual_segment = y_test[start_index : start_index + pred_length]

    if len(y_test_actual_segment) < pred_length:
        # 如果实际数据不足，返回高误差
        return {"pred_length": pred_length, "mse": float("inf")}

    # 进行递归预测
    pred = recursive_forecast(model_path, initial_input, pred_length)

    # 计算均方误差
    mse = mean_squared_error(y_test_actual_segment, pred)

    return {"pred_length": pred_length, "mse": mse}


# 定义选择多个随机起始索引的函数
def select_random_start_indices(total_length, N, time_step, max_steps):
    """
    随机选择多个起始索引，确保每个起始点后有足够的步数进行预测。
    """
    available_length = total_length - time_step - max_steps
    if available_length <= 0:
        raise ValueError("测试集长度不足以进行所需的预测步数。")

    # 随机选择 N 个起始点
    start_indices = random.sample(range(time_step, available_length), N)
    return start_indices


# 定义多进程预测函数
def run_multiprocessing_forecast(
    model_path, start_indices, X_test, y_test, time_step, pred_length
):
    """
    使用多进程进行预测，并返回所有结果。
    """
    # 准备参数列表
    args_list = [
        (model_path, idx, X_test, y_test, time_step, pred_length)
        for idx in start_indices
    ]

    # 获取 CPU 核心数量
    cpu_count = multiprocessing.cpu_count()
    print(f"使用 {cpu_count} 个进程进行预测...")

    # 使用 ProcessPoolExecutor 进行多进程预测
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        results = list(executor.map(process_forecast, args_list))

    return results


# 定义计算不同预测长度下误差的函数
def compute_variance_over_time(
    model_path, X_test, y_test, time_step, N, max_pred_length
):
    """
    计算不同预测长度下预测误差的均值。
    """
    # 随机选择 N 个起始点
    start_indices = select_random_start_indices(
        len(X_test), N, time_step, max_pred_length
    )
    print(f"随机选择的起始索引: {start_indices}")

    # 获取所有预测长度
    pred_lengths = list(range(1, max_pred_length + 1))  # 1小时到72小时

    variance_means = []

    for pred_length in pred_lengths:
        print(f"计算预测长度: {pred_length} 小时")
        results = run_multiprocessing_forecast(
            model_path, start_indices, X_test, y_test, time_step, pred_length
        )
        # 过滤掉因数据不足返回的无穷大误差
        valid_results = [res["mse"] for res in results if res["mse"] != float("inf")]
        if valid_results:
            mean_mse = np.mean(valid_results)
        else:
            mean_mse = None  # 或者设为 np.nan
        variance_means.append(mean_mse)
        print(f"预测长度 {pred_length} 小时的平均MSE: {mean_mse}")

    return pred_lengths, variance_means


# 主函数
def main():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # 加载测试数据
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    time_step = 24  # 假设时间步为24小时
    N = 100  # 随机选择20个起始点
    max_pred_length = 48  # 最大预测长度为48小时（2天）

    # 模型保存路径
    model_path = "lstm_model.h5"

    # 计算不同预测长度下的平均MSE
    pred_lengths, variance_means = compute_variance_over_time(
        model_path, X_test, y_test, time_step, N, max_pred_length
    )
    print("不同预测长度下的平均MSE计算完成！")

    # 绘制预测长度与平均MSE的关系图
    plt.figure(figsize=(12, 6))
    plt.plot(pred_lengths, variance_means, marker="o", linestyle="-", color="b")
    plt.title("预测时间长度与预测误差方差均值关系图")
    plt.xlabel("预测时间长度（小时）")
    plt.ylabel("预测误差均值（MSE）")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
