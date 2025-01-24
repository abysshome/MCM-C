# predict.py

import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt
import sys

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


# 定义处理每个起始点的函数
def process_starting_point(args):
    """
    处理单个起始点的预测任务。
    """
    (model_path, start_index, X_test, y_test, time_step, day_length, week_length) = args

    initial_input = X_test[start_index].reshape(1, time_step, 1)
    y_test_actual_segment = y_test[start_index:]

    # 预测一天的数据
    day_predict = recursive_forecast(model_path, initial_input, day_length)
    day_actual = y_test_actual_segment[:day_length]
    day_error = mean_squared_error(day_actual, day_predict)

    # 预测一周的数据
    week_predict = recursive_forecast(model_path, initial_input, week_length)
    week_actual = y_test_actual_segment[:week_length]
    week_error = mean_squared_error(week_actual, week_predict)

    return {
        "start_index": start_index,
        "day_error": day_error,
        "day_predict": day_predict,
        "day_actual": day_actual,
        "week_error": week_error,
        "week_predict": week_predict,
        "week_actual": week_actual,
    }


# 定义选择多个起始索引的函数
def select_start_indices(total_length, N, time_step, max_steps):
    """
    选择多个起始索引，确保每个起始点后有足够的步数进行预测。
    """
    available_length = total_length - time_step - max_steps
    if available_length <= 0:
        raise ValueError("测试集长度不足以进行所需的预测步数。")
    start_indices = np.linspace(time_step, available_length, N).astype(int)
    return start_indices


# 定义多进程预测函数
def run_multiprocessing(
    model_path, start_indices, X_test, y_test, time_step, day_length, week_length
):
    """
    使用多进程进行预测，并返回所有结果。
    """
    # 准备参数列表
    args_list = [
        (model_path, idx, X_test, y_test, time_step, day_length, week_length)
        for idx in start_indices
    ]

    # 获取 CPU 核心数量
    cpu_count = multiprocessing.cpu_count()
    print(f"使用 {cpu_count} 个进程进行预测...")

    # 使用 ProcessPoolExecutor 进行多进程预测
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        results = list(executor.map(process_starting_point, args_list))

    return results


# 主函数
def main():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # 数据预处理和模型训练部分（假设之前已经完成并保存了模型）
    # 如果需要重新训练模型，可以在这里添加代码

    # 加载测试数据（假设在运行该脚本前已经处理好）
    # 这里假设 X_test 和 y_test 已经保存为 .npy 文件
    # 你需要根据实际情况调整数据加载方式

    # 示例：加载已处理好的测试数据
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    time_step = 24
    day_length = 24
    week_length = 24 * 7
    N = 100  # 选择100个起始点

    # 选择多个起始索引
    start_indices = select_start_indices(len(X_test), N, time_step, week_length)
    print(f"选择的起始索引: {start_indices}")

    # 模型保存路径
    model_path = "lstm_model.h5"

    # 运行多进程预测
    results = run_multiprocessing(
        model_path, start_indices, X_test, y_test, time_step, day_length, week_length
    )
    print("多进程预测完成！")

    # 从结果中选取一天和一周预测误差最小的
    best_day_result = min(results, key=lambda x: x["day_error"])
    best_week_result = min(results, key=lambda x: x["week_error"])

    print(
        f"最佳一天预测起始索引: {best_day_result['start_index']} - RMSE: {best_day_result['day_error']:.4f}"
    )
    print(
        f"最佳一周预测起始索引: {best_week_result['start_index']} - RMSE: {best_week_result['week_error']:.4f}"
    )

    # 可视化最佳预测结果
    plt.figure(figsize=(14, 10))

    # 可视化最佳一天的预测结果
    plt.subplot(2, 1, 1)
    plt.plot(best_day_result["day_actual"], label="实际值（一天）", linewidth=1)
    plt.plot(
        best_day_result["day_predict"],
        label="预测值（一天）",
        color="red",
        linewidth=1,
        alpha=0.6,
    )
    plt.title(
        f'一天的滚动预测 - 最佳结果 (起始索引: {best_day_result["start_index"]}) - RMSE: {best_day_result["day_error"]:.4f}'
    )
    plt.legend(prop={"family": "SimHei"})

    # 可视化最佳一周的预测结果
    plt.subplot(2, 1, 2)
    plt.plot(best_week_result["week_actual"], label="实际值（一周）", linewidth=1)
    plt.plot(
        best_week_result["week_predict"],
        label="预测值（一周）",
        color="red",
        linewidth=1,
        alpha=0.6,
    )
    plt.title(
        f'一周的滚动预测 - 最佳结果 (起始索引: {best_week_result["start_index"]}) - RMSE: {best_week_result["week_error"]:.4f}'
    )
    plt.legend(prop={"family": "SimHei"})

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
