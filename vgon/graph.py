import matplotlib.pyplot as plt

def plot(data1,data2):    
    # データの生成
    data_num = len(data1)
    x = range(data_num)
    grad_x = range(2, data_num)
    fig, ax1 = plt.subplots()

    # ax1とx軸を共有する新たなAxesを作成
    ax2 = ax1.twinx()

    # グラフの描画
    ax2.plot(x, data1, 'b-', label="loss")  # 青色の破線
    ax1.plot(x, data2, 'g--', label="grad")  # 緑色の実線
    # ax2.set_ylim(0.0,0.2)

    # グラフのタイトルとラベルの設定
    ax1.set_title("loss and grad")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("grad", color='g')
    ax2.set_ylabel("loss", color='b')
    ax2.set_ylim(-1)
    ax1.set_ylim(0)

    # 凡例の表示
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')

    # グラフの表示
    plt.show()