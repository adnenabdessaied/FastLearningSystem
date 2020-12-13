import pickle
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import torch as T
from torch.autograd import Variable as var
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def generate_data(batch_size, length, size, device):
    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros(
        (batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    input_data = T.from_numpy(input_data)
    target_output = T.from_numpy(target_output)
    input_data = input_data.to(device)
    target_output = target_output.to(device)

    return var(input_data), var(target_output)


def plot_io_data_copy_task():
    i, o = generate_data(1, 3, 6, T.device("cpu"))
    i = np.transpose(i.numpy().squeeze(0))
    o = np.transpose(o.numpy().squeeze(0))
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.imshow(i)
    ax2.imshow(o)
    ax1.grid(b=False)
    ax2.grid(b=False)


def plot_analysis_data_DNC_QC():
    with open("collected_data/q_read_weights_1.pkl", "rb") as f:
        q_read_weights_1 = pickle.load(f)

    with open("collected_data/q_read_weights_2.pkl", "rb") as f:
        q_read_weights_2 = pickle.load(f)

    with open("collected_data/q_read_weights_3.pkl", "rb") as f:
        q_read_weights_3 = pickle.load(f)

    with open("collected_data/q_read_weights_4.pkl", "rb") as f:
        q_read_weights_4 = pickle.load(f)

    with open("collected_data/q_read_modes_1.pkl", "rb") as f:
        q_read_modes_1 = pickle.load(f)

    with open("collected_data/q_read_modes_2.pkl", "rb") as f:
        q_read_modes_2 = pickle.load(f)

    with open("collected_data/q_read_modes_3.pkl", "rb") as f:
        q_read_modes_3 = pickle.load(f)

    with open("collected_data/q_read_modes_4.pkl", "rb") as f:
        q_read_modes_4 = pickle.load(f)

    with open("collected_data/q_free_gates_1.pkl", "rb") as f:
        q_free_gates_1 = pickle.load(f)

    with open("collected_data/q_free_gates_2.pkl", "rb") as f:
        q_free_gates_2 = pickle.load(f)

    with open("collected_data/q_free_gates_3.pkl", "rb") as f:
        q_free_gates_3 = pickle.load(f)

    with open("collected_data/q_free_gates_4.pkl", "rb") as f:
        q_free_gates_4 = pickle.load(f)

    with open("collected_data/q_write_weights.pkl", "rb") as f:
        q_write_weights = pickle.load(f)

    with open("collected_data/q_mem_contrib.pkl", "rb") as f:
        q_mem_contrib = pickle.load(f)

    with open("collected_data/q_alloc_gates.pkl", "rb") as f:
        q_alloc_gates = pickle.load(f)

    with open("collected_data/q_write_gates.pkl", "rb") as f:
        q_write_gates = pickle.load(f)
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1, sharex=True)

    q_alloc_gates = np.squeeze(q_alloc_gates[0], axis=-1)
    q_write_gates = np.squeeze(q_write_gates[0], axis=-1)
    q_alloc_gates = np.squeeze(q_alloc_gates[0], axis=-1)
    q_write_gates = np.squeeze(q_write_gates[0], axis=-1)

    q_mem_contrib = q_mem_contrib[0]

    ax1.bar(np.array(range(0, 30)), np.ones_like(q_alloc_gates), color="y", label="content", width=0.4)
    ax1.bar(range(0, 30), q_alloc_gates, label="usage", width=0.4)
    ax1.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    ax1.set_xticks([0, 29])

    ax2.bar(np.array(range(0, 30)) - 0.15, np.ones_like(q_free_gates_1), color="brown", label="keep", width=0.1)
    ax2.bar(np.array(range(0, 30)) - 0.15, q_free_gates_1, color="seagreen", label="free", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax2.bar(np.array(range(0, 30)) - 0.05, np.ones_like(q_free_gates_2), color="brown", width=0.1)
    ax2.bar(np.array(range(0, 30)) - 0.05, q_free_gates_2, color="seagreen", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax2.bar(np.array(range(0, 30)) + 0.05, np.ones_like(q_free_gates_3), color="brown", width=0.1)
    ax2.bar(np.array(range(0, 30)) + 0.05, q_free_gates_3, color="seagreen", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax2.bar(np.array(range(0, 30)) + 0.15, np.ones_like(q_free_gates_4), color="brown", width=0.1)
    ax2.bar(np.array(range(0, 30)) + 0.15, q_free_gates_4, color="seagreen", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax3.bar(range(0, 30), np.ones_like(q_write_gates), color="mediumaquamarine", label="no write", width=0.4)
    ax3.bar(range(0, 30), q_write_gates, color="darksalmon", label="write", width=0.4)
    ax3.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax4.bar(range(0, 30), np.ones_like(q_read_modes_1[:, 1]), color="indigo", label="content", width=0.4)
    ax4.bar(range(0, 30), q_read_modes_1[:, 1] + q_read_modes_1[:, 0], color="#e24a33", label="forward", width=0.4)
    ax4.bar(range(0, 30), q_read_modes_1[:, 1], color="royalblue", label="backward", width=0.4)
    ax4.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax5.bar(range(0, 30), np.ones_like(q_read_modes_2[:, 1]), color="indigo", label="content", width=0.4)
    ax5.bar(range(0, 30), q_read_modes_2[:, 1] + q_read_modes_2[:, 0], color="#e24a33", label="forward", width=0.4)
    ax5.bar(range(0, 30), q_read_modes_2[:, 1], color="royalblue", label="backward", width=0.4)
    ax5.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax6.bar(range(0, 30), np.ones_like(q_read_modes_3[:, 1]), color="indigo", label="content", width=0.4)
    ax6.bar(range(0, 30), q_read_modes_3[:, 1] + q_read_modes_3[:, 0], color="#e24a33", label="forward", width=0.4)
    ax6.bar(range(0, 30), q_read_modes_3[:, 1], color="royalblue", label="backward", width=0.4)
    ax6.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax7.bar(range(0, 30), np.ones_like(q_read_modes_4[:, 1]), color="indigo", label="content", width=0.4)
    ax7.bar(range(0, 30), q_read_modes_4[:, 1] + q_read_modes_4[:, 0], color="#e24a33", label="forward", width=0.4)
    ax7.bar(range(0, 30), q_read_modes_4[:, 1], color="royalblue", label="backward", width=0.4)
    ax7.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax8.bar(range(0, 30), np.ones_like(q_mem_contrib), color="peru", label="controller", width=0.4)
    ax8.bar(range(0, 30), q_mem_contrib, color="indigo", label="memory", width=0.4)
    ax8.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    ax8.set_xlabel("time")

    ax1.set_ylabel("Allocation gates", rotation=0)
    ax1.yaxis.set_label_coords(-0.1,0.9)

    ax2.set_ylabel("Free gates", rotation=0)
    ax2.yaxis.set_label_coords(-0.1,0.9)

    ax3.set_ylabel("write gates", rotation=0)
    ax3.yaxis.set_label_coords(-0.1,0.9)

    ax4.set_ylabel("Read modes\nHead 1", rotation=0)
    ax4.yaxis.set_label_coords(-0.1,0.9)

    ax5.set_ylabel("Read modes\nHead 2", rotation=0)
    ax5.yaxis.set_label_coords(-0.1,0.9)

    ax6.set_ylabel("Read modes\nHead 3", rotation=0)
    ax6.yaxis.set_label_coords(-0.1,0.9)

    ax7.set_ylabel("Read modes\nHead 4", rotation=0)
    ax7.yaxis.set_label_coords(-0.1,0.9)

    ax8.set_ylabel("Output\nContributions", rotation=0)
    ax8.yaxis.set_label_coords(-0.1,0.9)

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    i1= ax1.imshow(q_write_weights)
    ax1.grid(b=False)

    i2 = ax2.imshow(q_read_weights_1)
    ax2.grid(b=False)

    i3 = ax3.imshow(q_read_weights_2)
    ax3.grid(b=False)

    i4 = ax4.imshow(q_read_weights_3)
    ax4.grid(b=False)

    i5 = ax5.imshow(q_read_weights_4)
    ax5.grid(b=False)
    ax5.set_xlabel("memory_cells")
    ax1.set_ylabel("time")
    ax2.set_ylabel("time")
    ax3.set_ylabel("time")
    ax4.set_ylabel("time")
    ax5.set_ylabel("time")
    ax1.set_title("Write memory locations")
    ax2.set_title("Read memory locations 1")
    ax3.set_title("Read memory locations 2")
    ax4.set_title("Read memory locations 3")
    ax5.set_title("Read memory locations 4")
    plt.colorbar(i1, ax=ax1)
    plt.colorbar(i2, ax=ax2)
    plt.colorbar(i3, ax=ax3)
    plt.colorbar(i4, ax=ax4)
    plt.colorbar(i5, ax=ax5)

    with open("collected_data/c_write_weights.pkl", "rb") as f:
        c_write_weights = pickle.load(f)

    with open("collected_data/c_read_weights_1.pkl", "rb") as f:
        c_read_weights_1 = pickle.load(f)

    with open("collected_data/c_read_weights_2.pkl", "rb") as f:
        c_read_weights_2 = pickle.load(f)

    with open("collected_data/c_read_weights_3.pkl", "rb") as f:
        c_read_weights_3 = pickle.load(f)

    with open("collected_data/c_read_weights_4.pkl", "rb") as f:
        c_read_weights_4 = pickle.load(f)


    c_write_weights = np.repeat(c_write_weights, 30, axis=0)
    c_read_weights_1 = np.repeat(c_read_weights_1, 30, axis=0)
    c_read_weights_2 = np.repeat(c_read_weights_2, 30, axis=0)
    c_read_weights_3 = np.repeat(c_read_weights_3, 30, axis=0)
    c_read_weights_4 = np.repeat(c_read_weights_4, 30, axis=0)

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    i1= ax1.imshow(c_write_weights)
    ax1.grid(b=False)

    i2 = ax2.imshow(c_read_weights_1)
    ax2.grid(b=False)

    i3 = ax3.imshow(c_read_weights_2)
    ax3.grid(b=False)

    i4 = ax4.imshow(c_read_weights_3)
    ax4.grid(b=False)

    i5 = ax5.imshow(c_read_weights_4)
    ax5.grid(b=False)
    ax5.set_xlabel("memory_cells")
    ax1.set_ylabel("time")
    ax2.set_ylabel("time")
    ax3.set_ylabel("time")
    ax4.set_ylabel("time")
    ax5.set_ylabel("time")
    ax1.set_title("Write memory locations")
    ax2.set_title("Read memory locations 1")
    ax3.set_title("Read memory locations 2")
    ax4.set_title("Read memory locations 3")
    ax5.set_title("Read memory locations 4")
    plt.colorbar(i1, ax=ax1)
    plt.colorbar(i2, ax=ax2)
    plt.colorbar(i3, ax=ax3)
    plt.colorbar(i4, ax=ax4)
    plt.colorbar(i5, ax=ax5)


def plot_copy():
    with open("collected_data/inputs.pkl", "rb") as f:
        inputs = pickle.load(f)

    with open("collected_data/outputs.pkl", "rb") as f:
        outputs = pickle.load(f)

    with open("collected_data/write_weights.pkl", "rb") as f:
        write_weights = pickle.load(f)

    with open("collected_data/read_weights.pkl", "rb") as f:
        read_weights = pickle.load(f)

    with open("collected_data/mem_contrib.pkl", "rb") as f:
        mem_contrib = pickle.load(f)

    with open("collected_data/free_gates.pkl", "rb") as f:
        free_gates = pickle.load(f)

    with open("collected_data/alloc_gates.pkl", "rb") as f:
        alloc_gates = pickle.load(f)

    with open("collected_data/write_gates.pkl", "rb") as f:
        write_gates = pickle.load(f)

    with open("collected_data/read_modes.pkl", "rb") as f:
        read_modes = pickle.load(f)

    inputs = np.concatenate(inputs, axis=0)
    inputs = np.transpose(inputs)

    outputs = np.concatenate(outputs, axis=0)
    outputs = np.transpose(outputs)
    write_weights = np.concatenate(write_weights, axis=0)
    write_weights = np.transpose(write_weights)
    i = 0
    while i <= 55:
        c1 = deepcopy(write_weights[:, i:i+5])
        write_weights[:, i:i+5] = write_weights[:, i+6:i+6+5]
        write_weights[:, i+6:i+6+5] = c1
        i += 11
    read_weights = np.concatenate(read_weights, axis=0)
    read_weights = np.transpose(read_weights)

    free_gates = np.squeeze(np.concatenate(free_gates, axis=0), axis=-1)
    alloc_gates = np.squeeze(np.concatenate(alloc_gates, axis=0), axis=-1)
    write_gates = np.squeeze(np.concatenate(write_gates, axis=0), axis=-1)

    read_modes = np.concatenate(read_modes, axis=0)


    mem_contrib = np.concatenate(mem_contrib, axis=0)
    mem_contrib = np.transpose(mem_contrib)

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, sharex=True)
    ax1.imshow(inputs[:-1, :])
    ax1.grid(b=False)
    ax2.imshow(outputs[:-1, :])
    ax2.grid(b=False)
    im = ax3.imshow(write_weights)
    ax3.grid(b=False)
    asp = np.abs(np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0])
    asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    ax3.set_aspect(asp)
    ax4.imshow(read_weights)
    ax4.grid(b=False)
    ax4.set_aspect(asp)

    ax5.bar(range(0, 55), np.ones_like(alloc_gates), color="y", label="content")
    ax5.bar(range(0, 55), alloc_gates, label="usage")
    ax5.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax6.bar(range(0, 55), np.ones_like(free_gates), color="brown", label="keep")
    ax6.bar(range(0, 55), free_gates, color="seagreen", label="free")
    ax6.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax7.bar(range(0, 55), np.ones_like(write_gates), color="darksalmon", label="write")
    ax7.bar(range(0, 55), write_gates, color="mediumaquamarine", label="no write")
    ax7.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax8.bar(range(0, 55), np.ones_like(write_gates), color="indigo", label="content")
    ax8.bar(range(0, 55), read_modes[:, 1] + read_modes[:, 0], color="pink", label="forward")
    ax8.bar(range(0, 55), read_modes[:, 1], color="royalblue", label="backward")
    ax8.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax9.bar(range(0,55), np.ones_like(mem_contrib), color="peru", label="controller")
    ax9.bar(range(0,55), mem_contrib, color="indigo", label="memory")
    ax9.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    f2, (ax1, ax2) = plt.subplots(2,1)
    controller_good   = np.genfromtxt('collected_data/Contributions_Controller_good.csv', delimiter = ',', skip_header=1)[:, 1:]
    memory_good   = np.genfromtxt('collected_data/Contributions_Memory_good.csv', delimiter = ',', skip_header=1)[:, 1:]
    controller_bad = np.genfromtxt('collected_data/Contributions_Controller_bad.csv', delimiter = ',', skip_header=1)[:, 1:]
    memory_bad = np.genfromtxt('collected_data/Contributions_Memory_bad.csv', delimiter = ',', skip_header=1)[:, 1:]
    ax1.plot(controller_good[:, 0], controller_good[:, 1], label="Controller")
    ax1.plot(memory_good[:, 0], memory_good[:, 1], label="Memory")
    ax1.set_ylabel("Contributions of \n trained DNC")
    ax1.set_yticks([0.0, 1.0])
    ax1.legend()

    ax2.plot(controller_bad[:, 0], controller_bad[:, 1], label="Controller")
    ax2.plot(memory_bad[:, 0], memory_bad[:, 1], label="Memory")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Contributions of \n untrained DNC")
    ax2.set_yticks([0.0, 1.0])
    ax2.legend()

    f2, (ax1, ax2) = plt.subplots(2,1)
    dnc_acc = np.genfromtxt('collected_data/Accuracy_dnc_copy.csv', delimiter = ',', skip_header=1)[:, 1:]
    lstm_acc = np.genfromtxt('collected_data/Accuracy_lstm_copy.csv', delimiter = ',', skip_header=1)[:, 1:]

    dnc_loss = np.genfromtxt('collected_data/Loss_dnc_copy.csv', delimiter = ',', skip_header=1)[:, 1:]
    lstm_loss = np.genfromtxt('collected_data/Loss_lstm_copy.csv', delimiter = ',', skip_header=1)[:, 1:]

    ax1.plot(dnc_acc[:, 0], dnc_acc[:, 1] * 100, label="DNC")
    ax1.plot(lstm_acc[:, 0], lstm_acc[:, 1] * 100, label="LSTM")
    ax1.set_ylabel("Accuracy in %")
    # ax1.set_yticks([0.0, 100.0])
    ax1.legend()

    ax2.plot(dnc_loss[:, 0], dnc_loss[:, 1], label="DNC")
    ax2.plot(lstm_loss[:, 0], lstm_loss[:, 1], label="LSTM")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    # ax2.set_yticks([0.0, 1.0])
    ax2.legend()


def plot_babi():
    with open("collected_data/qa1_con_contrib.pkl", "rb") as f:
        q1_con_contrib = pickle.load(f)

    with open("collected_data/qa1_mem_contrib.pkl", "rb") as f:
        q1_mem_contrib = pickle.load(f)

    with open("collected_data/qa4_con_contrib.pkl", "rb") as f:
        q4_con_contrib = pickle.load(f)

    with open("collected_data/qa4_mem_contrib.pkl", "rb") as f:
        q4_mem_contrib = pickle.load(f)

    with open("collected_data/qa9_con_contrib.pkl", "rb") as f:
        q9_con_contrib = pickle.load(f)

    with open("collected_data/qa9_mem_contrib.pkl", "rb") as f:
        q9_mem_contrib = pickle.load(f)

    with open("collected_data/qa10_con_contrib.pkl", "rb") as f:
        q10_con_contrib = pickle.load(f)

    with open("collected_data/qa10_mem_contrib.pkl", "rb") as f:
        q10_mem_contrib = pickle.load(f)

    with open("collected_data/qa11_con_contrib.pkl", "rb") as f:
        q11_con_contrib = pickle.load(f)

    with open("collected_data/qa11_mem_contrib.pkl", "rb") as f:
        q11_mem_contrib = pickle.load(f)

    with open("collected_data/qa14_con_contrib.pkl", "rb") as f:
        q14_con_contrib = pickle.load(f)

    with open("collected_data/qa14_mem_contrib.pkl", "rb") as f:
        q14_mem_contrib = pickle.load(f)

    mem = [q1_mem_contrib, q4_mem_contrib, q9_mem_contrib, q10_mem_contrib, q11_mem_contrib, q14_mem_contrib]
    con = [q1_con_contrib, q4_con_contrib, q9_con_contrib, q10_con_contrib, q11_con_contrib, q14_con_contrib]
    names = ["qa1", "qa4", "qa9", "qa10", "qa11", "qa14"]

    for c, m, n in zip(con, mem, names):
        plt.figure()
        if n != "qa4":
            plt.plot(range(len(c)), c, label="controller")
            plt.plot(range(len(m)), m, label="memory")
        else:
            plt.plot(range(len(c)), c, alpha=0.2)
            plt.plot(range(len(m)), m, alpha=0.2)
            c_s = gaussian_filter1d(c, sigma=3)
            m_s = gaussian_filter1d(m, sigma=3)

            plt.plot(range(len(c_s)), c_s, label="controller", color="#e24a33")
            plt.plot(range(len(m_s)), m_s, label="memory", color="#348abd")

        plt.xlabel("Iterations")
        plt.title(n)
        plt.legend()

    with open("collected_data/read_weights_1_babi.pkl", "rb") as f:
        read_weights_1_babi = pickle.load(f)

    with open("collected_data/read_weights_2_babi.pkl", "rb") as f:
        read_weights_2_babi = pickle.load(f)

    with open("collected_data/read_weights_3_babi.pkl", "rb") as f:
        read_weights_3_babi = pickle.load(f)

    with open("collected_data/read_weights_4_babi.pkl", "rb") as f:
        read_weights_4_babi = pickle.load(f)

    with open("collected_data/read_modes_1_babi.pkl", "rb") as f:
        read_modes_1_babi = pickle.load(f)

    with open("collected_data/read_modes_2_babi.pkl", "rb") as f:
        read_modes_2_babi = pickle.load(f)

    with open("collected_data/read_modes_3_babi.pkl", "rb") as f:
        read_modes_3_babi = pickle.load(f)

    with open("collected_data/read_modes_4_babi.pkl", "rb") as f:
        read_modes_4_babi = pickle.load(f)

    with open("collected_data/free_gates_1_babi.pkl", "rb") as f:
        free_gates_1_babi = pickle.load(f)

    with open("collected_data/free_gates_2_babi.pkl", "rb") as f:
        free_gates_2_babi = pickle.load(f)

    with open("collected_data/free_gates_3_babi.pkl", "rb") as f:
        free_gates_3_babi = pickle.load(f)

    with open("collected_data/free_gates_4_babi.pkl", "rb") as f:
        free_gates_4_babi = pickle.load(f)

    with open("collected_data/write_weights_babi.pkl", "rb") as f:
        write_weights_babi = pickle.load(f)

    with open("collected_data/memory_influence_vec_babi.pkl", "rb") as f:
        memory_influence_vec_babi = pickle.load(f)

    with open("collected_data/allocation_gate_babi.pkl", "rb") as f:
        allocation_gate_babi = pickle.load(f)

    with open("collected_data/write_gate_babi.pkl", "rb") as f:
        write_gate_babi = pickle.load(f)

    write_weights_babi = write_weights_babi[:24, :]
    read_weights_1_babi = read_weights_1_babi[:24, :]
    read_weights_2_babi = read_weights_2_babi[:24, :]
    read_weights_3_babi = read_weights_3_babi[:24, :]
    read_weights_4_babi = read_weights_4_babi[:24, :]

    read_modes_1_babi = read_modes_1_babi[:24, :]
    read_modes_2_babi = read_modes_2_babi[:24, :]
    read_modes_3_babi = read_modes_3_babi[:24, :]
    read_modes_4_babi = read_modes_4_babi[:24, :]

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1, sharex=True)

    allocation_gate_babi = np.squeeze(allocation_gate_babi, axis=-1)[:24]
    write_gate_babi = np.squeeze(write_gate_babi, axis=-1)[:24]
    free_gates_1_babi = free_gates_1_babi[:24]
    free_gates_2_babi = free_gates_2_babi[:24]
    free_gates_3_babi = free_gates_2_babi[:24]
    free_gates_4_babi = free_gates_2_babi[:24]

    memory_influence_vec_babi = memory_influence_vec_babi[:24]

    ax1.bar(np.array(range(0, 24)), np.ones_like(allocation_gate_babi), color="y", label="content", width=0.4)
    ax1.bar(range(0, 24), allocation_gate_babi, label="usage", width=0.4)
    ax1.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    # ax1.set_xticks([0, 23])

    ax2.bar(np.array(range(0, 24)) - 0.15, np.ones_like(free_gates_1_babi), color="brown", label="keep", width=0.1)
    ax2.bar(np.array(range(0, 24)) - 0.15, free_gates_1_babi, color="seagreen", label="free", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    # ax2.set_xticks([0, 23])

    ax2.bar(np.array(range(0, 24)) - 0.05, np.ones_like(free_gates_2_babi), color="brown", width=0.1)
    ax2.bar(np.array(range(0, 24)) - 0.05, free_gates_2_babi, color="seagreen", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax2.bar(np.array(range(0, 24)) + 0.05, np.ones_like(free_gates_3_babi), color="brown", width=0.1)
    ax2.bar(np.array(range(0, 24)) + 0.05, free_gates_3_babi, color="seagreen", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax2.bar(np.array(range(0, 24)) + 0.15, np.ones_like(free_gates_4_babi), color="brown", width=0.1)
    ax2.bar(np.array(range(0, 24)) + 0.15, free_gates_4_babi, color="seagreen", width=0.1)
    ax2.legend(loc="right", bbox_to_anchor=(1.1, 0.5))

    ax3.bar(range(0, 24), np.ones_like(write_gate_babi), color="mediumaquamarine", label="no write", width=0.4)
    ax3.bar(range(0, 24), write_gate_babi, color="darksalmon", label="write", width=0.4)
    ax3.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    # ax3.set_xticks([0, 23])

    ax4.bar(range(0, 24), np.ones_like(read_modes_1_babi[:, 1]), color="indigo", label="content", width=0.4)
    ax4.bar(range(0, 24), read_modes_1_babi[:, 1] + read_modes_1_babi[:, 0], color="#e24a33", label="forward", width=0.4)
    ax4.bar(range(0, 24), read_modes_1_babi[:, 1], color="royalblue", label="backward", width=0.4)
    ax4.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    # ax4.set_xticks([0, 23])

    ax5.bar(range(0, 24), np.ones_like(read_modes_2_babi[:, 1]), color="indigo", label="content", width=0.4)
    ax5.bar(range(0, 24), read_modes_2_babi[:, 1] + read_modes_2_babi[:, 0], color="#e24a33", label="forward", width=0.4)
    ax5.bar(range(0, 24), read_modes_2_babi[:, 1], color="royalblue", label="backward", width=0.4)
    ax5.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    # ax5.set_xticks([0, 23])


    ax6.bar(range(0, 24), np.ones_like(read_modes_3_babi[:, 1]), color="indigo", label="content", width=0.4)
    ax6.bar(range(0, 24), read_modes_3_babi[:, 1] + read_modes_3_babi[:, 0], color="#e24a33", label="forward", width=0.4)
    ax6.bar(range(0, 24), read_modes_3_babi[:, 1], color="royalblue", label="backward", width=0.4)
    ax6.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    # ax6.set_xticks([0, 23])

    ax7.bar(range(0, 24), np.ones_like(read_modes_4_babi[:, 1]), color="indigo", label="content", width=0.4)
    ax7.bar(range(0, 24), read_modes_4_babi[:, 1] + read_modes_4_babi[:, 0], color="#e24a33", label="forward", width=0.4)
    ax7.bar(range(0, 24), read_modes_4_babi[:, 1], color="royalblue", label="backward", width=0.4)
    ax7.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    # ax7.set_xticks([0, 23])

    ax8.bar(range(0,24), np.ones_like(memory_influence_vec_babi), color="peru", label="controller", width=0.4)
    ax8.bar(range(0,24), memory_influence_vec_babi, color="indigo", label="memory", width=0.4)
    ax8.legend(loc="right", bbox_to_anchor=(1.1, 0.5))
    ax8.set_xlabel("time")

    ax1.set_ylabel("Allocation gates", rotation=0)
    ax1.yaxis.set_label_coords(-0.1,0.9)

    ax2.set_ylabel("Free gates", rotation=0)
    ax2.yaxis.set_label_coords(-0.1,0.9)

    ax3.set_ylabel("write gates", rotation=0)
    ax3.yaxis.set_label_coords(-0.1,0.9)

    ax4.set_ylabel("Read modes\nHead 1", rotation=0)
    ax4.yaxis.set_label_coords(-0.1,0.9)

    ax5.set_ylabel("Read modes\nHead 2", rotation=0)
    ax5.yaxis.set_label_coords(-0.1,0.9)

    ax6.set_ylabel("Read modes\nHead 3", rotation=0)
    ax6.yaxis.set_label_coords(-0.1,0.9)

    ax7.set_ylabel("Read modes\nHead 4", rotation=0)
    ax7.yaxis.set_label_coords(-0.1,0.9)

    ax8.set_ylabel("Output\nContributions", rotation=0)
    ax8.yaxis.set_label_coords(-0.1,0.9)

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    i1= ax1.imshow(write_weights_babi)
    ax1.grid(b=False)

    i2 = ax2.imshow(read_weights_1_babi)
    ax2.grid(b=False)

    i3 = ax3.imshow(read_weights_2_babi)
    ax3.grid(b=False)

    i4 = ax4.imshow(read_weights_3_babi)
    ax4.grid(b=False)

    i5 = ax5.imshow(read_weights_4_babi)
    ax5.grid(b=False)
    ax5.set_xlabel("memory_cells")
    ax1.set_ylabel("time")
    ax2.set_ylabel("time")
    ax3.set_ylabel("time")
    ax4.set_ylabel("time")
    ax5.set_ylabel("time")
    ax1.set_title("Write memory locations")
    ax2.set_title("Read memory locations 1")
    ax3.set_title("Read memory locations 2")
    ax4.set_title("Read memory locations 3")
    ax5.set_title("Read memory locations 4")
    plt.colorbar(i1, ax=ax1)
    plt.colorbar(i2, ax=ax2)
    plt.colorbar(i3, ax=ax3)
    plt.colorbar(i4, ax=ax4)
    plt.colorbar(i5, ax=ax5)

   
    loss_train_dnc  = np.genfromtxt('collected_data/run-Loss_TRAIN_Epoch_DNC-tag-Loss_TRAIN_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    loss_train_lstm  = np.genfromtxt('collected_data/run-Loss_TRAIN_Epoch_LSTM-tag-Loss_TRAIN_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_train_dnc = np.genfromtxt('collected_data/run-Accuracy_TRAIN_Epoch_DNC-tag-Accuracy_TRAIN_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_train_lstm = np.genfromtxt('collected_data/run-Accuracy_TRAIN_Epoch_LSTM-tag-Accuracy_TRAIN_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_val_dnc = np.genfromtxt('collected_data/run-Accuracy_VAL_Epoch_DNC-tag-Accuracy_VAL_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_val_lstm = np.genfromtxt('collected_data/run-Accuracy_VAL_Epoch_LSTM-tag-Accuracy_VAL_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]

    plt.figure()

    plt.plot(loss_train_dnc[:, 0], loss_train_dnc[:, 1], label="Train_Loss_DNC")
    plt.plot(loss_train_lstm[:, 0], loss_train_lstm[:, 1], label="Train_Loss_LSTM")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Train loss plots
    plt.figure()
    plt.plot(acc_train_dnc[:, 0], acc_train_dnc[:, 1], label="Train_Loss_DNC")
    plt.plot(acc_train_lstm[:, 0], acc_train_lstm[:, 1], label="Train_Loss_LSTM")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.legend()


    plt.figure()
    plt.plot(acc_val_dnc[:, 0], acc_val_dnc[:, 1], label="Train_Loss_DNC")
    plt.plot(acc_val_lstm[:, 0], acc_val_lstm[:, 1], label="Train_Loss_LSTM")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.legend()


def plot_vqa():
    with open("collected_data/memory_contrib_q_DNC_Q_256_.pkl", "rb") as f:
        memory_contrib_q = pickle.load(f)
    with open("collected_data/memory_contrib_c_DNC_C_256_.pkl", "rb") as f:
        memory_contrib_c = pickle.load(f)
    with open("collected_data/memory_contrib_q_DNC_QC_256_.pkl", "rb") as f:
        memory_contrib_q_qc = pickle.load(f)
    with open("collected_data/memory_contrib_c_DNC_QC_256_.pkl", "rb") as f:
        memory_contrib_c_qc = pickle.load(f)

    plt.figure()
    plt.plot(range(len(memory_contrib_q)), memory_contrib_q, alpha=0.2)
    plt.plot(range(len(memory_contrib_q_qc)), memory_contrib_q_qc, alpha=0.2)
    memory_contrib_q_s = gaussian_filter1d(memory_contrib_q, sigma=3)
    memory_contrib_q_qc_s = gaussian_filter1d(memory_contrib_q_qc, sigma=3)

    plt.plot(range(len(memory_contrib_q_s)), memory_contrib_q_s, label="memory-Q", color="#e24a33")
    plt.plot(range(len(memory_contrib_q_qc_s)), memory_contrib_q_qc_s, label="memory-QC", color="#348abd")
    plt.xlabel("Iterations")
    plt.legend()

    plt.figure()
    plt.plot(range(len(memory_contrib_c)), memory_contrib_c, alpha=0.2)
    plt.plot(range(len(memory_contrib_c_qc)), 10*np.array(memory_contrib_c_qc), alpha=0.2)
    memory_contrib_c_s = gaussian_filter1d(memory_contrib_c, sigma=3)
    memory_contrib_c_qc_s = gaussian_filter1d(10*np.array(memory_contrib_c_qc), sigma=3)
    plt.plot(range(len(memory_contrib_c_s)), memory_contrib_c_s, label="memory-C", color="#e24a33")
    plt.plot(range(len(memory_contrib_c_qc_s)), memory_contrib_c_qc_s, label="memory-QC", color="#348abd")
    plt.xlabel("Iterations")
    plt.legend()

    loss_tr_dnc_c_lstm  = np.genfromtxt('collected_data/Loss_train_dnc_c_lstm_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_val_dnc_c_lstm  = np.genfromtxt('collected_data/Acc_overall_valid_dnc_c_lstm_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]

    loss_tr_dnc_qc_w_init = np.genfromtxt('collected_data/Loss_train_dnc_qc_w_init_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_val_dnc_qc_w_init = np.genfromtxt('collected_data/Acc_overall_valid_dnc_qc_w_ini_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]

    loss_tr_dnc_c_mlp  = np.genfromtxt('collected_data/Loss_train_dnc_c_mlp_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_val_dnc_c_mlp  = np.genfromtxt('collected_data/Acc_overall_valid_dnc_c_mlp_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]

    loss_tr_dnc_qc = np.genfromtxt('collected_data/Loss_train_dnc_qc_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_val_dnc_qc = np.genfromtxt('collected_data/Acc_overall_valid_dnc_qc_Epoch.csv', delimiter = ',', skip_header=1)[:, 1:]

    plt.figure()

    plt.plot(loss_tr_dnc_c_lstm[:, 0], loss_tr_dnc_c_lstm[:, 1], label="LSTM")
    plt.plot(loss_tr_dnc_c_mlp[:, 0], loss_tr_dnc_c_mlp[:, 1], label="MLP")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()

    # Train loss plots
    plt.figure()
    plt.plot(acc_val_dnc_c_lstm[:, 0], acc_val_dnc_c_lstm[:, 1], label="LSTM")
    plt.plot(acc_val_dnc_c_mlp[:, 0], acc_val_dnc_c_mlp[:, 1], label="MLP")
    plt.xlabel("Epoch")
    plt.ylabel("Vaildation Accuracy [%]")
    plt.legend()


    plt.figure()
    plt.plot(loss_tr_dnc_qc_w_init[:, 0], loss_tr_dnc_qc_w_init[:, 1], label="With weight init.")
    plt.plot(loss_tr_dnc_qc[:, 0], loss_tr_dnc_qc[:, 1], label="No weight init.")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()

    # Train loss plots
    plt.figure()
    plt.plot(acc_val_dnc_qc_w_init[:, 0], acc_val_dnc_qc_w_init[:, 1], label="With weight init.")
    plt.plot(acc_val_dnc_qc[:, 0], acc_val_dnc_qc[:, 1], label="No weight init.")
    plt.xlabel("Epoch")
    plt.ylabel("Vaildation Accuracy [%]")
    plt.legend()

    loss_train_dnc  = np.genfromtxt('collected_data/Loss_train_Epoch_DNC_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]
    loss_train_lstm  = np.genfromtxt('collected_data/Loss_train_Epoch_LSTM_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]
    
    loss_val_dnc = np.genfromtxt('collected_data/Loss_valid_Epoch_DNC_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]
    loss_val_lstm = np.genfromtxt('collected_data/Loss_valid_Epoch_LSTM_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]
    
    acc_val_dnc = np.genfromtxt('collected_data/Acc_overall_valid_Epoch_DNC_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]
    acc_val_lstm = np.genfromtxt('collected_data/Acc_overall_valid_Epoch_LSTM_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]

    lr_dnc  = np.genfromtxt('collected_data/lr_DNC_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]
    lr_rest  = np.genfromtxt('collected_data/lr_Rest_VQA.csv', delimiter = ',', skip_header=1)[:, 1:]
    
    plt.figure()

    plt.plot(loss_train_dnc[:, 0], loss_train_dnc[:, 1], label="DNC-Q")
    plt.plot(loss_train_lstm[:, 0], loss_train_lstm[:, 1], label="Deep LSTM-Q")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()

    # Train loss plots
    plt.figure()
    plt.plot(loss_val_dnc[:, 0], loss_val_dnc[:, 1], label="DNC-Q")
    plt.plot(loss_val_lstm[:, 0], loss_val_lstm[:, 1], label="Deep LSTM-Q")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()


    plt.figure()
    plt.plot(acc_val_dnc[:, 0], acc_val_dnc[:, 1], label="DNC-Q")
    plt.plot(acc_val_lstm[:, 0], acc_val_lstm[:, 1], label="Deep LSTM-Q")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy [%]")
    plt.legend()

    plt.figure()
    plt.plot(lr_dnc[:, 0], lr_dnc[:, 1], label="lr-Q")
    plt.plot(lr_rest[:, 0], lr_rest[:, 1], label="lr-Rest")
    plt.xlabel("Iterations")
    plt.ylabel("lr")
    plt.legend()


if __name__ == "__main__":
    plot_io_data_copy_task()
    plot_copy()
    plot_babi()
    plot_vqa()
    plot_analysis_data_DNC_QC()
    plt.show()
