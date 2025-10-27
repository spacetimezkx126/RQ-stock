import torch
import torch.nn as nn
import pennylane as qml
class VQACircuit2(torch.nn.Module):
    def __init__(self):
        super(VQACircuit2, self).__init__()
        n_qubits = 7
        dev = qml.device("default.qubit", wires=n_qubits)
        n_layers = 4  # 量子电路层数
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))  # 量子电路参数
        self.n_qubits = n_qubits
        self.qnode = qml.QNode(self._qnode_func, dev, interface="torch", diff_method="backprop")
    def _qnode_func(self, x):
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)
        qml.StronglyEntanglingLayers(self.weights, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(self.n_qubits-1))  # 首尾关联
    def forward(self, x):
        return self.qnode(x)