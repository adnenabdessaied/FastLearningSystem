import torch
import torch.nn as nn
from DNC.memory import Memory
from DNC.util import *


class DNC_MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_hidden_layers,
        dropout,
        nr_cells,
        cell_size,
        read_heads,
        nonlinearity,
        gpu_id,
        debug,
        clip
    ):
        super(DNC_MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.n = nr_cells
        self.w = cell_size
        self.r = read_heads
        self.nonlinearity = nonlinearity
        self.gpu_id = gpu_id
        self.debug = debug
        self.clip = clip
        self.memory = Memory(
            input_size=self.hidden_size,
            mem_size=self.n,
            cell_size=self.w,
            read_heads=self.r,
            gpu_id=self.gpu_id,
            independent_linears=False
        )
        # self.input_size + self.r * self.w,
        self.linear_1 = nn.Linear(self.input_size + self.r * self.w, self.hidden_size)
        self.hidden_linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Dropout(self.dropout)) for _ in range(self.num_hidden_layers)])
        self.read_linear = nn.Linear(self.r * self.w, self.output_size)
        self.last_linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(1)

    def init_hidden(self, hx, batch_size, reset_experience):
        if hx is None:
            hx = (None, None)
        (mhx, last_read) = hx
        if last_read is None:
            last_read = cuda(torch.zeros(batch_size, self.w * self.r), gpu_id=self.gpu_id)
        if mhx is None:
            mhx = self.memory.reset(batch_size, erase=reset_experience)
        else:
            mhx = self.memory.reset(batch_size, hidden=mhx, erase=reset_experience)
        return mhx, last_read

    def _debug(self, mhx, debug_obj):
        if not debug_obj:
            debug_obj = {
                'memory': [],
                'link_matrix': [],
                'precedence': [],
                'read_weights': [],
                'write_weights': [],
                'usage_vector': [],
                'free_gates': [],
                'allocation_gate': [],
                'write_gate': [],
                'read_modes': []
            }

        debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
        debug_obj['link_matrix'].append(
            mhx['link_matrix'][0][0].data.cpu().numpy())
        debug_obj['precedence'].append(mhx['precedence'][0].data.cpu().numpy())
        debug_obj['read_weights'].append(
            mhx['read_weights'][0].data.cpu().numpy())
        debug_obj['write_weights'].append(
            mhx['write_weights'][0].data.cpu().numpy())
        debug_obj['usage_vector'].append(
            mhx['usage_vector'][0].unsqueeze(0).data.cpu().numpy())

        debug_obj['free_gates'].append(
            mhx['free_gates'][0].data.cpu().numpy())
        debug_obj['allocation_gate'].append(
            mhx['allocation_gate'][0].data.cpu().numpy())
        debug_obj['write_gate'].append(
            mhx['write_gate'][0].data.cpu().numpy())
        debug_obj['read_modes'].append(
            mhx['read_modes'][0].data.cpu().numpy())
        return debug_obj

    def forward(self, input_, mhx, rv):
        batch_size = input_.size(0)
        # rv = rv.view(batch_size, -1)
        intput_interface = torch.cat((input_, rv), 1)
        h = self.nonlinearity(self.linear_1(intput_interface))
        # if self.clip != 0:
        #     h = torch.clamp(h, -self.clip, self.clip)
        for m in self.hidden_linears:
            h = self.nonlinearity(m(h))
            # h = m(h)
            # if self.clip != 0:
            #     h = torch.clamp(h, -self.clip, self.clip)
        vu_t = self.last_linear(h)
        rv, mhx = self.memory(h, mhx)
        rv = rv.view(batch_size, -1)
        read_t = self.read_linear(rv.contiguous())
        y_t = torch.add(vu_t, read_t)
        if self.debug:
            viz = self._debug(mhx, None)
            viz = {k: np.array(v) for k, v in viz.items()}
            reshape_keys = [
                "memory", "link_matrix", "precedence", "read_weights", "write_weights", "usage_vector"]
            for key in reshape_keys:
                viz[key] = viz[key].reshape(viz[key].shape[0],  viz[key].shape[1] * viz[key].shape[2])
            mean_keys = ["free_gates", "allocation_gate", "write_gate", "read_modes"]
            for key in mean_keys:
                viz[key] = np.mean(viz[key], axis=0)

            # compute the memory and controller influences
            y_t_ = self.softmax(y_t)
            read_t_ = self.softmax(read_t)
            vu_t_ = self.softmax(vu_t)


            m_influence = torch.abs(y_t_ - vu_t_)
            m_influence = torch.mean(m_influence, 0)
            m_influence = torch.mean(m_influence, 0)

            c_influence = torch.abs(y_t_ - read_t_)
            c_influence = torch.mean(c_influence, 0)
            c_influence = torch.mean(c_influence, 0)

            m_inf = m_influence / (m_influence + c_influence)
            c_inf = c_influence / (m_influence + c_influence)
            viz["memory_influence"] = m_inf
            viz["controller_influence"] = c_inf
            return y_t, (mhx, rv), viz
        else:
            return y_t, (mhx, rv)
