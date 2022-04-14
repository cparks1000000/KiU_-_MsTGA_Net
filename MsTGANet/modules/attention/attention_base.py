from itertools import product
from typing import List, Callable, Any, Tuple

import torch
from torch import nn as nn, Tensor, zeros_like
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch.autograd.functional import jacobian
from torch.nn.functional import softmax


# todo: Check that this function works and add the backward pass.
# noinspection PyMethodOverriding
class KiAttention(Function):
    @staticmethod
    def forward(context: FunctionCtx, query: Tensor, key: Tensor, value: Tensor, heat: Tensor) -> Tensor:
        context.save_for_backward(query, key, value, heat)
        
        batch_size, channel_size, height, width = value.size()
        def H(x: int) -> int: return min(height-1, max(x, 0))
        def W(x: int) -> int: return min(width-1, max(x, 0))
        window_size: int = 10
        query_list: List[Tensor] = list(query)
        key_list: List[Tensor] = list(key)
        value_list: List[Tensor] = list(value)
        
        holder = torch.zeros(batch_size, channel_size, height*width)
        for b in range(batch_size):
            query = query_list[b]
            key = key_list[b]
            value = value_list[b]
            for c, h, w in product(range(channel_size), range(height), range(width)):
                dh = slice(H(h-window_size), H(h+window_size+1))
                dw = slice(W(w-window_size), W(w+window_size+1))
                temp = key[:, dh, dw] + heat[dh, dw]
                energy = (query[:, h, w].expand(-1, temp.size(1), temp.size(2)) * temp ).sum(1)
                holder[b, c, h, w] = sum(
                    value[c, dh, dw] * softmax(
                        energy.reshape(energy.size(0), -1),
                        dim=1
                    ).reshape_as(energy) #todo: I had to change this so the backward pass is now wrong.
                )
        return holder
    
    @staticmethod
    def backward(context: FunctionCtx, grad_outputs: Tensor) -> Tuple[Tensor, ...]:
        # noinspection PyUnresolvedReferences
        query, key, value, heat = context.saved_tensors
        batch_size, channel_size, height, width = value.size()
        def H(x: int) -> int: return min(height-1, max(x, 0))
        def W(x: int) -> int: return min(width-1, max(x, 0))
        window_size: int = 10
        
        d_value = zeros_like(value)
        d_key = zeros_like(key)
        d_query = zeros_like(query)
        d_heat = zeros_like(heat)
        for b in range(batch_size):
            for c, h, w in product(range(channel_size), range(height), range(width)):
                dh = slice(H(h-window_size), H(h+window_size+1))
                dw = slice(W(w-window_size), W(w+window_size+1))
                temp = key[:, dh, dw] + heat[dh, dw]
                energy = (query[:, h, w].expand(-1, temp.size(1), temp.size(2)) * temp ).sum(1)
                for u, v in product(range(dh.start, dh.stop), range(dw.start, dw.stop)):
                    d_softmax: Tensor = jacobian(
                        lambda x: softmax( x, dim=1 ).reshape_as(energy),
                        energy.reshape(energy.size(0), -1)
                    )
                    
                    d_value[b, c, u, v] += \
                        grad_outputs[b, c, h, w] * \
                        softmax( (query[:, h, w].expand(-1, temp.size(1), temp.size(2)) * temp[:, dh, dw] ).sum(1), dim=1)
                    
                    for U, V in product(range(dh.start, dh.stop), range(dw.start, dw.stop)):
                        # accumulate the key derivative todo: This is broken
                        d_key += grad_outputs[b, c, h, w] * value[b, c, h, w] * d_softmax[u, v, U, V] * \
                        
                    
                        # accumulate the query derivative todo: This is broken
                        d_query += grad_outputs[b, c, h, w] * value[b, c, h, w] * d_softmax[u, v, U, V] * \
                        
                    
                        # accumulate the heat derivative todo: This is broken
                        d_heat += grad_outputs[b, c, h, w] * value[b, c, h, w] * d_softmax[u, v, U, V] * \
                        
        
        return d_query, d_key, d_value, d_heat
    

def ki_attention(query: Tensor, key: Tensor, value: Tensor, heat: Tensor) -> Tensor:
    return KiAttention.apply(query, key, value, heat)


def u_attention_alt(query: Tensor, key: Tensor, value: Tensor, heat: Tensor) -> Tensor:
        batch_size, channel_size, height, width = value.size()
        query_list: List[Tensor] = list(query)
        key_list: List[Tensor] = list(key)
        value_list: List[Tensor] = list(value)
        
        holder = torch.zeros(batch_size, channel_size, height*width)
        for b in range(batch_size):
            query = query_list[b]
            key = key_list[b]
            value = value_list[b]
            for c, h, w in product(range(channel_size), range(height), range(width)):
                holder[b, c, h, w] = value[c] @ torch.softmax(query[h, w] @ (key + heat), dim=0).transpose(0, 1)
        return holder


def u_attention(query: Tensor, key: Tensor, value: Tensor, heat: Tensor) -> Tensor:
    batch_size, channel_size, height, width = value.size()
    query = query.view(batch_size, channel_size // 8, width * height).transpose(1, 2)
    key = key.view(batch_size, channel_size // 8, width * height)
    value = value.view(batch_size, channel_size, width * height)
    
    energy_content = torch.bmm(query, key)
    content_position = torch.matmul(query, heat)
    energy = energy_content+content_position
    scaling = torch.softmax(energy, dim=1)
    
    return torch.bmm(value, scaling.transpose(1, 2)).view(batch_size, channel_size, width, height)


class AttentionBase(nn.Module):
    def __init__(self, channels_in: int, height: int, width: int, *,
                 max_dimensions: int = 128,
                 attention_function: Callable[[Tensor, Tensor, Tensor, Tensor, Tuple], Tensor]):
        super().__init__()

        # added initialization of channels_in so it can be accessed in forward
        self._channels_in = channels_in
        self._max_dimensions = max_dimensions
        
        self._height_tensor = nn.Parameter(torch.randn([1, channels_in // 8, height, 1]))
        self._width_tensor = nn.Parameter(torch.randn([1, channels_in // 8, 1, width]))
        self._gamma = nn.Parameter(torch.zeros(1))

        self._query_conv = nn.Conv2d(channels_in, channels_in // 8, 1)
        self._key_conv = nn.Conv2d(channels_in, channels_in // 8, 1)
        self._value_conv = nn.Conv2d(channels_in, channels_in, 1)
        
        self._attention_function = attention_function

    def forward(self, x: Tensor, x_encoder: Tensor) -> Tensor:
        query = self._query_conv(x)
        key = self._key_conv(x_encoder)
        value = self._value_conv(x_encoder)
        heat = (self._height_tensor + self._width_tensor).view(1, self._channels_in // 8, -1)
        attention = self._attention_function(query, key, value, heat, x.shape())
        return self._gamma * attention + x
