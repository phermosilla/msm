# Copyright 2021 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import numpy as np
import torch 
from cumm import tensorview as tv 
from spconv.pytorch.cppcore import torch_tensor_to_tv, get_current_stream

from spconv.core_cc.csrc.hash.core import HashTable as _HashTable

from pointcept.models.utils.serialization import encode

_TORCH_DTYPE_TO_ITEMSIZE = {
    torch.int32: 4,
    torch.int64: 8,
    torch.float32: 4,
    torch.float64: 8,
}

class HashTable:
    """simple hash table for 32 and 64 bit data. support both cpu and cuda.
    for cuda, it's a fixed-size table, you must provide maximum size 
    (recommend 2 * num).
    key must be int32/int64.
    see spconv/pytorch/functional/sparse_add_hash_based, a real example
    that show how to use hash table to implement 
    sparse add (same shape, different indices)
    """
    def __init__(self, device: torch.device, key_dtype: torch.dtype, 
                value_dtype: torch.dtype, 
                max_size: int = -1) -> None:
        is_cpu = device.type == "cpu"
        self.is_cpu = is_cpu
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        key_data_tv = tv.Tensor()
        value_data_tv = tv.Tensor()
        if is_cpu:
            self.keys_data = None 
            self.values_data = None 
        else:
            assert max_size > 0, "you must provide max_size for fixed-size cuda hash table, usually *2 of num of keys"
            assert device is not None, "you must specify device for cuda hash table."
            self.keys_data = torch.empty([max_size], dtype=key_dtype, device=device)
            self.values_data = torch.empty([max_size], dtype=value_dtype, device=device)
            key_data_tv = torch_tensor_to_tv(self.keys_data)
            value_data_tv = torch_tensor_to_tv(self.values_data)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        self.key_itemsize = _TORCH_DTYPE_TO_ITEMSIZE[self.key_dtype]
        self.value_itemsize = _TORCH_DTYPE_TO_ITEMSIZE[self.value_dtype]
        self._valid_value_dtype_for_arange = set([torch.int32, torch.int64])

        self._table = _HashTable(is_cpu, self.key_itemsize, self.value_itemsize, key_data_tv, value_data_tv, stream)


    def insert(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None):
        """insert hash table by keys and values
        if values is None, only key is inserted, the value is undefined.
        """
        keys_tv = torch_tensor_to_tv(keys)
        values_tv = tv.Tensor()
        if values is not None:
            values_tv = torch_tensor_to_tv(values)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()

        return self._table.insert(keys_tv, values_tv, stream)

    def query(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None):
        """query value by keys, if values is not None, create a new one.
        return values and a uint8 tensor that whether query fail.
        """
        keys_tv = torch_tensor_to_tv(keys)
        if values is None:
            values = torch.empty([keys.shape[0]], dtype=self.value_dtype, device=keys.device)
        values_tv = torch_tensor_to_tv(values)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        is_empty = torch.empty([keys.shape[0]], dtype=torch.uint8, device=keys.device)
        is_empty_tv = torch_tensor_to_tv(is_empty)
        self._table.query(keys_tv, values_tv, is_empty_tv, stream)
        return values, is_empty > 0

    def insert_exist_keys(self, keys: torch.Tensor, values: torch.Tensor):
        """insert kv that k exists in table. return a uint8 tensor that
        whether insert fail.
        """
        keys_tv = torch_tensor_to_tv(keys)
        values_tv = torch_tensor_to_tv(values)
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        is_empty = torch.empty([keys.shape[0]], dtype=torch.uint8, device=keys.device)
        is_empty_tv = torch_tensor_to_tv(is_empty)
        self._table.insert_exist_keys(keys_tv, values_tv, is_empty_tv, stream)
        return is_empty

    def assign_arange_(self):
        """iterate table, assign values with "arange" value.
        equivalent to 1. get key by items(), 2. use key and arange(key.shape[0]) to insert
        """
        count_tv = tv.Tensor()
        count = torch.Tensor()
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        else:
            assert self.value_dtype in self._valid_value_dtype_for_arange
        if not self.is_cpu:
            assert self.values_data is not None
            if self.key_itemsize == 4:
                count = torch.zeros([1], dtype=torch.int32, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint32)
            elif self.key_itemsize == 8:
                count = torch.zeros([1], dtype=torch.int64, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint64)
            else:
                raise NotImplementedError
        else:
            max_size = self._table.size_cpu()
            count = torch.tensor([max_size], dtype=torch.int64)

        self._table.assign_arange_(count_tv, stream)
        return count

    def items(self, max_size: int = -1):
        count_tv = tv.Tensor()
        count = torch.Tensor()
        stream = 0
        if not self.is_cpu:
            stream = get_current_stream()
        if not self.is_cpu:
            assert self.values_data is not None
            if self.key_itemsize == 4:
                count = torch.zeros([1], dtype=torch.int32, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint32)
            elif self.key_itemsize == 8:
                count = torch.zeros([1], dtype=torch.int64, device=self.values_data.device)
                count_tv = torch_tensor_to_tv(count, dtype=tv.uint64)
            else:
                raise NotImplementedError
        if not self.is_cpu:
            assert self.values_data is not None
            if max_size == -1:
                max_size = self.values_data.shape[0]
            keys = torch.empty([max_size], dtype=self.key_dtype, device=self.values_data.device)
            values = torch.empty([max_size], dtype=self.value_dtype, device=self.values_data.device)

        else:
            max_size = self._table.size_cpu()
            count = torch.tensor([max_size], dtype=torch.int64)
            keys = torch.empty([max_size], dtype=self.key_dtype)
            values = torch.empty([max_size], dtype=self.value_dtype)
        keys_tv = torch_tensor_to_tv(keys)
        values_tv = torch_tensor_to_tv(values)
        self._table.items(keys_tv, values_tv, count_tv, stream)
        return keys, values, count

def ravel_hash_vec(arr, arr_max):
    assert arr.ndim == 2
    arr = arr.clone()
    arr = arr.to(torch.int64)
    keys = torch.zeros((arr.shape[0],)).to(torch.int64).to(arr.device)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys

@torch.no_grad()
def match_3d_coords(key_indices, query_indices, 
                    key_code = None, query_code = None):
    
    #arr_max = torch.amax(torch.cat((key_indices, query_indices), 0), 0)+1

    if key_code is None:
        code_key = encode(
            key_indices[:,1:], 
            batch=key_indices[:,0], 
            depth=16, order="z")
        #code_key = ravel_hash_vec(key_indices, arr_max)
    else:
        code_key = key_code
    
    if query_code is  None:
        code_query = encode(
            query_indices[:,1:], 
            batch=query_indices[:,0], 
            depth=16, order="z")
        #code_query = ravel_hash_vec(query_indices, arr_max)
    else:
        code_query = query_code

    my_table = HashTable(
        code_key.device, 
        key_dtype = torch.int64, 
        value_dtype = torch.int64,
        max_size = code_key.shape[0]*2)
    my_table.insert(code_key, 
                    torch.arange(code_key.shape[0], 
                    device=code_key.device))
    return my_table.query(code_query)


class CosineScheduler(torch.nn.Module):
    def __init__(self, 
                 base_value, 
                 final_value, 
                 total_iters, 
                 warmup_iters=0, 
                 start_warmup_value=0, 
                 freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        self.register_buffer('cur_iter', torch.tensor(0))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self):
        it = self.cur_iter.item()
        self.cur_iter = torch.tensor(it + 1)
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]