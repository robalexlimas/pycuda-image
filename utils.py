# Libraries
import numpy as np
import pycuda.driver as cuda
import os

from string import Template


def copy_host_to_device(*hots_variables):
    device_mem_allocations = []
    for index in range(len(hots_variables)):
        # Memory allocation
        device_mem_allocation = memory_allocation(hots_variables[index].astype(np.float32))
        # Copy host information to device
        cuda.memcpy_htod(device_mem_allocation, hots_variables[index])
        device_mem_allocations.append(device_mem_allocation)
    return device_mem_allocations if len(device_mem_allocations) > 1 else device_mem_allocation


def memory_allocation(hots_variable):
    return cuda.mem_alloc(hots_variable.nbytes)


def kernel_creation(path, **kernel_parameters):
    parameters = kernel_parameters["kernel_parameters"]
    path = os.path.join(path, "templates.cpp")
    template = Template(_get_template(path))
    return template.safe_substitute(**parameters)


def _get_template(path):
    with open(path, "r") as file:
        template = file.readlines()
    return "".join(template)