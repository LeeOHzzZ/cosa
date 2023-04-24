#!/usr/bin/env python3 
import argparse
import logging
import os
import pathlib
import time
from functools import reduce
from operator import mul
import json
import math

import numpy as np
import cosa.run_config as run_config
from cosa.cosa_constants import _A, _A_FLEXASR, _B, _B_HLSCNN, _B_FLEXASR
from cosa.cosa_input_objs import Prob, Arch, Mapspace
from gurobipy import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # capture everything
logger.disabled = False

try:
    _COSA_DIR = os.path.expanduser(os.environ['COSA_DIR'])
except KeyError:
    _COSA_DIR = os.path.abspath(__file__ + "/../")

hlscnn = 1
flexasr = 0
vta = 0
is_3la = hlscnn + flexasr + vta
assert hlscnn + flexasr + vta <= 1, "cannot set both hlscnn and flexasr to be True"

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir',
                        )
    parser.add_argument('-ap',
                        '--arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba.yaml',
                        )
    parser.add_argument('-mp',
                        '--mapspace_path',
                        type=str,
                        help='Mapspace Path',
                        default=f'{_COSA_DIR}/configs/mapspace/mapspace.yaml',
                        )
    parser.add_argument('-pp',
                        '--prob_path',
                        type=str,
                        help='Problem Dimension Path',
                        default=f'{_COSA_DIR}/configs/workloads/resnet50_graph/_outputs_input.2.yaml',
                        )
    return parser


def cosa(prob, arch, A, B, part_ratios, global_buf_idx, Z=None):
    """Run CoSA to generate a mapping with tiling, temporal/spatial, and permutation determined. 
        We currently assume there is a global buffer and only perform the permutation optimization
        at the global buffer level. Will add perm to all level in future version. 

    Args:
        prob: An object defines the layer dimension.
        arch: An object defines the hardware architecture dimension. 
        A: A 2d binary constant matrix that encodes the layer dimension to data tensor relationship.
            1 means related, 0 means unrelated
            Note that the R,S to the input tensor relation is specially handled in the formulation,
            and are specified to 0. 
        B: A 2d binary constant matrix that encodes the data tensor to memory level mapping. 
            It can be derived from the mapspace bypass pattern in Timeloop. 
            Note it is intended to be used for even mapping among different data tensors to different memory levels.
        part_ratios: A 2d array to represent the partition ratios of different data tensors 
            in different memory buffers. 
        global_buf_idx: An index point to the global buffer. 
        Z: Similar to B, but intended for uneven mapping among different data tensors to different memory levels.
            It is a 3d binary constant matrix that encodes the data tensor to memory level mapping.

    Returns: 
        factor_config: A 2d array specifying the allocation decision for each prime factor.
        spatial_config: A 2d array specifying the temporal/spatial decisions for each prime factor.
        perm_config: A 2d array specifyng the ordering of R,S,P,Q,C,K,N factors at each level.  
        run_time: Time-to-solution of CoSA.
    """
    # prime factors 
    prime_factors = prob.prob_factors
    # TODO: determine factor based on the padding and kernel value
    # larger_in_w = 1 if prob.prob["Padding"] > 0 or prob.prob["R"] > 1 else 0
    # larger_in_h = 1 if prob.prob["Padding"] > 0 or prob.prob["S"] > 1 else 0
    factor_in_w = prob.prob["Wstride"] * ((prob.prob["P"] - 1) * prob.prob["Wstride"] + prob.prob["R"] + 2 * prob.prob["Padding"]) / (prob.prob["P"] * prob.prob["Wstride"])
    factor_in_h = prob.prob["Hstride"] * ((prob.prob["Q"] - 1) * prob.prob["Hstride"] + prob.prob["S"] + 2 * prob.prob["Padding"]) / (prob.prob["Q"] * prob.prob["Hstride"])
    print(factor_in_w, factor_in_w)
    if factor_in_w > prob.prob["Wstride"]:
        factor_in_w *= 1.1
    if factor_in_h > prob.prob["Hstride"]:
        factor_in_h *= 1.1
    print(factor_in_w, factor_in_w)
    
    # strides = [prob.prob['Wstride'], prob.prob['Hstride'], prob.prob['Padding']]
    strides = [prob.prob['Wstride'], prob.prob['Hstride'], factor_in_w, factor_in_h]

    new_A = _A
    if flexasr or vta:
        new_A = _A_FLEXASR

    new_B = _B 
    if hlscnn:
        new_B = _B_HLSCNN
    elif flexasr or vta: # vta's B is exactly the same as flexasr's
        new_B = _B_FLEXASR

    if Z is None:
        Z = []
        for var in new_B:
            Z_var = []
            for i, val in enumerate(var):
                rank_arr = [0] * len(var)
                if val == 1:
                    for j in range(i + 1):
                        rank_arr[j] = 1
                Z_var.append(rank_arr)
            Z.append(Z_var)

    factor_config, spatial_config, outer_perm_config, run_time = mip_solver(prime_factors, strides, arch, part_ratios,
                                                                            global_buf_idx=global_buf_idx, A=new_A, Z=Z,
                                                                            compute_factor=10, util_factor=-0.1,
                                                                            traffic_factor=1)
    return factor_config, spatial_config, outer_perm_config, run_time


def mip_solver(f, strides, arch, part_ratios, global_buf_idx, A, Z, compute_factor=10, util_factor=-1,
               traffic_factor=1):
    """CoSA mixed integer programming(MIP) formulation."""

    logging.info(f"Prime factors {f}")
    logging.info(f"LAYER {f}")
    logging.info(f"A {A}")
    logging.info(f"Z {Z}")

    num_vars = len(A[0])
    num_mems = len(Z[0])

    env = Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = Model("mip", env=env)
    cost = []
    constraints = []

    org = ['spatial', 'temporal']

    M = []

    # ignore DRAM cap
    for i in range(num_mems - 1):
        mem_cap = arch.mem_entries[i]
        mem_cap_arr = []
        for j in range(num_vars):
            var_mem_cap = mem_cap * part_ratios[i][j]
            mem_cap_arr.append(var_mem_cap)
        M.append(mem_cap_arr)
    logging.info(f"M: {M}")
    # log friendly M
    M_log = []
    for i, mem in enumerate(M):
        M_v = []
        for bound in mem:
            if bound == 0:
                # turn 0 to 1 for taking the log
                bound = 1
            M_v.append(bound)
        M_log.append(M_v)
    logging.info(f"M_log: {M_log}")
    # spatial constraints
    S = arch.S

    # =================================================================

    # set the levels to be equal to the number of factors + 4 memory levels 
    perm_levels = 0
    for j, f_j in enumerate(f):
        perm_levels += len(f_j)
    gb_start_level = global_buf_idx

    total_levels = num_mems - 1 + perm_levels
    logging.info(f"num_mems: {num_mems}; perm_levels: {perm_levels}")
    logging.info(f"total {total_levels} levels")

    # ============== spatial and temporal exclusive constraints ==== #
    x = {}  # x_jn_jn
    for i in range(total_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    x[(i, j, n, k)] = m.addVar(vtype=GRB.BINARY, name=name)
                # sum for each sub factor spatial and temp must be less than 1 
                # NOT equals to one
                spatial_temp_sum = 0
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    spatial_temp_sum += x[(i, j, n, k)]
                m.addConstr(spatial_temp_sum <= 1, "spatial_temp_sum_{}_{}_{}".format(i, j, n))
    # logging.info(f"x: {x}")
    # j, n is the loop level 

    # ============== each mapper must have a mapping ================= #
    i = 0
    x_row_sums = []
    x_col_sums = []
    # for i in range(total_levels):
    for i in range(gb_start_level, gb_start_level + perm_levels):
        row_sum = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    row_sum += x[(i, j, n, k)]
        m.addConstr(row_sum <= 1, "row_sum_{}".format(i))
        x_row_sums.append(row_sum)

    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            col_sum = 0
            for i in range(total_levels):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    col_sum += x[(i, j, n, k)]
            # assume perm can be interleaved in diff perm level
            m.addConstr(col_sum == 1, "col_sum_{}_{}".format(j, n))
            x_col_sums.append(col_sum)

            # make sure v is one for all outer loop level, once a correlation exists
    # add another relation var v - f, 3 - 7*n loop-level
    # whether the type of variable has related loops at the certaion permutation level?
    # temporal iteration term in the paper (the third term in calculating traffic)
    s = {}
    y = {}
    for v in range(num_vars):
        for i in range(gb_start_level, gb_start_level + perm_levels):
            row_sum = 0
            y[(v, i)] = m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name="y({},{})".format(v, i))
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    row_sum += x[(i, j, n, 1)] * A[j][v]
            if i > gb_start_level:
                m.addConstr(y[(v, i)] >= y[(v, i - 1)], "y_v_i_sv_{}_{}".format(v, i))
                # can be ==
                m.addConstr(y[(v, i)] >= row_sum, "y_v_i_row_sum_{}_{}".format(v, i))
            else:
                # can be ==
                m.addConstr(y[(v, i)] == row_sum, "y_v_i_row_sum_{}_{}".format(v, i))
            s[(v, i)] = row_sum

    # ========= No idea what this is about ============= #
    ## exhausively list all scenarios where p or q is inside current mem
    if not is_3la:
        InputBufferLevel = 3
        zz = {}
        prefix = 0
        for var in [2, 3]: # this var is referring the P and Q instead of input and output var
            for mem_level in [InputBufferLevel]:
                zz[(var, mem_level)] = m.addVar(lb=0, ub=1, vtype=GRB.INTEGER,
                                                name="zz({},{},{})".format(prefix, var, mem_level))
                x_sums = 0
                for n, prime_factor in enumerate(f[var]):
                    for inner_mem_level_i in range(mem_level + 1):
                        for k in range(2):
                            filter_in = x[(inner_mem_level_i, var, n, k)]
                            m.addConstr(zz[(var, mem_level)] >= filter_in,
                                        "zz_x_sum_{}_{}_{}_{}_{}_{}".format(prefix, var, n, mem_level, inner_mem_level_i,
                                                                            k))
                            x_sums += filter_in
                m.addConstr(zz[(var, mem_level)] <= x_sums, "z_x_sum_{}_{}_{}".format(prefix, var, mem_level))

    l = {}
    for v in range(num_vars):
        for i in range(gb_start_level, gb_start_level + perm_levels):
            row_sum = 0
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    row_sum += np.log2(f[j][n]) * (x[(i, j, n, 1)])
            l[(v, i)] = row_sum
    
    if not is_3la:
        # Add spatial constraints
        spatial_tile = 0
        for i in range(gb_start_level, gb_start_level + perm_levels):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
        m.addConstr(spatial_tile <= np.log2(S[gb_start_level]), "spatial_tile_gb_{}".format(prefix))

        for i in range(gb_start_level):
            spatial_tile = 0
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
            m.addConstr(spatial_tile <= np.log2(S[i]), f"spatial_tile_{prefix}_{i}")

        for i in range(gb_start_level + perm_levels, total_levels):
            spatial_tile = 0
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
            m.addConstr(spatial_tile <= np.log2(S[i - perm_levels + 1]), f"spatial_tile_{i - perm_levels + 1}")

    # ========== memory sizes constraints ============== # 
    # Add inner gb buffer constraints
    buf_util = {}
    for v in range(num_vars):
        for i in range(num_mems):
            buf_util[(i, v)] = 0
    logging.debug(f"buf_util: {buf_util}")

    for v in range(num_vars):
        for i_ in range(gb_start_level + perm_levels):
            for i in range(num_mems):
                for j, f_j in enumerate(f):
                    for n, f_jn in enumerate(f_j):
                        factor = 1
                        # stride[2, 3] contains larger input_w, input_h info
                        # If there is no padding, the input image size is probably larger than the output size
                        if v == 1 and j == 2:
                            # factor = strides[0] + 0.1 if strides[2] else strides[0]
                            # factor = strides[2]
                            factor = strides[0]
                        if v == 1 and j == 3:
                            factor = strides[1]
                            # factor = strides[3]

                        if i_ > gb_start_level and i_ < gb_start_level + perm_levels:
                            Z_const = Z[v][i][gb_start_level]
                        else:
                            Z_const = Z[v][i][i_]
                        buf_util[(i, v)] += np.log2(factor * f[j][n]) * (x[(i_, j, n, 0)] + x[i_, j, n, 1]) * A[j][
                            v] * Z_const  # use the i for the cur mem for relationship 
                        # only add once
                        if not is_3la:
                            if i == 3 and j in [0, 1] and v == 1:
                                buf_util[(i, v)] += (x[(i_, j, n, 0)] + x[(i_, j, n, 1)]) * (1 - zz[(j + 2, i)]) * np.log2(
                                    f[j][n])
                                buf_util[(i, v)] += (x[(i_, j, n, 0)] + x[(i_, j, n, 1)]) * zz[(j + 2, i)] * np.log2(2)

    for v in range(num_vars):
        # excluding DRAM
        for i in range(num_mems - 1):
            if M_log[i][v] > 0:
                m.addConstr(buf_util[(i, v)] <= np.log2(M_log[i][v]), f"buffer_size_{i}_{v}")


    if hlscnn:
        # disable spatial 
        for i in range(total_levels):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    m.addConstr(x[(i, j, n, 0)] == 0, f"disable_spatial_{i}_{j}_{n}")
        # kernel row
        k_row_sum = 0
        k_col_sum = 0
        in_chan_size = 0
        out_chan_size = 0
        # for i in range(gb_start_level):
        for n, f_jn in enumerate(f[0]):
            k_row_sum += x[(0,0,n,1)]
        for n, f_jn in enumerate(f[1]):
            k_col_sum += x[(0,1,n,1)]
        for n, f_jn in enumerate(f[4]):
            in_chan_size += np.log2(f_jn) * x[(0,4,n,1)]
        for n, f_jn in enumerate(f[5]):
            out_chan_size += np.log2(f_jn) * x[(0,5,n,1)]
            
        m.addConstr(k_row_sum == len(f[0]), "hlscnn_k_row_in_spad")
        m.addConstr(k_col_sum == len(f[1]), "hlscnn_k_col_in_spad")
        # m.addConstr(in_chan_size >= 3, "hlscnn_in_chan")
        m.addConstr(in_chan_size <= 10, "hlscnn_max_in_chan")
        # m.addConstr(out_chan_size >= 3, "hlscnn_out_chan")
        m.addConstr(out_chan_size <= 8, "hlscnn_max_out_chan")

    if flexasr:
        # disable spatial 
        for i in range(total_levels):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    m.addConstr(x[(i, j, n, 0)] == 0, f"disable_spatial_{i}_{j}_{n}")
        # set the batch dimension to multiple of 16
        t_batch_num = 0
        for n, f_jn in enumerate(f[2]):
            t_batch_num += x[(0,2,n,1)] * np.log2(f_jn)
        tb_factor = m.addVar(lb=4, vtype=GRB.INTEGER, name=f"flexasr_t_batch_factor")
        m.addConstr(t_batch_num == tb_factor, "t_batch num be integer")
    
    if vta:
        # disable spatial 
        for i in range(total_levels):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    m.addConstr(x[(i, j, n, 0)] == 0, f"disable_spatial_{i}_{j}_{n}")
        # set the tx and ty to multiple of 16
        t_x_num, t_y_num = 0, 0
        for n, f_jn in enumerate(f[3]):
            t_y_num += x[(0,3,n,1)] * np.log2(f_jn)
        for n, f_jn in enumerate(f[4]):
            t_x_num += x[(0,4,n,1)] * np.log2(f_jn)
        tx_factor = m.addVar(lb=4, vtype=GRB.INTEGER, name="vta_tx_factor")
        ty_factor = m.addVar(lb=4, vtype=GRB.INTEGER, name="vta_ty_factor")
        m.addConstr(tx_factor == t_x_num, "tx be multiple of 16")
        m.addConstr(ty_factor == t_y_num, "ty be multiple of 16")


    # get compute cost 
    inner_gb_cycles = 0
    for i in range(gb_start_level):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                inner_gb_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])

    gb_cycles = 0
    for i in range(gb_start_level, gb_start_level + perm_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                gb_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])

    dram_cycles = 0
    for i in range(gb_start_level + perm_levels, total_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                dram_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])
    total_compute = inner_gb_cycles + gb_cycles + dram_cycles
    gb_compute = inner_gb_cycles + gb_cycles

    # get traffic cost
    spatial_cost = {}
    for v in range(num_vars):
        size = 0
        for i in range(gb_start_level, gb_start_level + perm_levels):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    size += np.log2(f[j][n]) * (x[(i, j, n, 0)])
        spatial_cost[v] = size

    data_size = {}
    for v in range(num_vars):
        size = 0
        for i in range(gb_start_level):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    # TRICK prioritize spatial
                    factors = 0.8 + 0.04 * i if not is_3la else 1
                    size += factors * np.log2(f[j][n]) * (x[(i, j, n, 0)] + x[i, j, n, 1]) * A[j][v]
        data_size[v] = size

    gb_traffic = {}
    for v in range(num_vars):
        size = 0
        for i in range(gb_start_level, gb_start_level + perm_levels):
            size += l[(v, i)] * y[(v, i)]
        gb_traffic[v] = size

        # use the last level gb y for DRAM 
    dram_traffic = {}
    for v in range(num_vars):
        corr = y[(v, gb_start_level + perm_levels - 1)]
        i = gb_start_level + perm_levels  # DRAM 
        size = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                size += np.log2(f[j][n]) * (x[(i, j, n, 1)])  # * corr 
        dram_traffic[v] = size

    total_util = 0
    for i in range(gb_start_level):
        # for each memory and each variable there is a constraint
        for v in range(num_vars):
            # make weight util more important since it directly comes from dram
            factor = 1.01 if i == 2 else 1
            total_util += buf_util[(i, v)] * factor

    total_traffic = 0
    for v in range(num_vars):
        #  TRICKS
        if v == 0:
            # encode dram latency for weights
            factor = 1.01
        else:
            factor = 1
        
        # if hlscnn:
        #     total_traffic += data_size[v] + gb_traffic[v] + dram_traffic[v] * factor
        # else:
        total_traffic += 0.99 * data_size[v] + 0.99 * spatial_cost[v] + gb_traffic[v] + dram_traffic[v] * factor

    # ========================== user-defined objective function ========================== #
    # cosa_obj = total_util * util_factor + total_compute * compute_factor + total_traffic * traffic_factor
    cosa_obj = total_traffic
    max_it = m.addVar(vtype=GRB.CONTINUOUS, name="max_it")
    its = []
    its.append(m.addVar(vtype=GRB.CONTINUOUS, name="a"))
    m.addConstr(its[-1] == total_traffic, "total_traffic")
    its.append(m.addVar(vtype=GRB.CONTINUOUS, name="b"))
    m.addConstr(its[-1] == total_compute, "total_compute")
    m.addConstr(max_it == max_(its), name="max_it_constr")

    total_util_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_util_var")
    total_comp_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_comp_var")
    total_traf_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_traf_var")

    # cycle count = total max 3 * all log factors variables 
    m.addConstr(total_util_var == total_util, "total_util_constraint")
    m.addConstr(total_comp_var == total_compute, "total_comp_constraint")
    m.addConstr(total_traf_var == total_traffic, "total_traf_constraint")

    m.ModelSense = GRB.MINIMIZE
    m.setObjective(cosa_obj, GRB.MINIMIZE)


    # optimize for the objective function
    milp_time = 0
    begin_time = time.time()
    m.optimize()
    end_time = time.time()
    milp_runtime = end_time - begin_time

    # if m.status == GRB.INFEASIBLE:
    #     iis_constr = []
    #     while (m.status == GRB.INFEASIBLE):
    #         m.computeIIS()
    #         for c in m.getConstrs():
    #             if c.IISConstr:
    #                 iis_constr.append(c)
    #                 m.remove(c)
    #                 break
    #         m.optimize()
            
    #     print("IIS:", iis_constr)

    # output all constraints and variables
    m.write("debug.lp")

    result_dict = {}
    for variable in m.getVars():
        # logging.debug("Variable %s: Value %s" % (variable.varName, variable.x))
        assert (variable.varName not in result_dict)
        # print(variable, variable.varName, variable.x)
        result_dict[variable.varName] = variable.x
    logging.debug('Obj: %g' % m.objVal)
    # logging.debug("result_dict", result_dict)
    # print(result_dict)

    # validate buf utilization
    buf_util_res = {}
    for v in range(num_vars):
        for i in range(num_mems):
            buf_util_res[(i, v)] = 0
    for v in range(num_vars):
        for i_ in range(gb_start_level + perm_levels):
            for i in range(num_mems):
                for j, f_j in enumerate(f):
                    for n, f_jn in enumerate(f_j):
                        factor = 1
                        # stride[2, 3] contains larger input_w, input_h info
                        # If there is no padding, the input image size is probably larger than the output size
                        if v == 1 and j == 2:
                            # factor = strides[0] + 0.1 if strides[2] else strides[0]
                            factor = strides[2]
                        if v == 1 and j == 3:
                            factor = strides[3]

                        if i_ > gb_start_level and i_ < gb_start_level + perm_levels:
                            Z_const = Z[v][i][gb_start_level]
                        else:
                            Z_const = Z[v][i][i_]

                        buf_util_res[(i, v)] += np.log2(factor * f[j][n]) * (result_dict[f"X({i_},{j},{n},0)"] + result_dict[f"X({i_},{j},{n},1)"]) * A[j][
                            v] * Z_const  # use the i for the cur mem for relationship 
    print(buf_util_res)
    for v in range(num_vars):
        # excluding DRAM
        for i in range(num_mems - 1):
            if M_log[i][v] > 0:
                print(Z[v][i])
                print(f"({i}, {v})::{buf_util_res[(i, v)]} <? {np.log2(M_log[i][v])}\n")

    all_x = np.zeros((total_levels, perm_levels, 2))
    for i in range(total_levels):
        level_idx = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                if hlscnn:
                    assert result_dict[f"X({i},{j},{n},0)"] == 0
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    all_x[i, level_idx, k] = val
                level_idx += 1
    np.set_printoptions(precision=0, suppress=True)

    var_outer_perm_config = [-1] * perm_levels
    outer_perm_config = [-1] * perm_levels
    x_arr = np.zeros((perm_levels, perm_levels, 2))
    for i in range(gb_start_level, gb_start_level + perm_levels):
        level_idx = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    x_arr[i - gb_start_level, level_idx, k] = val
                name = "X({},{},{},{})".format(i, j, n, 1)
                val = result_dict[name]
                if val == 1:
                    var_outer_perm_config[i - gb_start_level] = j
                level_idx += 1
    logging.info(f'var_outer_perm_config: {var_outer_perm_config}')

    y_arr = np.zeros((num_vars, perm_levels))
    for v in range(num_vars):
        for i in range(gb_start_level, gb_start_level + perm_levels):
            row_sum = 0
            val = result_dict["y({},{})".format(v, i)]
            y_arr[v, i - gb_start_level] = val

    # Merge the permutation, taking the first appearance of a prob to be the
    merge_outer_perm_config = []
    for i, var in enumerate(var_outer_perm_config):
        if var != -1 and var not in merge_outer_perm_config:
            merge_outer_perm_config.append(var)

    for i in range(len(f)):
        if i not in merge_outer_perm_config:
            merge_outer_perm_config.append(i)

    logging.info("var idx as the value {}".format(var_outer_perm_config))
    logging.info("merged var idx as the value {}".format(merge_outer_perm_config))

    outer_perm_config = [1] * len(f)
    for i, var in enumerate(merge_outer_perm_config):
        outer_perm_config[var] = i

    logging.info("ordering idx as the value {}".format(outer_perm_config))

    # init factor_config 
    # DRAM is the last level
    factor_config = []
    spatial_config = []
    dram_level = -1
    for j, f_j in enumerate(f):
        sub_factor_config = []
        sub_spatial_config = []
        for n, f_jn in enumerate(f_j):
            sub_factor_config.append(dram_level)
            sub_spatial_config.append(0)
        factor_config.append(sub_factor_config)
        spatial_config.append(sub_spatial_config)

    for i in range(gb_start_level):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                if f[j][n] == 1:
                    factor_config[j][n] = num_mems - 1
                    spatial_config[j][n] = 0
                    continue
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    if val >= 0.9:
                        factor_config[j][n] = i
                        if k == 0:
                            spatial_config[j][n] = 1

    for i in range(gb_start_level + perm_levels, total_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                if f[j][n] == 1:
                    factor_config[j][n] = num_mems - 1
                    spatial_config[j][n] = 0
                    continue

                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    if val >= 0.9:
                        if k == 0:
                            raise ValueError('Invalid Mapping')
                        factor_config[j][n] = i - perm_levels + 1

    # set to -1 for not specified 
    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            for i in range(gb_start_level, gb_start_level + perm_levels):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    val = result_dict[name]
                    if val >= 0.9:
                        factor_config[j][n] = gb_start_level
                        if k == 0:
                            spatial_config[j][n] = 1

    logging.info(f"prime factors: {f}")
    logging.info(f"factor configs: {factor_config}")
    logging.info(f"spatial configs: {spatial_config}")
    logging.info(f"milp runtime: {milp_runtime}")

    return (factor_config, spatial_config, outer_perm_config, milp_runtime)


def run_timeloop(prob_path, arch_path, mapspace_path, output_path):
    # init
    status_dict = {}
    prob = Prob(prob_path)
    arch = Arch(arch_path)

    # An object defines the user-defined bypass pattern. 
    mapspace = Mapspace(mapspace_path)
    mapspace.init(prob, arch)

    # even mapping
    B = _B if not hlscnn else _B_HLSCNN
    Z = None

    # uneven mapping config
    # Z = _Z
    # B = None
 
    # partition ratios for W, IA, OA
    part_ratios = [
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0.25, 0.75],
        [0.33, 0.33, 0.33],
    ]

    if hlscnn:
        part_ratios = [
            [0.5, 0.25, 0.25],
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
        ]
    if flexasr:
        part_ratios = [
            [0.8, 0.1, 0.1],
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
        ]
    if vta:
        part_ratios = [
            [0.8, 0.1, 0.1],
            [0.33, 0.33, 0.33],
            [0.33, 0.33, 0.33],
        ] 

    global_buf_idx = 4
    if is_3la:
        global_buf_idx = 1

    factor_config, spatial_config, outer_perm_config, run_time = cosa(prob, arch, _A, B, part_ratios, global_buf_idx=global_buf_idx,
                                                                      Z=Z)

    update_factor_config = factor_config
    spatial_to_factor_map = {}
    idx = arch.mem_levels
    for i, val in enumerate(arch.S):
        if val > 1:
            spatial_to_factor_map[i] = idx
            idx += 1
    logging.info(f'spatial_to_factor_map: {spatial_to_factor_map}')

    for j, f_j in enumerate(prob.prob_factors):
        for n, f_jn in enumerate(f_j):
            # if is mapped to spatial, look up the combined index
            if spatial_config[j][n] == 1:
                idx = factor_config[j][n]
                update_factor_config[j][n] = spatial_to_factor_map[idx]

    logging.info(f'update_factor_config: {update_factor_config}')
    perm_config = mapspace.get_default_perm()
    logging.info(f"default perm_config: {perm_config}")
    logging.info(f"outer_per_config: {outer_perm_config}")

    if hlscnn:
        assert all(
            [True if i == 0 or all([j == 1 for j in prob.prob_factors[0]]) 
             else False for i in update_factor_config[0]]
        ), (
            "Result has tiling on kernel width dimension which is not allowed for HLSCNN"
        )
        assert all(
            [True if i == 0 or all([j == 1 for j in prob.prob_factors[1]]) 
             else False for i in update_factor_config[1]]
        ), (
            "Result has tiling on kernel height dimension which is not allowed for HLSCNN"
        )
        # assert all([2 not in i for i in update_factor_config[:-1]]) # the pseudo-DRAM should not be used
        fn = lambda f, level: f if level == 0 else 1
        tile_size_dict = {
            "tw" : reduce(mul, map(fn, prob.prob_factors[2], update_factor_config[2])),
            "th" : reduce(mul, map(fn, prob.prob_factors[3], update_factor_config[3])),
            "tc" : reduce(mul, map(fn, prob.prob_factors[4], update_factor_config[4])),
            "tk" : reduce(mul, map(fn, prob.prob_factors[5], update_factor_config[5])),
        }
        logging.info(f"tile size dict: {tile_size_dict}")
        dim_order_dict = {"w": outer_perm_config[2], "h": outer_perm_config[3], "c": outer_perm_config[4], "k": outer_perm_config[5]}
        loopOrder = [k for k, _ in sorted(dim_order_dict.items(), key = lambda x : x[1])]
        logging.info(f"LoopOrder: {loopOrder}")

        results = {
            "tile_sizes" : tile_size_dict,
            "loopOrder" : loopOrder,
            "timeToSolution": run_time,
        }

        with open(output_path, "w") as fout:
            json.dump(results, fout, indent=4)
        
        logging.info(f"result has been dumped to {output_path}")

    elif flexasr or vta:
        fn = lambda f, level: f if level < 1 else 1
        tile_size_dict = {
            "tb" : reduce(mul, map(fn, prob.prob_factors[2], update_factor_config[2])),
            "ty" : reduce(mul, map(fn, prob.prob_factors[3], update_factor_config[3])),
            "tx" : reduce(mul, map(fn, prob.prob_factors[4], update_factor_config[4])),
        }
        logging.info(f"tile size dict: {tile_size_dict}")
        dim_order_dict = {"t": outer_perm_config[2], "y": outer_perm_config[3], "x": outer_perm_config[4]}
        loopOrder = [k for k, _ in sorted(dim_order_dict.items(), key = lambda x : x[1])]
        logging.info(f"LoopOrder: {loopOrder}")

        results = {
            "tile_sizes" : tile_size_dict,
            "loopOrder" : loopOrder,
            "timeToSolution": run_time,
        }

        with open(output_path, "w") as fout:
            json.dump(results, fout, indent=4)
        
        logging.info(f"result has been dumped to {output_path}")
        


    if not is_3la:
        perm_config[4] = outer_perm_config

        status_dict = {}
        try:
            spatial_configs = []
            results = run_config.run_config(mapspace, None, perm_config, update_factor_config, status_dict,
                                            run_gen_map=True, run_gen_tc=False, run_sim_test=False, output_path=output_path,
                                            spatial_configs=spatial_configs, valid_check=False, outer_loopcount_limit=100)
            logging.info(f'status_dict: {status_dict}')
        except:
            logging.error('Error: invalid schedule.')

        return status_dict

def run_cosa():
    parser = construct_argparser()
    args = parser.parse_args()

    prob_path = pathlib.Path(args.prob_path).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    mapspace_path = pathlib.Path(args.mapspace_path).resolve()
    output_path = args.output_dir

    run_timeloop(prob_path, arch_path, mapspace_path, output_path)


if __name__ == "__main__":
    run_cosa()