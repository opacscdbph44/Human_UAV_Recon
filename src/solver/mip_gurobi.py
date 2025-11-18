from dataclasses import dataclass
from datetime import datetime

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.element import InstanceClass, Solution
from src.evaluator import Population


@dataclass
class GurobiVariables:
    """Gurobi模型变量容器"""

    # 路径变量: 车辆k从节点i到节点j的访问路径
    x: gp.tupledict
    # 通信变量: 节点i的通信保障是否由车辆k在节点j处执行
    z: gp.tupledict
    # 任务执行变量: 节点i处任务是否被执行
    delta: gp.tupledict
    # 到达时间变量: 车辆k在节点i的访问时间
    arr_t: gp.tupledict
    # 剩余能量变量: 车辆k在节点i时的剩余续航
    omega: gp.tupledict
    # Makespan变量: 所有车辆返回基地的最晚时刻
    tau: gp.Var

    def __str__(self):
        return (
            f"GurobiVariables(x={self.x}, z={self.z}, delta={self.delta}, "
            f"arr_t={self.arr_t}, omega={self.omega}, tau={self.tau})"
        )

    __repr__ = __str__


class GurobiMIPSolver:
    """Gurobi MIP求解器基类"""

    def __init__(
        self,
        solution: Solution,
        instance: InstanceClass,
    ):
        # 存入解对象、算例对象
        self.solution = solution
        self.instance = instance

        # 提取必要的集合和参数
        # ---------- 节点集合 ----------
        self.id_base = list(instance.base_ids)
        self.id_demand = list(instance.demand_ids)
        accessible_flag = instance.accessible
        self.id_isolated = [i for i in self.id_demand if not accessible_flag[i]]
        self.id_connected = [i for i in self.id_demand if accessible_flag[i]]
        self.id_steiner = list(instance.steiner_ids)
        self.id_all_node = list(range(instance.total_node_num))
        self.id_node_no_base = self.id_demand + self.id_steiner

        # ---------- 车辆集合 ----------
        self.id_Ground = instance.ground_veh_ids
        self.id_Drone = instance.drone_ids
        self.id_vehicle = list(range(instance.total_veh_num))

        # ---------- 矩阵提取 ----------
        self.distance_matrix = instance.distance_matrix
        self.coverage_matrix = instance.comm_coverage_matrix
        self.travel_time_matrix = instance.travel_time_matrix
        self.travel_cost_matrix = instance.travel_consum_matrix
        self.comm_cost_matrix = instance.comm_consum_matrix

        # --------- 优先级得分 ----------
        self.node_score_list = instance.priority
        self.min_comm_time = instance.prob_config.instance_param.min_comm_time

        # --------- 求解器参数 ----------
        self.M = instance.prob_config.algorithm_config.big_m
        self.time_limit = instance.prob_config.algorithm_config.time_limit

        # 设置目标函数信息
        obj_config = instance.prob_config.algorithm_config.obj_config
        obj_names = obj_config.name
        obj_sense = obj_config.sense
        self.objective_info = list(zip(obj_names, obj_sense))

    def _create_model(self, model_name: str = "MIP_Optimizer"):
        """创建Gurobi模型"""
        self.model = gp.Model(model_name)

    # 创建Gurobi变量
    def _create_variables(self):
        # Vars: x_k_i,j: 车辆k的访问路径
        x = self.model.addVars(
            self.id_vehicle,
            self.id_all_node,
            self.id_all_node,
            vtype=GRB.BINARY,
            name="x",
        )
        # Vars: z_k_i,j: 节点i的通信保障是否由车辆k在节点j处执行
        z = self.model.addVars(
            self.id_vehicle,
            self.id_node_no_base,
            self.id_node_no_base,
            vtype=GRB.BINARY,
            name="z",
        )

        # Vars: delta_i, 节点i处任务是否被执行
        delta = self.model.addVars(
            self.id_demand, name="delta", vtype=GRB.BINARY, lb=0, ub=1
        )

        # Vars: a_k_i: 车辆k在i点的访问时间
        arr_t = self.model.addVars(
            self.id_vehicle, self.id_all_node, vtype=GRB.CONTINUOUS, lb=0, name="a"
        )
        # Vars: tau: 所有车辆返回基地的最晚时刻
        tau = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="tau")
        # Vars: omega_k_i: 车辆k在i点时的剩余续航
        omega = self.model.addVars(
            self.id_vehicle, self.id_all_node, vtype=GRB.CONTINUOUS, lb=0, name="omega"
        )

        self.variables = GurobiVariables(
            x=x,
            z=z,
            delta=delta,
            arr_t=arr_t,
            tau=tau,
            omega=omega,
        )

    def _get_makespan_objective_expr(self) -> gp.LinExpr:
        # 检查变量对象是否存在
        if self.variables is None:
            error_msg = "变量未创建"
            raise ValueError(error_msg)

        # 检查变量 tau 是否为 None
        if self.variables.tau is None:
            error_msg = "变量 'tau' 为空"
            raise ValueError(error_msg)

        # 获取 tau 变量
        tau = self.variables.tau

        # 检查必要的id集合是否已定义
        if not (self.id_demand and self.id_base and self.id_vehicle):
            error_msg = "id_demand, id_base, 或 id_vehicle 未定义，无法计算归一化因子。"
            raise ValueError(error_msg)

        # 尝试计算归一化因子（总旅行时间部分）
        try:
            # 创建副本，将M值替换为-inf（这样max会自动忽略）
            feasible_travel_time = np.where(
                self.travel_time_matrix < self.M, self.travel_time_matrix, -np.inf
            )
            # 理论总旅行时间上界为各个需求点均从基地往返
            # 注意：这里计算时候是每个车都计算了一遍，最后求平均
            _total_travel_time_norm_factor = sum(
                2
                * max(
                    feasible_travel_time[k, i, j]
                    for i in self.id_base
                    for k in self.id_vehicle
                )  # Max travel time from any base to j for vehicle k
                for j in self.id_demand
            )
        except Exception as e:  # 预防出现错误，记录错误日志并抛出异常
            error_msg = f"计算Makespan归一化因子过程中出现错误: {e}. "
            raise ValueError(error_msg)

        # 最小通信时间
        min_comm_time = self.instance.prob_config.instance_param.min_comm_time
        # 计算归一化因子（总通信时间部分）
        _total_comm_time_norm_factor = min_comm_time * len(self.id_demand)

        # 归一化因子，平均旅行时间+平均全覆盖最短通信时间
        total_time_norm_factor = (
            _total_travel_time_norm_factor + _total_comm_time_norm_factor
        ) / len(self.id_vehicle)

        # 检查归一化因子是否过小
        if total_time_norm_factor <= 1e-6:  # 如果过小或为零，则报错
            error_msg = (
                f"Makespan归一化因子过小或为0，当前取值为{total_time_norm_factor}"
            )
            raise ValueError(error_msg)

        # log
        # debug_msg = f"计算的Makespan上界: {total_time_norm_factor}. "
        # print(debug_msg)

        # 如果归一化因子正常，返回归一化后的Makespan目标函数表达式
        return gp.LinExpr(tau / total_time_norm_factor)  # type: ignore[operator]

    # 获取任务优先级目标函数的Gurobi表达式 (归一化)
    def _get_score_objective_expr(self) -> gp.LinExpr:
        """获取任务优先级目标函数的Gurobi表达式 (归一化)

        Returns:
            gp.LinExpr: 归一化后的任务优先级目标函数表达式
        """
        # 检查变量对象是否存在
        if self.variables is None:
            error_msg = "变量未创建"
            raise ValueError(error_msg)

        # 检查变量 delta 是否为 None
        if self.variables.delta is None:
            error_msg = "变量 'delta' 为空，请确保在调用此方法前已创建变量"
            raise ValueError(error_msg)

        delta = self.variables.delta

        # 获取每个需求点的优先级
        node_score_list = self.node_score_list

        # 计算所有需求点的总优先级（用于归一化）
        total_score = sum(node_score_list)

        if total_score <= 0:
            raise ValueError(f"总优先级得分为{total_score}，无法进行归一化处理。")

        # 创建目标函数表达式：每个需求点的优先级与delta变量的乘积之和，除以总优先级
        score_expr = (
            gp.quicksum(node_score_list[i] * delta[i] for i in self.id_demand)
            / total_score
        )

        return gp.LinExpr(score_expr)

    def _set_single_objective(self):
        """将目标函数设置为二者加和"""
        obj_expr = gp.LinExpr()
        for obj_name, obj_sense in self.objective_info:
            if obj_name == "Makespan" and obj_sense == "minimize":
                makespan_expr = self._get_makespan_objective_expr()
                obj_expr += 0.1 * makespan_expr
            elif obj_name == "Priority_Score" and obj_sense == "maximize":
                score_expr = self._get_score_objective_expr()
                obj_expr += 0.9 * (1 - score_expr)
            else:
                error_msg = f"不支持的目标函数: {obj_name}"
                raise NotImplementedError(error_msg)

        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def _set_makespan_objective(self):
        """将目标函数设置为Makespan最小化"""
        makespan_expr = self._get_makespan_objective_expr()
        score_expr = self._get_score_objective_expr()

        obj_expr = makespan_expr + 0.01 * (1 - score_expr)

        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def _add_constraints(self):
        """添加所有约束条件"""
        x = self.variables.x
        z = self.variables.z
        arr_t = self.variables.arr_t
        delta = self.variables.delta
        tau = self.variables.tau
        omega = self.variables.omega

        # -------------------------- 1. 车队规模约束  ------------------------------
        # Constrs 1.1 地面车辆出发数量约束
        self.model.addConstrs(
            (
                gp.quicksum(
                    x[k, i, j]
                    for i in self.id_base
                    for j in self.id_node_no_base
                    if i != j
                )
                <= 1
                for k in self.id_Ground
            ),
            name="Ground_departure_num_Cons",
        )

        # Constrs 1.2 无人机出发数量约束
        self.model.addConstrs(
            (
                gp.quicksum(
                    x[k, i, j]
                    for i in self.id_base
                    for j in self.id_node_no_base
                    if i != j
                )
                <= 1
                for k in self.id_Drone
            ),
            name="Drone_departure_num_Cons",
        )

        # Constrs 1.3 地面车辆返回数量约束
        self.model.addConstr(
            (
                gp.quicksum(
                    x[k, j, i]
                    for k in self.id_Ground
                    for i in self.id_base
                    for j in self.id_node_no_base
                    if i != j
                )
                == gp.quicksum(
                    x[k, i, j]
                    for k in self.id_Ground
                    for i in self.id_base
                    for j in self.id_node_no_base
                    if i != j
                )
            ),
            name="Ground_return_num_Cons",
        )

        # Constrs 1.4 无人机返回数量约束
        self.model.addConstr(
            (
                gp.quicksum(
                    x[k, j, i]
                    for k in self.id_Drone
                    for i in self.id_base
                    for j in self.id_node_no_base
                    if i != j
                )
                == gp.quicksum(
                    x[k, i, j]
                    for k in self.id_Drone
                    for i in self.id_base
                    for j in self.id_node_no_base
                    if i != j
                )
            ),
            name="Drone_return_num_Cons",
        )

        # ---------------------------  2. 流平衡约束  ----------------------------
        # Constrs 2.1 基地流平衡约束
        self.model.addConstrs(
            (
                (
                    gp.quicksum(x[k, i, j] for j in self.id_node_no_base if j != i)
                    - gp.quicksum(x[k, j, i] for j in self.id_node_no_base if j != i)
                    == 0
                )
                for i in self.id_base
                for k in self.id_vehicle
            ),
            name="Base_flow_balance_Cons_i_k",
        )

        # Constrs 2.2 其它点流平衡约束
        self.model.addConstrs(
            (
                (
                    gp.quicksum(x[k, i, j] for j in self.id_all_node if j != i)
                    - gp.quicksum(x[k, j, i] for j in self.id_all_node if j != i)
                    == 0
                )
                for i in self.id_node_no_base
                for k in self.id_vehicle
            ),
            name="Other_flow_balance_Cons_i_k",
        )

        # Constrs 2.3 节点访问变量z(如果z[k,i,i]==1，那么至少有一条弧x[k,i,j]==1)
        self.model.addConstrs(
            (
                (
                    gp.quicksum(
                        x[k, last_node_id, i]
                        for last_node_id in self.id_all_node
                        if last_node_id != i
                    )
                    == z[k, i, j]
                )
                for k in self.id_vehicle
                for i in self.id_demand + self.id_steiner
                for j in self.id_demand + self.id_steiner
                if i == j
            ),
            name="z_ii_visit_define_Cons_k_i_i",
        )
        # debug: 暂时改为，如果没有z[k,i,i]==1, 不允许从该点执行其它点的访问
        # Constrs 2.4 如果没有任何其它节点被覆盖，则Steiner点不能被访问
        self.model.addConstrs(
            (
                gp.quicksum(z[k, i, j] for k in self.id_vehicle for j in self.id_demand)
                <= z[k, i, i] * self.M
                for k in self.id_vehicle
                for i in self.id_steiner
            ),
            name="Steiner_no_coverage_no_visit_Cons",
        )

        # # -------------------------------  3. 任务分配约束  --------------------------------
        # Constrs 3.1 单一任务分配约束
        self.model.addConstrs(
            (
                gp.quicksum(
                    z[k, i, j] for k in self.id_vehicle for j in self.id_node_no_base
                )
                == delta[i]
                for i in self.id_demand
            ),
            name="Priority_task_assign_Cons_i",
        )

        # Constrs 3.2 可达任务分配约束
        self.model.addConstrs(
            (
                gp.quicksum(
                    z[k, i, j]
                    for k in self.id_Ground
                    for j in self.id_connected + self.id_steiner
                )
                + gp.quicksum(
                    z[k, i, j] for k in self.id_Drone for j in self.id_node_no_base
                )
                == delta[i]
                for i in self.id_connected
            ),
            name="Priority_connected_task_Cons_i",
        )

        # Constrs 3.3 不可达受灾点任务分配约束
        self.model.addConstrs(
            (
                gp.quicksum(
                    z[k, i, j]
                    for k in self.id_Ground
                    for j in self.id_connected + self.id_steiner
                    if i != j
                )
                + gp.quicksum(
                    z[k, i, j] for k in self.id_Drone for j in self.id_node_no_base
                )
                == delta[i]
                for i in self.id_isolated
            ),
            name="Priority_isolated_task_Cons_i",
        )

        # -------------------------------  4. 访问逻辑约束  --------------------------------
        # Constrs 4.1 信号覆盖枢纽约束
        self.model.addConstrs(
            (
                (z[k, i, j] <= z[k, j, j])
                for k in self.id_vehicle
                for i in self.id_demand
                for j in self.id_node_no_base
                if i != j
            ),
            name="Hub_logic_Cons_k_i_j",
        )
        # Constrs 4.2 Steiner点约束
        self.model.addConstrs(
            (
                (gp.quicksum(z[k, i, i] for k in self.id_vehicle) <= 1)
                for i in self.id_steiner
            ),
            name="Steiner_logic_Cons_i",
        )

        # ------------------------------  5. 访问时间约束  ---------------------------------
        # Constrs 5.1 出发弧访问时间约束
        self.model.addConstrs(
            (
                (
                    arr_t[k, j]
                    >= self.travel_time_matrix[k, i, j] * x[k, i, j]
                    - self.M * (1 - x[k, i, j])
                )
                for k in self.id_vehicle
                for i in self.id_base
                for j in self.id_node_no_base
            ),
            name="Departure_travel_time_Cons_k_i_j",
        )

        # Constrs 5.1.2 出发弧访问时间约束 - 附加约束
        self.model.addConstrs(
            (
                (
                    arr_t[k, j]
                    <= self.travel_time_matrix[k, i, j] * x[k, i, j]
                    + self.M * (1 - x[k, i, j])
                )
                for k in self.id_vehicle
                for i in self.id_base
                for j in self.id_node_no_base
            ),
            name="Departure_travel_time_Cons_k_i_j_additional",
        )

        # Constrs 5.2 节点间访问时间约束
        self.model.addConstrs(
            (
                (
                    arr_t[k, j]
                    >= arr_t[k, i]
                    + self.travel_time_matrix[k, i, j] * x[k, i, j]
                    + self.min_comm_time * z[k, i, i]
                    - self.M * (1 - x[k, i, j])
                )
                for k in self.id_vehicle
                for i in self.id_node_no_base
                for j in self.id_all_node
                if i != j
            ),
            name="Travel_time_Cons_k_i_j",
        )

        # Constrs 5.2 节点间访问时间约束 - 附加约束
        self.model.addConstrs(
            (
                (
                    arr_t[k, j]
                    <= arr_t[k, i]
                    + self.travel_time_matrix[k, i, j] * x[k, i, j]
                    + self.min_comm_time * z[k, i, i]
                    + self.M * (1 - x[k, i, j])
                )
                for k in self.id_vehicle
                for i in self.id_node_no_base
                for j in self.id_all_node
                if i != j
            ),
            name="Travel_time_Cons_k_i_j_additional",
        )

        # Constrs 5.3 定义tau变量
        self.model.addConstrs(
            (tau >= arr_t[k, i] for k in self.id_vehicle for i in self.id_base),
            name="Tau_def_Cons_k_i",
        )

        # -----------------------------  6. 信号覆盖半径约束  -------------------------------
        # Constrs 6.1 信号覆盖约束
        self.model.addConstrs(
            (
                (
                    self.distance_matrix[i, j] * z[k, i, j]
                    <= self.instance.comm_radius_array[k]
                )
                for k in self.id_vehicle
                for i in self.id_demand
                for j in self.id_node_no_base
                if i != j
            ),
            name="Signal_cover_Cons_k_i_j",
        )

        # ----------------------------  7. 应急通信平台续航约束  ----------------------------
        # Constrs 7.1 任意节点剩余续航都应该大于等于0，小于等于车辆能量容量
        self.model.addConstrs(
            ((omega[k, i] >= 0) for k in self.id_vehicle for i in self.id_all_node),
            name="Energy_lower_bound_Cons_k_i",
        )

        # Constrs 7.1.2 任意节点剩余续航都应该小于等于车辆能量容量
        self.model.addConstrs(
            (
                omega[k, i] <= self.instance.capacity_array[k]
                for k in self.id_vehicle
                for i in self.id_all_node
            ),
            name="Energy_upper_bound_Cons_k_i",
        )

        # Constrs 7.2 节点间能量消耗关系
        self.model.addConstrs(
            (
                (
                    omega[k, j]
                    <= omega[k, i]
                    - self.travel_cost_matrix[k, i, j]
                    - self.comm_cost_matrix[k, i] * z[k, i, i]
                    + self.M * (1 - x[k, i, j])
                )
                for k in self.id_vehicle
                for i in self.id_node_no_base
                for j in self.id_all_node
            ),
            name="Energy_consume_Upper_bound_Cons_k_i_j",
        )

        # Constrs 7.2.2 节点间能量消耗关系附加约束
        self.model.addConstrs(
            (
                (
                    omega[k, j]
                    >= omega[k, i]
                    - self.travel_cost_matrix[k, i, j]
                    - self.comm_cost_matrix[k, i] * z[k, i, i]
                    - self.M * (1 - x[k, i, j])
                )
                for k in self.id_vehicle
                for i in self.id_node_no_base
                for j in self.id_all_node
            ),
            name="Energy_consume_Lower_bound_Cons_k_i_j",
        )

        # Constrs 7.3 出发弧能量消耗定义
        self.model.addConstrs(
            (
                (
                    omega[k, j]
                    <= self.instance.capacity_array[k]
                    - self.travel_cost_matrix[k, i, j]
                    + (1 - x[k, i, j]) * self.M
                )
                for k in self.id_vehicle
                for i in self.id_base
                for j in self.id_node_no_base
            ),
            name="Departure_energy_consume_Upper_bound_Cons_k_i_j",
        )
        # Constrs 7.3.2 出发弧能量消耗定义附加约束
        self.model.addConstrs(
            (
                (
                    omega[k, j]
                    >= self.instance.capacity_array[k]
                    - self.travel_cost_matrix[k, i, j]
                    - (1 - x[k, i, j]) * self.M
                )
                for k in self.id_vehicle
                for i in self.id_base
                for j in self.id_node_no_base
            ),
            name="Departure_energy_consume_Lower_bound_Cons_k_i_j",
        )
        # Constrs 7.4 非访问节点能量定义为0
        self.model.addConstrs(
            (
                (omega[k, i] <= self.M * z[k, i, i])
                for k in self.id_vehicle
                for i in self.id_node_no_base
            ),
            name="Non_visit_energy_consume_Cons_k_i",
        )

    def _setup_basic_model(self):
        """设置基础模型，包括创建模型、变量和目标函数"""

        # todo： 初始化尚未检查
        # 创建Gurobi模型
        self._create_model()

        # 创建Gurobi变量
        self._create_variables()

        # 添加约束
        self._add_constraints()

        # 更新模型，应对Gurobi惰性更新方法
        self.model.update()

    def _extract_solution(self):
        """从Gurobi模型中提取解并存入Solution对象中"""
        # 定义求解状态映射
        status_map = {
            GRB.OPTIMAL: "最优解",
            GRB.INFEASIBLE: "无可行解",
            GRB.UNBOUNDED: "无界解",
            GRB.TIME_LIMIT: "时间限制，提前结束",
            GRB.INTERRUPTED: "中断",
            GRB.USER_OBJ_LIMIT: "用户目标限制",
        }
        # 获取求解状态
        try:
            self.solution.status = status_map.get(
                self.model.Status, f"未知状态 ({self.model.Status})"
            )
            self.solution.solve_time = self.model.Runtime
            self.solution.mip_gap = self.model.MIPGap
        except AttributeError as e:
            self.solution.status = "模型未运行或状态未知"
            raise e

        # 如果没有找到可行解,直接返回
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.USER_OBJ_LIMIT]:
            raise ValueError("模型无解，无法提取解信息。")

        # 检查是否有解
        sol_count = self.model.SolCount
        if sol_count == 0:
            print(f"当前可行解数量为{sol_count}, 返回空solution")
            return self.solution

        # 提取变量
        x = self.variables.x
        z = self.variables.z

        # 1. 构建路径 (routes)
        for k in self.id_vehicle:
            route = []
            # 找到起始节点(从基地出发)
            current_node = None
            for i in self.id_base:
                for j in self.id_node_no_base:
                    if x[k, i, j].X > 0.5:  # 使用0.5作为阈值判断二值变量
                        current_node = j
                        route.append(i)  # 添加基地作为起点
                        route.append(j)
                        break
                if current_node is not None:
                    break

            # 如果该车辆没有出发,跳过
            if current_node is None:
                continue

            # 按照路径顺序遍历
            visited = set(route)
            while current_node not in self.id_base:
                found_next = False
                for j in self.id_all_node:
                    if j not in visited and x[k, current_node, j].X > 0.5:
                        route.append(j)
                        visited.add(j)
                        current_node = j
                        found_next = True
                        break

                # 如果是返回基地
                if not found_next:
                    for j in self.id_base:
                        if x[k, current_node, j].X > 0.5:
                            route.append(j)
                            current_node = j
                            break
                    break

            self.solution.routes[k] = route

        # 2. 构建覆盖关系 (coverage)
        for k in self.id_vehicle:
            for j in self.id_node_no_base:
                # 检查车辆k是否访问了节点j
                if z[k, j, j].X > 0.5:
                    covered_nodes = set()
                    # 找到所有由节点j覆盖的需求节点
                    for i in self.id_demand:
                        if z[k, i, j].X > 0.5:
                            covered_nodes.add(i)

                    # 如果有覆盖的节点,设置覆盖关系
                    if covered_nodes:
                        self.solution.set_coverage(k, j, covered_nodes)

        # 3. 提取目标函数值
        try:
            # 获取归一化后的目标函数表达式
            makespan_expr = self._get_makespan_objective_expr()
            score_expr = self._get_score_objective_expr()

            # 计算表达式的值（这些都是归一化后的值，范围在0-1之间）
            makespan_value = makespan_expr.getValue()
            priority_score = score_expr.getValue()

            # 存储目标值（归一化后的值）
            self.solution.objectives = [makespan_value, priority_score]

        except (AttributeError, gp.GurobiError) as e:
            # 如果无法获取目标值,保持默认值
            raise ValueError(f"警告: 无法提取目标函数值: {e}")
        return self.solution

    def _set_model_params(self):
        """设置模型求解参数，主要包括
        1. 设置最大求解时间
        2. 关闭Gurobi原生输出
        """
        algorithm_config = self.instance.prob_config.algorithm_config

        time_limit = algorithm_config.time_limit

        # 关闭gurobi输出
        self.model.setParam("OutputFlag", 0)

        self.model.setParam("TimeLimit", time_limit)

        # # 设置数值容差参数
        self.model.setParam("FeasibilityTol", 1e-6)  # 可行性容差
        self.model.setParam("OptimalityTol", 1e-6)  # 最优性容差
        self.model.setParam("IntFeasTol", 1e-5)  # 整数可行性容差

        # 设置整数优先级
        self.model.setParam(GRB.Param.IntegralityFocus, 1)

    def mip_solve(self) -> Solution:
        """求解方法"""

        # 设置基础模型（初始化模型、添加变量、添加约束）
        self._setup_basic_model()

        # 设置目标函数(直接将二者加和)
        self._set_single_objective()

        # 设置求解器参数
        self._set_model_params()

        # 优化模型
        self.model.optimize()

        # 提取解信息
        self._extract_solution()

        return self.solution


class GurobiEpsilonConstraintSolver(GurobiMIPSolver):
    """Gurobi ε-约束法求解器基类"""

    def __init__(
        self,
        population: Population,
        instance: InstanceClass,
    ):
        # 存入种群对象、算例对象
        self.population = population
        self.instance = instance

        # 提取必要的集合和参数
        # ---------- 节点集合 ----------
        self.id_base = list(instance.base_ids)
        self.id_demand = list(instance.demand_ids)
        accessible_flag = instance.accessible
        self.id_isolated = [i for i in self.id_demand if not accessible_flag[i]]
        self.id_connected = [i for i in self.id_demand if accessible_flag[i]]
        self.id_steiner = list(instance.steiner_ids)
        self.id_all_node = list(range(instance.total_node_num))
        self.id_node_no_base = self.id_demand + self.id_steiner

        # ---------- 车辆集合 ----------
        self.id_Ground = instance.ground_veh_ids
        self.id_Drone = instance.drone_ids
        self.id_vehicle = list(range(instance.total_veh_num))

        # ---------- 矩阵提取 ----------
        self.distance_matrix = instance.distance_matrix
        self.coverage_matrix = instance.comm_coverage_matrix
        self.travel_time_matrix = instance.travel_time_matrix
        self.travel_cost_matrix = instance.travel_consum_matrix
        self.comm_cost_matrix = instance.comm_consum_matrix

        # --------- 参数提取（优先级得分、最小通信时间）----------
        self.node_score_list = instance.priority
        self.min_comm_time = instance.prob_config.instance_param.min_comm_time

        # --------- 求解器参数 ----------
        self.M = instance.prob_config.algorithm_config.big_m
        self.time_limit = instance.prob_config.algorithm_config.time_limit

        # 设置目标函数信息
        obj_config = instance.prob_config.algorithm_config.obj_config
        obj_names = obj_config.name
        obj_sense = obj_config.sense
        self.objective_info = list(zip(obj_names, obj_sense))

    def _create_model(self, model_name: str = "EpsilonCons_Optimizer"):
        """创建Gurobi模型"""
        self.model = gp.Model(model_name)

    def _add_epsilon_constraint(self, epsilon: float):
        priority_score_expr = self._get_score_objective_expr()
        # 添加ε-约束
        self.model.addConstr(
            (priority_score_expr >= epsilon),
            name=f"Epsilon_constraint_{epsilon}",
        )

    def epsilon_cons_solve(self) -> Population:
        """求解方法"""
        # [X] 设定优先级得分阈值范围
        Priority_Score_Max = 1.0
        Priority_Score_Min = 0.0
        num_epsilon_point = (
            self.instance.prob_config.algorithm_config.num_epsilon_points
        )

        epsilon_values = np.round(
            np.linspace(Priority_Score_Min, Priority_Score_Max, num=num_epsilon_point),
            2,
        )

        # # debug: 暂时强制设置epsilon方便找到模型问题
        # epsilon_values = [0.11]

        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"ε-约束法求解器: 设定的优先级得分阈值为 {epsilon_values}"
        )

        for i, epsilon in enumerate(epsilon_values):
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:"
                f" 求解第 {i+1} 个 epsilon = {epsilon} 的单目标MIP模型..."
            )
            self.solution = Solution.new_solution(
                self.instance, solver_name=f"epsilon_cons_{epsilon}"
            )

            # 构建模型
            self._setup_basic_model()

            self._set_makespan_objective()

            # 添加当前ε-约束
            self._add_epsilon_constraint(epsilon)

            # 设置求解器参数
            self._set_model_params()

            # 执行优化
            self.model.optimize()

            # 已兼容无可行解的情况
            # 提取解具体信息，包括运行状态、求解时间、目标函数值、解详情（如有可行解）
            current_solution = self._extract_solution()

            # 格式化目标值
            print_makespan = (
                f"{current_solution.objectives[0]:.4f}"
                if current_solution.objectives
                else "N/A"
            )
            print_priority_score = (
                f"{current_solution.objectives[1]:.4f}"
                if current_solution.objectives
                else "N/A"
            )
            # 打印当前优化结果
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:"
                f" 第 {i+1} 个epsilon = {epsilon} 求解完成，"
                f"状态 = ({current_solution.status}), "
                f"求解时间 = {current_solution.solve_time:.2f} 秒, "
                f"[ Makespan = {print_makespan}, "
                f"Priority_Score = {print_priority_score} ]"
            )
            # 将当前解存入种群
            self.population.add_solution(current_solution)

            # # debug: 存储当前的数学模型和非零变量
            # save_folder = f"./gurobi_models/epsilon_{epsilon}"
            # save_gurobi_model(
            #     self.model,
            #     folder_path=save_folder,
            #     base_name=f"epsilon_{epsilon}_model",
            # )

            # 清理当前迭代的模型
            if self.model:
                self.model.dispose()

            if self.solution:
                del self.solution

        #
        return self.population


def gurobi_single_obj_solver(solution: Solution, instance: InstanceClass) -> Solution:
    """使用Gurobi求解单目标ECCP问题"""
    # 初始化mip求解器
    mip_solver = GurobiMIPSolver(solution, instance)
    # 执行求解
    solution = mip_solver.mip_solve()

    return solution


def gurobi_epsilon_constraint_solver(
    population: Population, instance: InstanceClass
) -> Population:
    """使用Gurobi求解多目标ECCP问题，采用ε-约束法"""
    epsilon_cons_solver = GurobiEpsilonConstraintSolver(population, instance)
    population = epsilon_cons_solver.epsilon_cons_solve()
    return population


def save_gurobi_model(model, folder_path, base_name="model"):
    """
    保存Gurobi模型和非零变量到指定文件夹

    参数:
        model: Gurobi模型对象
        folder_path: 目标文件夹路径
        base_name: 文件基础名称，默认为"model"
    """
    import os

    # 创建文件夹（如果不存在）
    os.makedirs(folder_path, exist_ok=True)

    # 1. 保存模型文件（多种格式）
    # LP格式 - 人类可读
    model.write(os.path.join(folder_path, f"{base_name}.lp"))

    # MPS格式 - 标准格式
    model.write(os.path.join(folder_path, f"{base_name}.mps"))

    # 2. 保存求解结果（如果已求解）
    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        # 保存解（所有变量值）
        model.write(os.path.join(folder_path, f"{base_name}.sol"))

        # 3. 单独保存非零变量
        with open(
            os.path.join(folder_path, f"{base_name}_nonzero_vars.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"模型名称: {model.ModelName}\n")
            f.write(f"目标值: {model.ObjVal}\n")
            f.write("=" * 60 + "\n")
            f.write("非零变量:\n")
            f.write("=" * 60 + "\n\n")

            # 获取所有变量
            vars_list = model.getVars()

            # 筛选非零变量
            nonzero_count = 0
            for var in vars_list:
                if abs(var.X) > 1e-6:  # 容差阈值
                    f.write(f"{var.VarName:30s} = {var.X:15.6f}\n")
                    nonzero_count += 1

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"非零变量总数: {nonzero_count} / {len(vars_list)}\n")

        print(f"✓ 模型和结果已保存到: {folder_path}")
        print(f"  - {base_name}.lp (模型LP格式)")
        print(f"  - {base_name}.mps (模型MPS格式)")
        print(f"  - {base_name}.sol (完整解)")
        print(f"  - {base_name}_nonzero_vars.txt (非零变量)")
    else:
        print("⚠ 模型未求解或无可行解，仅保存模型文件")
        print(f"  - {base_name}.lp")
        print(f"  - {base_name}.mps")

    return folder_path
