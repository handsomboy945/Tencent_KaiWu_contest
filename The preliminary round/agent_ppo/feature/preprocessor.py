#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_ppo.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process
import json

#将v限制在给定的范围内，然后将其进行归一化
def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16
        self.reset()
    # 重置函数
    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = [[0,0], [0,0], [0,0], [0,0], [0,0],
                            [0,0], [0,0], [0,0], [0,0], [0,0],
                           ]
        self.bad_move_ids = set()
        self.is_flashed = True
        self.treasure_pos = [[0,0], [0,0], [0,0], [0,0], [0,0],
                            [0,0], [0,0], [0,0], [0,0], [0,0],
                            [0,0], [0,0], [0,0]]
        self.buff_pos = [0,0]
        self.treasure_found = np.zeros((13,), dtype=np.float32)
        self.hero_status = np.zeros((15,), dtype=np.float32) # 英雄状态，13个宝箱，1buff，1英雄加速状态 1表示可拾取，0表示不可拾取
        self.global_memory_map = np.zeros((128,128), dtype=np.float32)
        self.local_memory_map = np.zeros((11,11), dtype=np.float32)
        self._obs = None
        self._extra_info = None
        self.buff_collect_step = 0
        # 初始化标志
        self.init_flag = False
        self.buff_pos_record = [0,0]

    #返回特征包括：归一化后的与目标相差距离的坐标x与y，目标坐标，归一化后的距离
    def _get_pos_feature(self, found, cur_pos, target_pos):
        if target_pos == [0,0]:
            return [0,0,0,0,0,0]
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128), #这里的1.41是根号2，这里就是利用其最大值空间中最大距离进行归一化
            ),
        )
        return feature

    # 记录每个位置智能体走过的权重，每走过1次权重加0.1
    def memory_update(self, cur_pos):
        # 全局记忆矩阵
        x,z = cur_pos
        current_value = self.global_memory_map[x, z]
        self.global_memory_map[z, x] = min(1.0, current_value + 0.1)

        # 局部记忆矩阵
        # 计算在全局地图上的原区域边界
        src_top = max(0, z - 5)
        src_bottom = min(128, z + 6)
        src_left = max(0, x - 5)
        src_right = min(128, x + 6)

        # 计算在局部地图上的目标区域边界
        dst_top = src_top - (z - 5)
        dst_bottom = src_bottom - (z - 5)
        dst_left = src_left - (x - 5)
        dst_right = src_right - (x - 5)

        # 从全局地图复制有效区域到局部地图
        self.local_memory_map[dst_top:dst_bottom, dst_left:dst_right] = self.global_memory_map[src_top:src_bottom, src_left:src_right]
        self.memory_flag = self.local_memory_map


    #处理游戏状态的函数
    def pb2struct(self, frame_state, last_action):
        with open('frame_state.json', mode='w') as f:
            json.dump(frame_state, f, indent=4)

        obs, extra_info = frame_state
        if self._obs is None: 
            self._obs = obs
        if self._extra_info is None:
            self._extra_info = extra_info
        self.obs = obs
        self.extra_info = extra_info
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        #闪现是否可用 -- 新增一个字段来代表闪现是否可用，默认初始化的时候可以为True
        if hero['talent']['status'] == 0:
            self.is_flashed = False
        elif hero['talent']['status'] == 1:
            self.is_flashed = True
        
        #初始化宝箱状态1表示可以被拾取，0表示不可以被拾取
        if self.init_flag == False:
            self.init_flag = True
            for organ in obs["frame_state"]["organs"]:
                if organ["sub_type"] == 1:
                    self.hero_status[organ['config_id']-1] = 1
                if organ["sub_type"] == 2:
                    self.hero_status[13] = 1
        self.hero_status[14] = hero["speed_up"]

        # 终点位置
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )

                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis
            elif organ["sub_type"] == 1:
                if organ['status'] == 1:
                    self.treasure_found[organ['config_id']-1] = 1
                    self.treasure_pos[organ['config_id']-1] = [organ['pos']['x'], organ['pos']['z']]
                elif organ['status'] == 0:
                    self.treasure_found[organ['config_id']-1] = 0
                    self.hero_status[organ['config_id']-1] = 0
                    self.treasure_pos[organ['config_id']-1] = [0,0]
            elif organ["sub_type"] == 2:
                if organ["status"] == 1:
                    self.buff_pos = [organ['pos']['x'], organ['pos']['z']]
                    self.buff_pos_record = self.buff_pos
                elif organ["status"] == 0:
                    self.hero_status[13] = 0
                    self.buff_pos = [0,0]
        self.buff_collect_step = self.step_no if self._obs["score_info"]['buff_count'] < self.obs["score_info"]['buff_count'] else self.buff_collect_step
        if self.step_no-self.buff_collect_step >= 100:
            self.buff_pos = self.buff_pos_record
            self.hero_status[13] = 1

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)
        if len(self.history_pos) != 0:
            self.feature_lastend_pos = self._get_pos_feature(self.is_end_pos_found, self.history_pos[-1], self.end_pos)
        else:
            self.feature_lastend_pos = np.zeros((6,),dtype=np.float32)

        # History position
        # 历史位置，限制记录的位置为10，将最旧的位置移除
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # History position feature
        # 历史位置特征，这里是从当前位置到达历史记录的位置的最初的位置位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True
        self.last_action = last_action

        map_info = obs['map_info']
        # 新增的地图特征处理
        self.treasure_map = np.zeros((11, 11), dtype=np.float32)
        self.end_map = np.zeros((11, 11), dtype=np.float32)
        self.obstacle_map = np.zeros((11, 11), dtype=np.float32)
        self.buff_map = np.zeros((11, 11), dtype=np.float32)
        
        for r, row_data in enumerate(map_info):
            for c, value in enumerate(row_data['values']):
                # 宝箱
                if value == 4:
                    self.treasure_map[r, c] = 1.0
                # 终点
                elif value == 3:
                    self.end_map[r, c] = 1.0
                # 障碍物
                elif value == 0:
                    self.obstacle_map[r, c] = 1.0
                # 加速增益
                elif value == 6:
                    self.buff_map[r, c] = 1.0 

        #进行记忆矩阵的更新
        self.memory_update(self.cur_pos)

    def process(self, frame_state, last_action, is_exploit=False):
        self.pb2struct(frame_state, last_action)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        self.feature_treasure_pos = np.array([self._get_pos_feature(self.treasure_found[i], self.cur_pos, treasure) for i,treasure in enumerate(self.treasure_pos)]).flatten()
        self.feature_buff_pos = np.array(self._get_pos_feature(1, self.cur_pos, self.buff_pos))
        feature = np.concatenate([
            self.treasure_map.flatten(),    # 宝箱信息
            self.end_map.flatten(),         # 终点信息
            self.obstacle_map.flatten(),    # 障碍信息
            self.buff_map.flatten(),        # 增益信息
            self.memory_flag.flatten(),      # 记忆矩阵
            self.cur_pos_norm, 
            self.feature_end_pos, 
            self.feature_history_pos, 
            self.feature_treasure_pos,  # 宝箱特征位置信息
            self.feature_buff_pos, # buff位置特征信息
            self.hero_status,
            legal_action
            ])

        if is_exploit:
            reward = [0]
        else:
            reward = reward_process(self)
        self._obs = self.obs
        self._extra_info = self.extra_info
        if is_exploit:
            return (feature, legal_action)
        else:
            return (feature, legal_action, reward)

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        legal_action = [self.move_usable] * self.move_action_num
        # 添加闪现的合法性
        if self.is_flashed:
            legal_action[8:] = [True] * 8
        else:
            legal_action[8:] = [False] * 8


        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            legal_action[:8] = [self.move_usable] * 8

        return legal_action
