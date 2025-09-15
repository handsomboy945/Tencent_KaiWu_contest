#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.conf.conf import Config
from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np

# The create_cls function is used to dynamically create a class.
# The first parameter of the function is the type name, and the remaining parameters are the attributes of the class.
# The default value of the attribute should be set to None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
    reward=None,
)


ActData = create_cls(
    "ActData",
    probs=None,
    value=None,
    target=None,
    predict=None,
    action=None,
    prob=None,
)

SampleData = create_cls("SampleData", npdata=None)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}


RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


def reward_process(preprocessor):
    reward = {}
    # 从 preprocessor 提取所需数据
    _obs = preprocessor._obs or {}  # 当前帧观测
    obs = preprocessor.obs or {}  # 上一帧观测
    _extra_info = preprocessor._extra_info or {}
    extra_info = preprocessor.extra_info or {}
    target = preprocessor.feature_end_pos[-1]  # 当前目标距离
    last_target = preprocessor.feature_lastend_pos[-1]  # 上一帧目标距离
    memory_map = preprocessor.local_memory_map  # 局部记忆矩阵
    _hero = _obs['frame_state']['heroes'][0]
    hero = obs['frame_state']['heroes'][0]
    _score_info = _obs['score_info']
    score_info = obs['score_info']

    # 步数惩罚
    reward['step_reward'] = -0.01

    # 终点奖励
    reward['end_dist_reward'] = (last_target - target) * 0.1 * (1.41 * 128)# 反归一化，乘以对角线长度

    # 徘徊惩罚
    reward['memory_reward'] = -memory_map[5][5] if memory_map[5][5] > 0.2 else 0

    # 停止惩罚
    cur_pos = (_hero["pos"]["x"], _hero["pos"]["z"]) if _hero else None
    last_pos = (hero["pos"]["x"], hero["pos"]["z"]) if hero else None
    reward['stop_reward'] = -0.5 if (cur_pos and last_pos and cur_pos == last_pos) else 0

    # 加速奖励
    _buff_cnt = _score_info['buff_count']
    buff_cnt = score_info['buff_count'] 
    reward['buff_reward'] = (0.5)**buff_cnt if _buff_cnt < buff_cnt else 0 

    # 技能惩罚
    _talent_cnt = _score_info['talent_count']
    talent_cnt = score_info['talent_count']
    reward['talent_reward'] = -0.5 if (_talent_cnt < talent_cnt) else 0

    #宝箱奖励
    _treasure_cnt = _score_info['treasure_collected_count']
    treasure_cnt = score_info['treasure_collected_count']
    reward['treasure_reward'] = 5*(treasure_cnt - _treasure_cnt) if (treasure_cnt - _treasure_cnt) > 0 else 0

    #目标奖励
    total_treasure_cnt = extra_info['game_info']['treasure_count'] - 2 #减去2来处理计入终点和buff的bug
    end_pos = extra_info['game_info']['end_pos']
    cur_pos = preprocessor.cur_pos
    end_reward = 10 if (total_treasure_cnt == treasure_cnt) else -5 * (total_treasure_cnt - treasure_cnt)
    reward['end_reward'] = end_reward if (end_pos and cur_pos and cur_pos == end_pos) else 0
    return [sum(reward.values())]


class SampleManager:
    def __init__(
        self,
        gamma=0.99,
        tdlambda=0.95,
    ):
        self.gamma = Config.GAMMA
        self.tdlambda = Config.TDLAMBDA

        self.feature = []
        self.probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []

    def add(self, feature, legal_action, prob, action, value, reward):
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.probs.append(prob)
        self.actions.append(action)
        self.value.append(value)
        self.reward.append(reward)
        self.adv.append(np.zeros_like(value))
        self.tdlamret.append(np.zeros_like(value))
        self.count += 1

    def add_last_reward(self, reward):
        self.reward.append(reward)
        self.value.append(np.zeros_like(reward))

    def update_sample_info(self):
        last_gae = 0
        for i in range(self.count - 1, -1, -1):
            reward = self.reward[i + 1]
            next_val = self.value[i + 1]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.tdlambda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val

    def sample_process(self, feature, legal_action, prob, action, value, reward):
        self.add(feature, legal_action, prob, action, value, reward)

    def process_last_frame(self, reward):
        self.add_last_reward(reward)
        # 发送前的后向传递更新
        # Backward pass updates before sending
        self.update_sample_info()
        self.samples = self._get_game_data()

    def get_game_data(self):
        ret = self.samples
        self.samples = []
        return ret

    def _get_game_data(self):
        feature = np.array(self.feature).transpose()
        probs = np.array(self.probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward[:-1]).transpose()
        value = np.array(self.value[:-1]).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()

        data = np.concatenate([feature, reward, value, tdlamret, adv, actions, probs, legal_action]).transpose()

        samples = []
        for i in range(0, self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))

        return samples


@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
