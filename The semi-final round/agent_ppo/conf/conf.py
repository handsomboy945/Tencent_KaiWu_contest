#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

HERO_FEATURE_DIM = 199
ENEMY_FEATURE = 194
FRI_SOILDER_FEATURE = 36
ENE_SOILDER_FEATURE = 36
ORGAN_FEATURE_DIM = 26
GAMESTATE_FEATURE_DIM = 13
MONSTER_FEATURE_DIM = 4
UNIT_FEATURE_DIM = HERO_FEATURE_DIM + ENEMY_FEATURE + FRI_SOILDER_FEATURE + ENE_SOILDER_FEATURE + ORGAN_FEATURE_DIM + MONSTER_FEATURE_DIM
DIM_OF_FEATURE = UNIT_FEATURE_DIM + GAMESTATE_FEATURE_DIM

class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 6.5,
        "money": 0.008,
        "exp": 0.008,
        "ep_rate": 0.8,
        "death": -1.0,
        "kill": -0.5,
        "last_hit": 0.5,
        "forward": 0,
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 20000
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    HERO_FEATURE_DIM = 199
    ENEMY_FEATURE = 194
    FRI_SOILDER_FEATURE = 36
    ENE_SOILDER_FEATURE = 36
    ORGAN_FEATURE_DIM = 26
    GAMESTATE_FEATURE_DIM = 13
    MONSTER_FEATURE_DIM = 4
    UNIT_FEATURE_DIM = HERO_FEATURE_DIM + ENEMY_FEATURE + FRI_SOILDER_FEATURE + ENE_SOILDER_FEATURE + ORGAN_FEATURE_DIM + MONSTER_FEATURE_DIM
    DIM_OF_FEATURE = UNIT_FEATURE_DIM + GAMESTATE_FEATURE_DIM


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    VALUE_DIM = 5
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        DIM_OF_FEATURE + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(DIM_OF_FEATURE,), (85,)] #这里后边的85维是legal_action前边的为处理的些许特征
    FEATURE_SPLIT = [HERO_FEATURE_DIM, ENEMY_FEATURE, FRI_SOILDER_FEATURE, ENE_SOILDER_FEATURE, ORGAN_FEATURE_DIM, MONSTER_FEATURE_DIM, GAMESTATE_FEATURE_DIM]
    INIT_LEARNING_RATE_START = 1e-4
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(DIM_OF_FEATURE + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.996
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # dual_-clip PPO参数
    DUAL_CLIP_C = 3.0

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
