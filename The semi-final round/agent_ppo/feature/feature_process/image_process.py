#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import configparser
import os
import math
from collections import OrderedDict


class ImageProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_image_config()
        
        # 初始化实体类型编码
        self.entity_codes = {
            "empty": 0,
            "friendly_minion": 1,
            "enemy_minion": 2,
            "friendly_tower": 3,
            "enemy_tower": 4,
            "enemy_hero": 5,
        }
        
        # 获取图像参数
        self.image_width = 35
        self.image_height = 35
        self.cell_size = 400
        self.view_dist = 7000
        
        # 存储游戏实体信息
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}
        self.minions = []  # 小兵和野怪

    def get_image_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "image_feature_config.ini")
        self.config.read(config_path)
        
        # 获取特征函数配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_image(self, frame_state):
        """生成单通道图像特征"""
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)
        
        # 初始化图像
        image = np.full((self.image_height, self.image_width), 
                       self.entity_codes["empty"], dtype=np.float32)
        # 获取主英雄位置
        main_hero = self.main_hero_info
        self.center_x = main_hero["actor_state"]["location"]["x"]
        self.center_z = main_hero["actor_state"]["location"]["z"]
           
        # 如果视野非空就print
        self._add_entities_to_image(image)
        # if image.sum():
        #     print(image)
        
        # 将图像展平为一维向量以便与其他特征拼接
        flattened_image = image.flatten().tolist()

        # 设置numpy打印选项
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        
        return flattened_image

    def _add_entities_to_image(self, image):
        """添加各种实体到图像中"""
        # 先添加小兵和野怪
        for minion in self.minions:
            camp_code = self.entity_codes["friendly_minion"] if minion["camp"] == self.main_camp else self.entity_codes["enemy_minion"]
            self._add_entity_to_image(image, 
                                    minion["location"]["x"], minion["location"]["z"], 
                                    camp_code)
        
        # 再添加英雄
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            self._add_entity_to_image(image, 
                                    hero["actor_state"]["location"]["x"], 
                                    hero["actor_state"]["location"]["z"], 
                                    self.entity_codes["enemy_hero"])

        # 最后添加防御塔
        for organ_id, organ in self.main_camp_organ_dict.items():
            if organ["sub_type"] == "ACTOR_SUB_TOWER":
                self._add_entity_to_image(image,
                                        organ["location"]["x"], organ["location"]["z"], 
                                        self.entity_codes["friendly_tower"])
        
        for organ_id, organ in self.enemy_camp_organ_dict.items():
            if organ["sub_type"] == "ACTOR_SUB_TOWER":
                self._add_entity_to_image(image, 
                                        organ["location"]["x"], organ["location"]["z"], 
                                        self.entity_codes["enemy_tower"])

    def _add_entity_to_image(self, image, entity_x, entity_z, entity_code):
        """将单个实体添加到图像中"""
        # 计算实体在图像中的位置
        i = int((entity_z - self.center_z) // self.cell_size) + 17
        j = int((entity_x - self.center_x) // self.cell_size) + 17
        
        # 确保位置在图像范围内
        if 0 <= i < self.image_height and 0 <= j < self.image_width:
            image[i, j] = entity_code

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["actor_state"]["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["actor_state"]["config_id"]] = hero

    def generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()
        self.minions = []

        for organ in frame_state["npc_states"]:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_subtype == "ACTOR_SUB_TOWER":
                if organ_camp == self.main_camp:
                    self.main_camp_organ_dict[organ["runtime_id"]] = organ
                else:
                    self.enemy_camp_organ_dict[organ["runtime_id"]] = organ
            elif organ_subtype == "ACTOR_SUB_SOLDIER":
                # print(organ_subtype, organ["actor_type"], organ_camp)
                self.minions.append(organ)

    def generate_image_feature(self, frame_state, vector_feature):
        """生成图像特征的主函数"""
        image_feature = self.process_image_feature(frame_state)
        vector_feature.extend(image_feature)