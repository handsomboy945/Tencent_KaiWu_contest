#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from enum import Enum
from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math
from collections import OrderedDict


class OrganProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp

        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}
        self.main_camp_cake_dict = {}
        self.enemy_camp_cake_dict = {}
        self.monster = {}

        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.organ_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 8
        self.unit_buff_num = 1

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "organ_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.organ_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.organ_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
    
    # 获取双方血包对象
    def generate_cake_info_dict(self, frame_state):
        self.main_camp_cake_dict.clear()
        self.enemy_camp_cake_dict.clear()

        cakes = frame_state.get("cakes", [])
        if not cakes:
            return

        main_tower_loc = self.main_camp_organ_dict["tower"]["location"]
        enemy_tower_loc = self.enemy_camp_organ_dict["tower"]["location"]

        for idx, cake in enumerate(cakes):
            cake_loc = cake["collider"]["location"]
            dist_main = (cake_loc["x"] - main_tower_loc["x"]) ** 2 + (cake_loc["z"] - main_tower_loc["z"]) ** 2
            dist_enemy = (cake_loc["x"] - enemy_tower_loc["x"]) ** 2 + (cake_loc["z"] - enemy_tower_loc["z"]) ** 2

            if dist_main < dist_enemy:
                self.main_camp_cake_dict["cake"] = cake
            else:
                self.enemy_camp_cake_dict["cake"] = cake

    def process_vec_organ(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)
        self.generate_cake_info_dict(frame_state)
        self.generate_monster_info_list(frame_state)

        local_vector_feature = []

        # Generate features for enemy team's towers
        # 生成敌方阵营的防御塔特征
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.enemy_camp_organ_dict, "enemy_camp")
        local_vector_feature.extend(enemy_camp_organ_vector_feature)

        vector_feature = local_vector_feature
        return vector_feature

    def generate_monster_info_list(self, frame_state):
        self.monster.clear()
        for npc in frame_state["npc_states"]:
            if npc["camp"] != "PLAYERCAMP_2" and npc["camp"] != "PLAYERCAMP_2":
                self.main_camp_hero_dict['monster'] = npc

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

        for organ in frame_state["npc_states"]:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == self.main_camp:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    self.main_camp_organ_dict["tower"] = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    self.enemy_camp_organ_dict["tower"] = organ

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](organ, value)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_organs_considered += 1

        if "tower" in one_type_organ_info:
            organ = one_type_organ_info["tower"]
            process_organ(organ)

        if num_organs_considered < self.unit_buff_num:
            self.no_organ_feature(vector_feature, num_organs_considered)
        return vector_feature

    def no_organ_feature(self, vector_feature, num_organs_considered):
        for _ in range((self.unit_buff_num - num_organs_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_main_hp_rate(self, organ, vector_feature):
        organ = self.main_camp_organ_dict['tower']
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def get_enemy_hp_rate(self, organ, vector_feature):
        organ = self.enemy_camp_organ_dict['tower']
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False

    def mian_is_alive(self, organ, vector_feature):
        organ = self.main_camp_organ_dict['tower']
        value = 0.0
        if organ["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def enemy_is_alive(self, organ, vector_feature):
        organ = self.enemy_camp_organ_dict['tower']
        value = 0.0
        if organ["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_main_organ_location(self, organ, vector_feature):
        organ = self.main_camp_organ_dict['tower']
        value = organ["location"]["x"] / 60000
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

        value = organ["location"]["z"] / 60000
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_enemy_organ_location(self, organ, vector_feature):
        organ = self.enemy_camp_organ_dict['tower']
        value = organ["location"]["x"] / 60000
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

        value = organ["location"]["z"] / 60000
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_main_relative_location(self, organ, vector_feature):
        organ = self.main_camp_organ_dict['tower']
        organ_location_x = organ["location"]["x"]
        location_x = self.main_hero_info["actor_state"]["location"]["x"]
        x_diff = organ_location_x - location_x
        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

        organ_location_z = organ["location"]["z"]
        location_z = self.main_hero_info["actor_state"]["location"]["z"]
        z_diff = organ_location_z - location_z
        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def get_enemy_relative_location(self, organ, vector_feature):
        organ = self.enemy_camp_organ_dict['tower']
        organ_location_x = organ["location"]["x"]
        location_x = self.main_hero_info["actor_state"]["location"]["x"]
        x_diff = organ_location_x - location_x
        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

        organ_location_z = organ["location"]["z"]
        location_z = self.main_hero_info["actor_state"]["location"]["z"]
        z_diff = organ_location_z - location_z
        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def get_mian_tower_hp(self, organ, vector_feature):
        organ = self.main_camp_organ_dict['tower']
        value = organ["hp"]/6000
        vector_feature.append(value)

    def get_enemy_tower_hp(self, organ, vector_feature):
        organ = self.enemy_camp_organ_dict['tower']
        value = organ["hp"]/6000
        vector_feature.append(value)

    # 得到塔攻击目标和英雄的攻击目标
    def get_main_tower_target(self, organ, vector_feature):
        # 塔的攻击目标
        main_tower_target = self.main_camp_organ_dict["tower"]["attack_target"]*1e-3
        vector_feature.append(main_tower_target)

    # 得到塔攻击目标和英雄的攻击目标
    def get_enemy_tower_target(self, organ, vector_feature):
        enemy_tower_target = self.enemy_camp_organ_dict["tower"]["attack_target"]*1e-3
        vector_feature.append(enemy_tower_target)

    
    # 获取塔的攻击范围
    def get_tower_attack_range(self, organ, vector_feature):
        main_tower = self.main_camp_organ_dict['tower']
        vector_feature.append(main_tower['attack_range']*1e-4)

        # main_camp_cake_useable
    def is_main_camp_cake_useable(self, organ, vector_feature):
        cake = self.main_camp_cake_dict.get("cake",{})
        if not cake:
            vector_feature.append(0.0)
        else:
            vector_feature.append(1.0)

    # enemy_camp_cake_useable
    def is_enemy_camp_cake_useable(self, organ, vector_feature):
        cake = self.enemy_camp_cake_dict.get("cake",{})
        if not cake:
            vector_feature.append(0.0)
        else:
            vector_feature.append(1.0)

    # main_camp_cake_x
    def get_main_camp_cake_x(self, organ, vector_feature):
        cake = self.main_camp_cake_dict.get("cake",{})
        if not cake:
            vector_feature.append(0)
        else:
            value = cake["collider"]["location"]["x"] / 60000
            vector_feature.append(value)

    # main_camp_cake_z
    def get_main_camp_cake_z(self, organ, vector_feature):
        cake = self.main_camp_cake_dict.get("cake",{})
        if not cake:
            vector_feature.append(0)
        else:
            value = cake["collider"]["location"]["z"] / 60000
            vector_feature.append(value)

    # enemy_camp_cake_x
    def get_enemy_camp_cake_x(self, organ, vector_feature):
        cake = self.enemy_camp_cake_dict.get("cake",{})
        if not cake:
            vector_feature.append(0)
        else:
            value = cake["collider"]["location"]["x"] / 60000
            vector_feature.append(value)

    # enemy_camp_cake_z
    def get_enemy_camp_cake_z(self, organ, vector_feature):
        cake = self.enemy_camp_cake_dict.get("cake",{})
        if not cake:
            vector_feature.append(0)
        else:
            value = cake["collider"]["location"]["z"] / 60000
            vector_feature.append(value)

    def get_mian_runtime_id(self, organ, vector_feature):
        vector_feature.append(self.main_camp_organ_dict['tower']['runtime_id']*1e-3)

    def get_enemy_runtime_id(self, organ, vector_feature):
        vector_feature.append(self.enemy_camp_organ_dict['tower']['runtime_id']*1e-3)

    def get_monster_feature(self, organ, vector_feature):
        monster = self.monster
        monster_feature = [0 for i in range(4)]
        if not monster:
            vector_feature.extend(monster_feature)
        else:
            monster_feature[0] = monster.get('hp',0) / 6000
            monster_feature[1] = monster.get('kill_income',0) / 12000
            monster_feature[2] = monster.get('location',{}).get('x',0) / 60000
            monster_feature[3] = monster.get('location',{}).get('z',0) / 60000
            vector_feature.extend(monster_feature)

