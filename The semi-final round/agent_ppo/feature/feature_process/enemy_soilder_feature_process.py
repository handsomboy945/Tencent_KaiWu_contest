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


class EnemySoilderProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000
        # 特征的维度
        self.one_unit_feature_num = 172
        self.unit_buff_num = 1
        # organ信息
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}
        # cake信息
        self.main_camp_cake_dict = {}
        self.enemy_camp_cake_dict = {}
        # bullet信息
        self.main_camp_bullet_dict = {}
        self.enemy_camp_bullet_dict = {}
        # soldier信息
        self.main_camp_soldier_dict = {}
        self.enemy_camp_soldier_dict = {}
        self.enemy_location = [0,0]
        self.enemy_hp = 0

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "enemy_soilder_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):

        self.generate_hero_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)
        # 获取organ信息
        self.generate_organ_info_dict(frame_state)
        # 获取子弹信息
        self.generate_bullet_info_dict(frame_state)
        # 获取小兵信息
        self.generate_soldier_info_dict(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")

        frameNo = frame_state["frameNo"]
        bullets = frame_state.get("bullets", [])
        cakes = frame_state.get("cakes", [])
        # print(f"帧号：{frameNo}，我方英雄特征：{main_camp_hero_vector_feature}")
        # print(bullets)
        # print(self.enemy_camp_hero_dict)
        # print(self.main_camp_soldier_dict)
        # print(self.enemy_camp_soldier_dict)

        return main_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["actor_state"]["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["actor_state"]["config_id"]] = hero

    def generate_hero_info_dict(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()

        # Find our heroes and number them in order
        # 找到我方英雄并按照顺序编号
        for hero in frame_state["npc_states"]:
            if hero["sub_type"] != "ACTOR_SUB_hero" or hero["hp"] <= 0:
                continue
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["runtime_id"]] = hero
        self.main_camp_hero_dict = OrderedDict(sorted(self.main_camp_hero_dict.items()))

        # Find enemy heroes and number them in order
        # 找到敌方英雄并按照顺序编号
        for hero in frame_state["npc_states"]:
            if hero["sub_type"] != "ACTOR_SUB_hero" or hero["hp"] <= 0:
                continue
            if hero["camp"] != self.main_camp:
                self.enemy_camp_hero_dict[hero["runtime_id"]] = hero
        self.enemy_camp_hero_dict = OrderedDict(sorted(self.enemy_camp_hero_dict.items()))

    # 获取防御塔信息
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

    # 获取子弹信息
    def generate_bullet_info_dict(self, frame_state):
        self.main_camp_bullet_dict.clear()
        self.enemy_camp_bullet_dict.clear()

        for bullet in frame_state.get("bullets", []):
            bullet_camp = bullet["camp"]
            if bullet_camp == self.main_camp:   
                self.main_camp_bullet_dict[bullet["runtime_id"]] = bullet
            else:
                self.enemy_camp_bullet_dict[bullet["runtime_id"]] = bullet

    # 获取小兵信息
    def generate_soldier_info_dict(self, frame_state):
        self.main_camp_soldier_dict.clear()
        self.enemy_camp_soldier_dict.clear()

        for soldier in frame_state["npc_states"]:
            soldier_camp = soldier["camp"]
            soldier_subtype = soldier["sub_type"]
            if soldier_camp == self.main_camp:
                if soldier_subtype == "ACTOR_SUB_SOLDIER":
                    self.main_camp_soldier_dict[soldier["runtime_id"]] = soldier
            else:
                if soldier_subtype == "ACTOR_SUB_SOLDIER":
                    self.enemy_camp_soldier_dict[soldier["runtime_id"]] = soldier


    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    print(feature_name)
                    assert False
                for k in value:
                    value_vec = []
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)

        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["actor_state"]["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    # 新增，获取英雄的当前hp
    def get_current_hp(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["hp"]
        vector_feature.append(value)

    # 新增，获取最大hp
    def get_max_hp(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["max_hp"]
        vector_feature.append(value)

    # 新增：获取英雄的当前hp比例，dim=1
    def get_current_hp_rate(self, hero, vector_feature, feature_name):
        cur_hp = hero["actor_state"]["hp"]
        max_hp = hero["actor_state"]["max_hp"]
        value = cur_hp/max_hp
        vector_feature.append(value)

    # 新增：获取英雄的等级（1-15），dim=15
    def get_hero_level(self, hero, vector_feature, feature_name):
        value = hero["level"]
        vector_feature.append(value)

    # 新增，获取生命回复属性
    def get_hp_recover(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["hp_recover"]
        vector_feature.append(value)

    # 新增，获取MP（数据协议里面是ep，法力值）
    def get_ep(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["ep"]
        vector_feature.append(value)

    # 获取ep_rate
    def get_ep_rate(self, hero, vector_feature, feature_name):
        ep = hero["actor_state"]["values"]["ep"]
        max_ep = hero["actor_state"]["values"]["max_ep"]
        value = ep/max_ep
        vector_feature.append(value)
    
    # 获取max_ep
    def get_max_ep(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["max_ep"]
        vector_feature.append(value)

    # 获取ep_recover
    def get_ep_recover(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["ep_recover"]
        vector_feature.append(value)

    # 获取phy_atk
    def get_phy_atk(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["phy_atk"]
        vector_feature.append(value)

    # 获取phy_def
    def get_phy_def(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["phy_def"]
        vector_feature.append(value)

    # 获取mgc_atk
    def get_mgc_atk(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["mgc_atk"]
        vector_feature.append(value)

    # 获取mgc_def
    def get_mgc_def(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["mgc_def"]
        vector_feature.append(value)

    # 获取killCnt
    def get_killCnt(self, hero, vector_feature, feature_name):
        value = hero["killCnt"]
        vector_feature.append(value)

    # 获取deadCnt
    def get_deadCnt(self, hero, vector_feature, feature_name):
        value = hero["deadCnt"]
        vector_feature.append(value)

    # 获取moneyCnt
    def get_moneyCnt(self, hero, vector_feature, feature_name):
        value = hero["moneyCnt"]
        vector_feature.append(value)

    # 获取dist_from_all_heros（与敌方英雄的距离）
    def get_dist_from_all_heros(self, hero, vector_feature, feature_name):
        hx, hz = hero["actor_state"]["location"]["x"], hero["actor_state"]["location"]["z"]

        # 如果开启换边逻辑，镜像坐标
        if self.transform_camp2_to_camp1 and hx != 100000 and hz != 100000:
            hx, hz = -hx, -hz
        
        # 如果我方英雄死亡，直接所有距离填最大值
        if hero["actor_state"]["hp"] <= 0:
            for _ in self.enemy_camp_hero_dict:
                vector_feature.append(116000.0)
            return
        
        for _, enemy in self.enemy_camp_hero_dict.items():
            # 敌方死亡，最大距离
            if enemy["actor_state"]["hp"] <= 0:
                dist = 116000.0
            else:
                ex, ez = enemy["actor_state"]["location"]["x"], enemy["actor_state"]["location"]["z"]
                if self.transform_camp2_to_camp1 and ex != 100000 and ez != 100000:
                    ex, ez = -ex, -ez
                dist = math.sqrt((hx - ex) ** 2 + (hz - ez) ** 2)
            vector_feature.append(dist)

    # 获取mov_spd
    def get_mov_spd(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["mov_spd"]
        vector_feature.append(value)

    # 获取attack_range
    def get_attack_range(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["attack_range"]
        vector_feature.append(value)

    # atk_spd
    def get_atk_spd(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["atk_spd"]
        vector_feature.append(value)

    # phy_armor_hurt
    def get_phy_armor_hurt(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["phy_armor_hurt"]
        vector_feature.append(value)

    # mgc_armor_hurt
    def get_mgc_armor_hurt(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["mgc_armor_hurt"]
        vector_feature.append(value)

    # crit_rate
    def get_crit_rate(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["crit_rate"]
        vector_feature.append(value)

    # crit_effe
    def get_crit_effe(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["crit_effe"]
        vector_feature.append(value) 

    # phy_vamp
    def get_phy_vamp(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["phy_vamp"]
        vector_feature.append(value)

    # mgc_vamp
    def get_mgc_vamp(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["mgc_vamp"]
        vector_feature.append(value)

    # cd_reduce
    def get_cd_reduce(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["cd_reduce"]
        vector_feature.append(value)

    # ctrl_reduce
    def get_ctrl_reduce(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["values"]["ctrl_reduce"]
        vector_feature.append(value)

    # exp
    def get_exp(self, hero, vector_feature, feature_name):
        value = hero["exp"]
        vector_feature.append(value)

    # money
    def get_money(self, hero, vector_feature, feature_name):
        value = hero["money"]
        vector_feature.append(value)

    # revive_time
    def get_revive_time(self, hero, vector_feature, feature_name):
        value = hero["revive_time"]
        vector_feature.append(value)

    # kill_income
    def get_kill_income(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["kill_income"]
        vector_feature.append(value)

    # skill_1_useable
    def is_skill_1_useable(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][1]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # hero_skill_1_cd
    def get_hero_skill_1_cd(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][1]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][1]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][1]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # skill_2_useable
    def is_skill_2_useable(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][2]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # hero_skill_2_cd
    def get_hero_skill_2_cd(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][2]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][2]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][2]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # skill_3_useable
    def is_skill_3_useable(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][3]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # hero_skill_3_cd
    def get_hero_skill_3_cd(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][3]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][3]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][3]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # heal_skill_useable
    def is_heal_skill_useable(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][4]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # heal_skill_cd
    def get_heal_skill_cd(self, hero, vector_feature, feature_name):
        value = hero["skill_state"]["slot_states"][4]["cooldown"]
        vector_feature.append(value)

    # summon_skill_useable
    def is_summon_skill_useable(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][5]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # summon_skill_cd
    def get_summon_skill_cd(self, hero, vector_feature, feature_name):
        value = hero["skill_state"]["slot_states"][5]["cooldown"]
        vector_feature.append(value)

    # normal_attack_useable
    def is_normal_attack_useable(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][0]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # normal_attack_cd
    def get_normal_attack_cd(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["skill_state"]["slot_states"][0]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][0]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][0]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # hero_in_main_camp_tower_atk_range
    def is_hero_in_main_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        tower = self.main_camp_organ_dict["tower"]
        tower_loc = tower["location"]
        atk_range = tower["attack_range"]
        hero_loc = hero["actor_state"]["location"]

        dx = tower_loc["x"] - hero_loc["x"]
        dz = tower_loc["z"] - hero_loc["z"]
        dist_sq = math.sqrt(dx**2 + dz**2)

        if dist_sq <= atk_range:
            vector_feature.append(1.0)
        else:
            vector_feature.append(0.0)
        
    # hero_in_enemy_camp_tower_atk_range
    def is_hero_in_enemy_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        tower = self.enemy_camp_organ_dict["tower"]
        tower_loc = tower["location"]
        atk_range = tower["attack_range"]
        hero_loc = hero["actor_state"]["location"]

        dx = tower_loc["x"] - hero_loc["x"]
        dz = tower_loc["z"] - hero_loc["z"]
        dist_sq = math.sqrt(dx**2 + dz**2)

        if dist_sq <= atk_range:
            vector_feature.append(1.0)
        else:
            vector_feature.append(0.0)

    # hero_under_tower_atk
    def is_hero_under_tower_atk(self, hero, vector_feature, feature_name):
        hero_id = hero["actor_state"]["runtime_id"]
        tower = self.enemy_camp_organ_dict["tower"]
        target_id = tower['attack_target']

        if target_id == hero_id:
            vector_feature.append(1.0)
        else:
            vector_feature.append(0.0)

    # cur_location_of_nearest_enemy_bullet_of_enemy_x_diff
    def get_cur_location_of_nearest_enemy_bullet_of_enemy_x_diff(self, hero, vector_feature, feature_name):
        hx = hero["actor_state"]["location"]["x"]
        hz = hero["actor_state"]["location"]["z"]

        if self.transform_camp2_to_camp1 and hx != 100000:
            hx = -hx
            hz = -hz

        nearest_bullet = None
        min_dist = float("inf")

        # 遍历敌方子弹
        for bullet in self.enemy_camp_bullet_dict.values():
            bx = bullet["location"]["x"]
            bz = bullet["location"]["z"]

            if self.transform_camp2_to_camp1 and bx != 100000:
                bx = -bx
                bz = -bz

            # 计算与英雄的距离
            dist = math.sqrt((hx - bx) ** 2 + (hz - bz) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_bullet = bullet

        # 如果找到最近的敌方子弹，计算x差值，否则给一个默认大值
        if nearest_bullet is not None:
            bx = nearest_bullet["location"]["x"]
            x_diff = hx - bx   # 英雄位置减子弹位置
            vector_feature.append(x_diff)
        else:
            vector_feature.append(12000)

    # cur_location_of_nearest_enemy_bullet_of_enemy_z_diff
    def get_cur_location_of_nearest_enemy_bullet_of_enemy_z_diff(self, hero, vector_feature, feature_name):
        hx = hero["actor_state"]["location"]["x"]
        hz = hero["actor_state"]["location"]["z"]

        if self.transform_camp2_to_camp1 and hx != 100000:
            hx = -hx
            hz = -hz

        nearest_bullet = None
        min_dist = float("inf")

        # 遍历敌方子弹
        for bullet in self.enemy_camp_bullet_dict.values():
            bx = bullet["location"]["x"]
            bz = bullet["location"]["z"]

            if self.transform_camp2_to_camp1 and bx != 100000:
                bx = -bx
                bz = -bz

            # 计算与英雄的距离
            dist = math.sqrt((hx - bx) ** 2 + (hz - bz) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_bullet = bullet

        # 如果找到最近的敌方子弹，计算x差值，否则给一个默认大值
        if nearest_bullet is not None:
            bz = nearest_bullet["location"]["z"]
            z_diff = hz - bz   # 英雄位置减子弹位置
            vector_feature.append(z_diff)
        else:
            vector_feature.append(12000)

    # enemy_in_attack_range
    def is_enemy_in_attack_range(self, hero, vector_feature, feature_name):
        atk_range = hero["actor_state"]["attack_range"]
        hx = hero["actor_state"]["location"]["x"]
        hz = hero["actor_state"]["location"]["z"]
        if self.transform_camp2_to_camp1 and hx != 100000 and hz != 100000:
            hx, hz = -hx, -hz

        # 如果主英雄死亡或坐标无效，全部记作不在射程内
        if hero["actor_state"]["hp"] <= 0 or hx == 100000 or hz == 100000:
            for _ in self.enemy_camp_hero_dict:
                vector_feature.append(0.0)
            return

        for _, enemy in self.enemy_camp_hero_dict.items():
            if enemy["actor_state"]["hp"] <= 0:
                vector_feature.append(0.0)
                continue

            ex = enemy["actor_state"]["location"]["x"]
            ez = enemy["actor_state"]["location"]["z"]
            if ex == 100000 or ez == 100000:
                vector_feature.append(0.0)
                continue

            if self.transform_camp2_to_camp1:
                ex, ez = -ex, -ez

            dx, dz = ex - hx, ez - hz
            dist = math.sqrt(dx * dx + dz * dz)

            # 在射程内 -> 1.0，否则 0.0
            vector_feature.append(1.0 if dist <= atk_range else 0.0)




    # 小兵相关特征
    # in_main_exp_range
    def is_in_main_exp_range(self, hero, vector_feature, feature_name):
        hero_vision_range = hero["actor_state"]["sight_area"]
        hx = hero["actor_state"]["location"]["x"]
        hz = hero["actor_state"]["location"]["z"]
        # 遍历“敌方小兵”找最近距离（平方距离，避免 sqrt）
        best_d2 = float("inf")
        soldiers = getattr(self, "enemy_camp_soldier_dict", {}) or {}
        for s in soldiers.values():
            loc = s.get("location") or s.get("actor_state", {}).get("location")
            if not isinstance(loc, dict):
                continue
            sx, sz = loc.get("x", 0), loc.get("z", 0)
            if abs(sx) >= 1e5 or abs(sz) >= 1e5:
                continue
            dx, dz = float(hx) - float(sx), float(hz) - float(sz)
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2 = d2

        vector_feature.append(1.0 if best_d2 <= hero_vision_range * hero_vision_range else 0.0)

    # in_enemy_exp_range
    def is_in_enemy_exp_range(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_vision_range = hero["actor_state"]["sight_area"]
        hx = hero["actor_state"]["location"]["x"]
        hz = hero["actor_state"]["location"]["z"]

        best_d2 = float("inf")
        soldiers = getattr(self, "main_camp_soldier_dict", {}) or {}
        for s in soldiers.values():
            loc = s.get("location") or s.get("actor_state", {}).get("location")
            if not isinstance(loc, dict):
                continue
            sx, sz = loc.get("x", 100000), loc.get("z", 100000)
            if abs(sx) >= 1e5 or abs(sz) >= 1e5:
                continue
            dx, dz = float(hx) - float(sx), float(hz) - float(sz)
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2 = d2

        vector_feature.append(1.0 if best_d2 <= hero_vision_range * hero_vision_range else 0.0)

    # ===== 我方英雄：最近【我方小兵】5维 =====
    def get_nearest_main_soldier_location_x(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist = float("inf")
        nearest = None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["x"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def get_nearest_main_soldier_location_z(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["z"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def nearest_main_soldier_relative_location_x(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(1)
            return
        x_diff = nearest["location"]["x"] - hero_pos["x"]
        if self.transform_camp2_to_camp1 and nearest["location"]["x"] != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_main_soldier_relative_location_z(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(1)
            return
        z_diff = nearest["location"]["z"] - hero_pos["z"]
        if self.transform_camp2_to_camp1 and nearest["location"]["z"] != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_main_soldier_distance(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist = float("inf")
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist = dist
        vector_feature.append(0.0 if min_dist == float("inf") else min_dist)

    # ===== 我方英雄：最近【敌方小兵】5维 =====
    def get_nearest_enemy_soldier_location_x(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["x"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def get_nearest_enemy_soldier_location_z(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["z"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def nearest_enemy_soldier_relative_location_x(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(0.5)
            return
        x_diff = nearest["location"]["x"] - hero_pos["x"]
        if self.transform_camp2_to_camp1 and nearest["location"]["x"] != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_enemy_soldier_relative_location_z(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(0.5)
            return
        z_diff = nearest["location"]["z"] - hero_pos["z"]
        if self.transform_camp2_to_camp1 and nearest["location"]["z"] != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_enemy_soldier_distance(self, hero, vector_feature, feature_name):
        hero_pos = hero["actor_state"]["location"]
        min_dist = float("inf")
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist = dist
        vector_feature.append(0.0 if min_dist == float("inf") else min_dist)

    # ===== 敌方英雄：相同 10 维（互换小兵阵营） =====
    def get_nearest_main_soldier_location_x_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():  # 敌方英雄的“友军”=enemy_camp
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["x"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def get_nearest_main_soldier_location_z_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["z"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def nearest_main_soldier_relative_location_x_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(0.5)
            return
        x_diff = nearest["location"]["x"] - hero_pos["x"]
        if self.transform_camp2_to_camp1 and nearest["location"]["x"] != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_main_soldier_relative_location_z_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(0.5)
            return
        z_diff = nearest["location"]["z"] - hero_pos["z"]
        if self.transform_camp2_to_camp1 and nearest["location"]["z"] != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_main_soldier_distance_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist = float("inf")
        for soldier in getattr(self, "enemy_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist = dist
        vector_feature.append(0.0 if min_dist == float("inf") else min_dist)

    def get_nearest_enemy_soldier_location_x_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():  # 敌方英雄的“敌军”=main_camp
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["x"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def get_nearest_enemy_soldier_location_z_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        value = 100000
        if nearest is not None:
            value = nearest["location"]["z"]
            if self.transform_camp2_to_camp1 and value != 100000:
                value = -value
        vector_feature.append(value)

    def nearest_enemy_soldier_relative_location_x_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(0.5)
            return
        x_diff = nearest["location"]["x"] - hero_pos["x"]
        if self.transform_camp2_to_camp1 and nearest["location"]["x"] != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_enemy_soldier_relative_location_z_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist, nearest = float("inf"), None
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist, nearest = dist, soldier
        if nearest is None:
            vector_feature.append(0.5)
            return
        z_diff = nearest["location"]["z"] - hero_pos["z"]
        if self.transform_camp2_to_camp1 and nearest["location"]["z"] != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def nearest_enemy_soldier_distance_for_enemy(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        hero_pos = hero["actor_state"]["location"]
        min_dist = float("inf")
        for soldier in getattr(self, "main_camp_soldier_dict", {}).values():
            if soldier.get("hp", 1) <= 0:
                continue
            sx, sz = soldier["location"]["x"], soldier["location"]["z"]
            dx = (hero_pos["x"] - sx) / 100.0
            dz = (hero_pos["z"] - sz) / 100.0
            dist = (dx * dx + dz * dz) ** 0.5
            if dist < min_dist:
                min_dist = dist
        vector_feature.append(0.0 if min_dist == float("inf") else min_dist)

    def nearest_soldier_attr(self, hero, vector_feature, feature_name):
        """
        统一处理 20 个“最近小兵”属性维度：
          nearest_main_soldier_{hp|hp_rate|max_hp|atk|kill_income}[_for_enemy]
          nearest_enemy_soldier_{hp|hp_rate|max_hp|atk|kill_income}[_for_enemy]
        """
        # 1) 解析“友军/敌军”与是否在敌方英雄通道
        #    这里不直接用 _for_enemy 做阵营反转，因为 hero 本身就是当前被遍历的那位英雄
        is_main_group = "nearest_main_soldier_" in feature_name
        is_for_enemy = feature_name.endswith("_for_enemy")  # 仅用于名字解析，无需改变 hero 的阵营判断

        # 2) 取当前英雄坐标与阵营
        actor = hero.get("actor_state", {})
        hero_camp = actor.get("camp", None)
        hero_loc = actor.get("location", {})
        hx, hz = hero_loc.get("x", 100000), hero_loc.get("z", 100000)
        if abs(hx) >= 1e5 or abs(hz) >= 1e5:
            vector_feature.append(0.0)
            return

        # 3) 针对“当前 hero”选择 友军/敌军 小兵字典
        #    当前 hero 的友军 = 与 hero_camp 相同的阵营；敌军 = 相反阵营
        friendly_dict = self.main_camp_soldier_dict if hero_camp == self.main_camp else self.enemy_camp_soldier_dict
        enemy_dict = self.enemy_camp_soldier_dict if hero_camp == self.main_camp else self.main_camp_soldier_dict
        soldiers_dict = friendly_dict if is_main_group else enemy_dict

        # 4) 找最近的小兵（仅用 x/z，比较平方距离）
        nearest = None
        best_d2 = float("inf")
        for s in (soldiers_dict or {}).values():
            loc = s.get("location") or s.get("actor_state", {}).get("location")
            if not isinstance(loc, dict):
                continue
            sx, sz = loc.get("x", 100000), loc.get("z", 100000)
            if abs(sx) >= 1e5 or abs(sz) >= 1e5:
                continue
            dx, dz = float(hx) - float(sx), float(hz) - float(sz)
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2, nearest = d2, s

        if nearest is None:
            vector_feature.append(0.0)
            return

        # 5) 解析属性名：hp / hp_rate / max_hp / atk / kill_income
        attr = feature_name
        attr = attr.replace("nearest_main_soldier_", "").replace("nearest_enemy_soldier_", "")
        attr = attr.replace("_for_enemy", "")

        # 6) 取值（缺失兜底为 0；hp_rate 优先字段，否则 hp/max_hp）
        hp = float(nearest.get("hp", 0.0))
        max_hp = float(nearest.get("max_hp", 0.0))
        atk = float(nearest.get("atk", 0.0))
        kill_income = float(nearest.get("kill_income", 0.0))
        if "hp_rate" in nearest:
            hp_rate = float(nearest.get("hp_rate", 0.0))
        else:
            hp_rate = (hp / max_hp) if max_hp > 0 else 0.0

        if attr == "hp":
            value = hp
        elif attr == "hp_rate":
            value = hp_rate
        elif attr == "max_hp":
            value = max_hp
        elif attr == "atk":
            value = atk
        elif attr == "kill_income":
            value = kill_income
        else:
            value = 0.0

        vector_feature.append(value)



    # 与敌方英雄有关的特征
    # enemy_alive
    def is_enemy_alive(self, hero, vector_feature, feature_name):
        res = 0.0
        for _, value in self.enemy_camp_hero_dict.items():
            hp = value["actor_state"]["hp"]
            if hp > 0 :
                res = 1.0
        vector_feature.append(res)

    # enemy_level
    def get_enemy_level(self, hero, vector_feature, feature_name):
        value = 0.0
        for _, enemy in self.enemy_camp_hero_dict.items():
            value = enemy["level"]
        vector_feature.append(value)

    # enemy_max_hp
    def get_enemy_max_hp(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["max_hp"]
        vector_feature.append(value)

    # enemy_hp_rate
    def get_enemy_hp_rate(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        cur_hp = hero["actor_state"]["hp"]
        max_hp = hero["actor_state"]["max_hp"]
        value = cur_hp/max_hp
        vector_feature.append(value)

    # enemy_hp_recover
    def get_enemy_hp_recover(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["hp_recover"]
        vector_feature.append(value)

    # get_enemy_ep
    def get_enemy_ep(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["ep"]
        vector_feature.append(value)

    # enemy_ep_rate
    def get_enemy_ep_rate(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        ep = hero["actor_state"]["values"]["ep"]
        max_ep = hero["actor_state"]["values"]["max_ep"]
        value = ep/max_ep
        vector_feature.append(value)
    
    # enemy_max_ep
    def get_enemy_max_ep(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["max_ep"]
        vector_feature.append(value)

    # enemy_ep_recover
    def get_enemy_ep_recover(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["ep_recover"]
        vector_feature.append(value)

    # enemy_phy_atk
    def get_enemy_phy_atk(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["phy_atk"]
        vector_feature.append(value)

    # enemy_phy_def
    def get_enemy_phy_def(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["phy_def"]
        vector_feature.append(value)

    # enemy_mgc_atk
    def get_enemy_mgc_atk(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["mgc_atk"]
        vector_feature.append(value)

    # enemy_mgc_def
    def get_enemy_mgc_def(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["mgc_def"]
        vector_feature.append(value)

    # enemy_killCnt
    def get_enemy_killCnt(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["killCnt"]
        vector_feature.append(value)

    # enemy_deadCnt
    def get_enemy_deadCnt(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["deadCnt"]
        vector_feature.append(value)

    # enemy_moneyCnt
    def get_enemy_moneyCnt(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["moneyCnt"]
        vector_feature.append(value)

    # enemy_mov_spd
    def get_enemy_mov_spd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["mov_spd"]
        vector_feature.append(value)

    # enemy_attack_range
    def get_enemy_attack_range(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["attack_range"]
        vector_feature.append(value)

    # enemy_atk_spd
    def get_enemy_atk_spd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["atk_spd"]
        vector_feature.append(value)

    # enemy_phy_armor_hurt
    def get_enemy_phy_armor_hurt(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["phy_armor_hurt"]
        vector_feature.append(value)

    # enemy_mgc_armor_hurt
    def get_enemy_mgc_armor_hurt(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["mgc_armor_hurt"]
        vector_feature.append(value)

    # enemy_crit_rate
    def get_enemy_crit_rate(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["crit_rate"]
        vector_feature.append(value)

    # enemy_crit_effe
    def get_enemy_crit_effe(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["crit_effe"]
        vector_feature.append(value) 

    # enemy_phy_vamp
    def get_enemy_phy_vamp(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["phy_vamp"]
        vector_feature.append(value)

    # enemy_mgc_vamp
    def get_enemy_mgc_vamp(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["mgc_vamp"]
        vector_feature.append(value)

    # enemy_cd_reduce
    def get_enemy_cd_reduce(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["cd_reduce"]
        vector_feature.append(value)

    # enemy_ctrl_reduce
    def get_enemy_ctrl_reduce(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["values"]["ctrl_reduce"]
        vector_feature.append(value)

    # enemy_exp
    def get_enemy_exp(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["exp"]
        vector_feature.append(value)

    # enemy_money
    def get_enemy_money(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["money"]
        vector_feature.append(value)

    # enemy_revive_time
    def get_enemy_revive_time(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["revive_time"]
        vector_feature.append(value)

    # enemy_kill_income
    def get_enemy_kill_income(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["actor_state"]["kill_income"]
        vector_feature.append(value)

    # enemy_skill_1_useable
    def is_enemy_skill_1_useable(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][1]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # enemy_skill_1_cd
    def get_enemy_skill_1_cd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][1]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][1]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][1]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # enemy_skill_2_useable
    def is_enemy_skill_2_useable(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][2]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # enemy_skill_2_cd
    def get_enemy_skill_2_cd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][2]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][2]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][2]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # enemy_skill_3_useable
    def is_enemy_skill_3_useable(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][3]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # enemy_skill_3_cd
    def get_enemy_skill_3_cd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][3]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][3]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][3]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # enemy_heal_skill_useable
    def is_enemy_heal_skill_useable(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][4]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # enemy_heal_skill_cd
    def get_enemy_heal_skill_cd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["skill_state"]["slot_states"][4]["cooldown"]
        vector_feature.append(value)

    # enemy_summon_skill_useable
    def is_enemy_summon_skill_useable(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][5]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # enemy_summon_skill_cd
    def get_enemy_summon_skill_cd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = hero["skill_state"]["slot_states"][5]["cooldown"]
        vector_feature.append(value)

    # enemy_normal_attack_useable
    def is_enemy_normal_attack_useable(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][0]["usable"]:
            value = 1.0
        vector_feature.append(value)

    # enemy_normal_attack_cd
    def get_enemy_normal_attack_cd(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        value = 0.0
        if hero["skill_state"]["slot_states"][0]["level"] != 0:
            cd = hero["skill_state"]["slot_states"][0]["cooldown"]
            max_cd = hero["skill_state"]["slot_states"][0]["cooldown_max"]
            value = cd/max_cd
        else:
            value = 1.0
        vector_feature.append(value)

    # 获得双方英雄的英雄类型
    def get_enemy_config_id(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enenmy_config_id = hero["actor_state"]["config_id"] * 1e-5
        vector_feature.append(enenmy_config_id)

    def get_hero_config_id(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_config_id = hero["actor_state"]["config_id"] * 1e-5
        vector_feature.append(main_config_id)
        # print(f"英雄id{enenmy_config_id}{main_config_id}")

    # 获取我方英雄行为
    def get_hero_behav_mode(self, hero, vector_feature, feature_name):
        behav_list = ["state_idle", "Normal_Attack", "Direction_move", "Useskill_1", "Useskill_2", "Useskill_3", "death"]
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_behav_mode = hero["actor_state"]["behav_mode"]
        main_one_hot = [1 if main_behav_mode.lower() == behav.lower() else 0 for behav in behav_list]
        vector_feature.extend(main_one_hot)

    # 获取我方英雄行为
    def get_enemy_behav_mode(self, hero, vector_feature, feature_name):
        behav_list = ["state_idle", "Normal_Attack", "Direction_move", "Useskill_1", "Useskill_2", "Useskill_3", "death"]
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_behav_mode = hero["actor_state"]["behav_mode"]
        enemy_one_hot = [1 if enemy_behav_mode.lower() == behav.lower() else 0 for behav in behav_list]
        vector_feature.extend(enemy_one_hot)
        # print(f"英雄行为{main_one_hot}{enemy_one_hot}")

    # 获取我方朝向
    def get_hero_forward(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_forward = hero["actor_state"]['forward']
        main_forward = [i/1000 for i in main_forward.values()]
        if self.transform_camp2_to_camp1:\
            main_forward = [0-i for i in main_forward]
        vector_feature.extend(main_forward)

    # 获取敌方朝向
    def get_enemy_forward(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_forward = hero["actor_state"]['forward']
        enemy_forward = [i/1000 for i in enemy_forward.values()]
        if hero["actor_state"]['camp_visible'][1] or hero["actor_state"]['camp_visible'][0]==False:
            enemy_forward = [0, 0, 0]
        if self.transform_camp2_to_camp1:
            enemy_forward = [0-i for i in enemy_forward]
        vector_feature.extend(enemy_forward)
        # print(f"英雄朝向{main_forward}{enemy_forward}")

    # 获取我方击中目标信息
    def get_hero_hit_target_runtimeid(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_hit_target_info = hero["actor_state"].get('hit_target_info', 0)
        if main_hit_target_info == 0:
            vector_feature.append(0)
        else:
            vector_feature.append(main_hit_target_info[0]['hit_target']*1e-3)

    # 获取敌方击中目标信息
    def get_enemy_hit_target_runtimeid(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_hit_target_info = hero["actor_state"].get('hit_target_info', 0)
        if enemy_hit_target_info == 0:
            vector_feature.append(0)
        else:
            vector_feature.append(enemy_hit_target_info[0]['hit_target']*1e-3)
        # print(f"英雄命中目标{vector_feature[-1]}{vector_feature[-1]}")

    def get_hero_hit_target_skill_id(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_hit_target_info = hero["actor_state"].get('hit_target_info', 0)
        if main_hit_target_info == 0:
            vector_feature.append(0)
        else:
            vector_feature.append(main_hit_target_info[0]['skill_id']*1e-3)

    def get_enemy_hit_target_skill_id(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_hit_target_info = hero["actor_state"].get('hit_target_info', 0)
        if enemy_hit_target_info == 0:
            vector_feature.append(0)
        else:
            vector_feature.append(enemy_hit_target_info[0]['skill_id']*1e-3)
        # print(f"英雄命中技能{vector_feature[-1]}{vector_feature[-1]}")

    def get_hero_conti_hit_count(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_hit_target_info = hero["actor_state"].get('hit_target_info', 0)
        if main_hit_target_info == 0:
            vector_feature.append(0)
        else:
            vector_feature.append(main_hit_target_info[0]['conti_hit_count']*0.1)

    def get_enemy_conti_hit_count(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_hit_target_info = hero["actor_state"].get('hit_target_info', 0)
        if enemy_hit_target_info == 0:
            vector_feature.append(0)
        else:
            vector_feature.append(enemy_hit_target_info[0]['conti_hit_count']*0.1)
        # print(f"英雄命中次数{vector_feature[-1]}{vector_feature[-1]}")

    # 获取视野可见性
    def get_hero_camp_visible(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_hit_camp_visible = hero["actor_state"]['camp_visible'][0] and hero["actor_state"]['camp_visible'][1]
        main_hit_camp_visible = 1 if main_hit_camp_visible else 0
        vector_feature.append(main_hit_camp_visible)

    def get_enemy_camp_visible(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_hit_camp_visible = hero["actor_state"]['camp_visible'][0] and hero["actor_state"]['camp_visible'][1]
        enemy_hit_camp_visible = 1 if enemy_hit_camp_visible else 0
        vector_feature.append(enemy_hit_camp_visible)
        # print(f"视野可见性{main_hit_camp_visible}{enemy_hit_camp_visible}")

    # 我方获取buff状态
    def get_hero_buff_state(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_buff_state = hero["actor_state"]['buff_state'].get("buff_skills",{})
        main_buff_list = [0 for i in range(18)]
        for buff_list, buff_state in zip([main_buff_list], [main_buff_state]):
            buff_cnt = min(6, len(buff_state))
            for i in range(buff_cnt):
                buff_list[3*i] = buff_state[i]["configId"] * 1e-5
                buff_list[3*i+1] = buff_state[i]["times"]
                buff_list[3*i+2] = int(buff_state[i]["startTime"]) * 1e-5
        vector_feature.extend(main_buff_list)

    # 敌方获取buff状态
    def get_enemy_buff_state(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_buff_state = hero["actor_state"]['buff_state'].get("buff_skills",{})
        enemy_buff_list = [0 for i in range(18)]
        for buff_list, buff_state in zip([enemy_buff_list], [enemy_buff_state]):
            buff_cnt = min(6, len(buff_state))
            for i in range(buff_cnt):
                buff_list[3*i] = buff_state[i]["configId"] * 1e-5
                buff_list[3*i+1] = buff_state[i]["times"]
                buff_list[3*i+2] = int(buff_state[i]["startTime"]) * 1e-5
        vector_feature.extend(enemy_buff_list)
        # print(f"英雄buff{main_buff_list}{enemy_buff_list}")

    # 获取技能状态
    def get_hero_skill_state(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_skill_state = hero['skill_state']["slot_states"]
        main_skill_list = [0 for i in range(70)]

        for skill_list, skill_state in zip([main_skill_list], [main_skill_state]):
            for i in range(7):
                skill_list[10*i] = skill_state[i]["configId"]*1e-5
                skill_list[10*i+1] = skill_state[i]["level"]
                skill_list[10*i+2] = 1 if skill_state[i]["usable"] else 0
                skill_list[10*i+3] = skill_state[i]["cooldown"]*1e-5
                skill_list[10*i+4] = skill_state[i]["cooldown_max"]*1e-5
                skill_list[10*i+5] = skill_state[i]["usedTimes"]*0.1
                skill_list[10*i+6] = skill_state[i]["hitHeroTimes"]*1e-2
                skill_list[10*i+7] = skill_state[i]["succUsedInFrame"]*1e-5
                skill_list[10*i+8] = skill_state[i]["nextConfigID"]*1e-5
                skill_list[10*i+9] = skill_state[i]["comboEffectTime"]*1e-5
        vector_feature.extend(main_skill_list)

    # 获取技能状态
    def get_enemy_skill_state(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_skill_state = hero['skill_state']["slot_states"]
        enemy_skill_list = [0 for i in range(70)]
        for skill_list, skill_state in zip([enemy_skill_list], [enemy_skill_state]):
            for i in range(7):
                skill_list[10*i] = skill_state[i]["configId"]*1e-5
                skill_list[10*i+1] = skill_state[i]["level"]
                skill_list[10*i+2] = 1 if skill_state[i]["usable"] else 0
                skill_list[10*i+3] = skill_state[i]["cooldown"]*1e-5
                skill_list[10*i+4] = skill_state[i]["cooldown_max"]*1e-5
                skill_list[10*i+5] = skill_state[i]["usedTimes"]*0.1
                skill_list[10*i+6] = skill_state[i]["hitHeroTimes"]*1e-2
                skill_list[10*i+7] = skill_state[i]["succUsedInFrame"]*1e-5
                skill_list[10*i+8] = skill_state[i]["nextConfigID"]*1e-5
                skill_list[10*i+9] = skill_state[i]["comboEffectTime"]*1e-5
        vector_feature.extend(enemy_skill_list)
        # print(f"英雄skill{main_skill_list}{enemy_skill_list}")

    # 获取装备状态
    def get_hero_equip_state(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_equip_state = hero["equip_state"]["equips"]
        main_equip_list = [0 for i in range(18)]
        for equip_list, equip_state in zip([main_equip_list], [main_equip_state]):
            for i in range(6):
                equip_list[3*i] = equip_state[i]["configId"]*1e-5
                equip_list[3*i+1] = equip_state[i]["amount"]
                equip_list[3*i+2] = equip_state[i]["buyPrice"]*3e-3
        vector_feature.extend(main_equip_list)

    def get_enemy_equip_state(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_equip_state = hero["equip_state"]["equips"]
        enemy_equip_list = [0 for i in range(18)]
        for equip_list, equip_state in zip([enemy_equip_list], [enemy_equip_state]):
            for i in range(6):
                equip_list[3*i] = equip_state[i]["configId"]*1e-5
                equip_list[3*i+1] = equip_state[i]["amount"]
                equip_list[3*i+2] = equip_state[i]["buyPrice"]*3e-3
        vector_feature.extend(enemy_equip_list)
        # print(f"英雄装备{main_equip_list}{enemy_equip_list}")

    # 获取被动技能状态
    def get_hero_passive_skill(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_passive_skill = hero["passive_skill"]
        main_passive_skill_list = [0 for i in range(6)]
        for passive_skill_list, passive_skill_state in zip([main_passive_skill_list], [main_passive_skill]):
            passive_skill_cnt = min(3, len(passive_skill_state))
            for i in range(passive_skill_cnt):
                passive_skill_list[2*i] = passive_skill_state[i]["passive_skillid"]*1e-5
                passive_skill_list[2*i+1] = passive_skill_state[i]["cooldown"]
        vector_feature.extend(main_passive_skill_list)

    def get_enemy_passive_skill(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_passive_skill = hero["passive_skill"]
        enemy_passive_skill_list = [0 for i in range(6)]
        for passive_skill_list, passive_skill_state in zip([enemy_passive_skill_list], [enemy_passive_skill]):
            passive_skill_cnt = min(3, len(passive_skill_state))
            for i in range(passive_skill_cnt):
                passive_skill_list[2*i] = passive_skill_state[i]["passive_skillid"]*1e-5
                passive_skill_list[2*i+1] = passive_skill_state[i]["cooldown"]

        vector_feature.extend(enemy_passive_skill_list)
        # print(f"英雄被动{main_passive_skill_list}{enemy_passive_skill_list}")

    # 获取是否再草丛内
    def get_hero_in_grass(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        main_in_grass = 1 if hero["isInGrass"] else 0
        vector_feature.append(main_in_grass)

    def get_enemy_in_grass(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_in_grass = 1 if hero["isInGrass"] else 0
        vector_feature.append(enemy_in_grass)
        # print(f"英雄在草丛{main_in_grass}{enemy_in_grass}")

    # 得到英雄的runtimeid
    def get_enemy_runtime_id(self, hero, vector_feature, feature_name):
        enemy_hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_run_time_id = enemy_hero["actor_state"]["runtime_id"]*1e-3
        vector_feature.append(enemy_run_time_id)

    # 得到英雄的runtimeid
    def get_hero_runtime_id(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        hero_run_time_id = hero["actor_state"]["runtime_id"]*1e-3
        vector_feature.append(hero_run_time_id)
    
    # 获取各个士兵的位置
    def get_main_soliders_feature(self, hero, vector_feature, feature_name):
        main_soliders_location = [0 for i in range(36)] 
        main_soldiers = list(self.main_camp_soldier_dict.values())
        main_soldiers_sorted = sorted(main_soldiers, key=lambda x: x.get("runtime_id", 0))
        soldier_count = min(6, len(main_soldiers_sorted))
        for i in range(soldier_count):
            soldier = main_soldiers_sorted[i]
            main_soliders_location[i*6] = soldier.get("config_id", 0) * 1e-3
            main_soliders_location[i*6+1] = soldier.get("runtime_id", 0) * 1e-3
            if self.transform_camp2_to_camp1:
                main_soliders_location[i*6+2] = -soldier.get("location", {}).get('x', 0) / 60000
                main_soliders_location[i*6+3] = -soldier.get("location", {}).get('z', 0) / 60000
            else:
                main_soliders_location[i*6+2] = soldier.get("location", {}).get('x', 0) / 60000
                main_soliders_location[i*6+3] = soldier.get("location", {}).get('z', 0) / 60000
            main_soliders_location[i*6+4] = soldier.get("hp", 0) / 12000
            main_soliders_location[i*6+5] = soldier.get("max_hp", 0) / 12000
        vector_feature.extend(main_soliders_location)

    # 获取各个士兵的位置
    def get_enemy_soliders_feature(self, hero, vector_feature, feature_name):
        main_soliders_location = [0 for i in range(36)] 
        main_soldiers = list(self.enemy_camp_soldier_dict.values())
        main_soldiers_sorted = sorted(main_soldiers, key=lambda x: x.get("runtime_id", 0))
        soldier_count = min(6, len(main_soldiers_sorted))
        for i in range(soldier_count):
            soldier = main_soldiers_sorted[i]
            main_soliders_location[i*6] = soldier.get("config_id", 0) * 1e-3
            main_soliders_location[i*6+1] = soldier.get("runtime_id", 0) * 1e-3
            if self.transform_camp2_to_camp1:
                main_soliders_location[i*6+2] = -soldier.get("location", {}).get('x', 0) / 60000
                main_soliders_location[i*6+3] = -soldier.get("location", {}).get('z', 0) / 60000
            else:
                main_soliders_location[i*6+2] = soldier.get("location", {}).get('x', 0) / 60000
                main_soliders_location[i*6+3] = soldier.get("location", {}).get('z', 0) / 60000
            main_soliders_location[i*6+4] = soldier.get("hp", 0) / 12000
            main_soliders_location[i*6+5] = soldier.get("max_hp", 0) / 12000
        vector_feature.extend(main_soliders_location)

    # 获取敌方坐标
    def get_enemy_location(self, hero, vector_feature, feature_name):
        enemy_hero = next(iter(self.enemy_camp_hero_dict.values()))
        if enemy_hero["actor_state"]['camp_visible'][1] and enemy_hero["actor_state"]['camp_visible'][0]  == True:
            self.enemy_location = [enemy_hero['actor_state']["location"]['x']/60000, enemy_hero['actor_state']["location"]['z']/60000]
            if self.transform_camp2_to_camp1:
                self.enemy_location = [0-i for i in self.enemy_location]
        if enemy_hero["actor_state"]['hp'] <= 0:
            self.enemy_location = [0,0]

        vector_feature.extend(self.enemy_location)

    # 获取敌方英雄hp
    def get_enemy_hp(self, hero, vector_feature, feature_name):
        enemy_hero = next(iter(self.enemy_camp_hero_dict.values()))
        if enemy_hero["actor_state"]['camp_visible'][1] and enemy_hero["actor_state"]['camp_visible'][0]  == True:
            self.enemy_hp = enemy_hero["actor_state"]['hp']
        if enemy_hero["actor_state"]['hp'] <= 0:
            self.enemy_hp = 0
        vector_feature.append(self.enemy_hp)

    # enemy_in_main_camp_tower_atk_range
    def is_enemy_in_main_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        tower = self.main_camp_organ_dict["tower"]
        tower_loc = tower["location"]
        atk_range = tower["attack_range"]
        hero_loc = next(iter(self.enemy_camp_hero_dict.values()))["actor_state"]["location"]

        dx = tower_loc["x"] - hero_loc["x"]
        dz = tower_loc["z"] - hero_loc["z"]
        dist_sq = math.sqrt(dx**2 + dz**2)

        if dist_sq <= atk_range:
            vector_feature.append(1.0)
        else:
            vector_feature.append(0.0)

    # enemy_in_enemy_camp_tower_atk_range
    def is_enemy_in_enemy_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        tower = self.enemy_camp_organ_dict["tower"]
        tower_loc = tower["location"]
        atk_range = tower["attack_range"]
        hero_loc = next(iter(self.enemy_camp_hero_dict.values()))["actor_state"]["location"]

        dx = tower_loc["x"] - hero_loc["x"]
        dz = tower_loc["z"] - hero_loc["z"]
        dist_sq = math.sqrt(dx**2 + dz**2)

        if dist_sq <= atk_range:
            vector_feature.append(1.0)
        else:
            vector_feature.append(0.0)

    # enemy_under_tower_atk
    def is_enemy_under_tower_atk(self, hero, vector_feature, feature_name):
        hero_id = next(iter(self.enemy_camp_hero_dict.values()))["actor_state"]["runtime_id"]
        tower = self.main_camp_organ_dict["tower"]
        target_id = tower['attack_target']

        if target_id == hero_id:
            vector_feature.append(1.0)
        else:
            vector_feature.append(0.0)

    # enemy_in_attack_range
    def is_enemy_in_attack_range(self, hero, vector_feature, feature_name):
        hero = next(iter(self.enemy_camp_hero_dict.values()))
        atk_range = hero["actor_state"]["attack_range"]
        hx = hero["actor_state"]["location"]["x"]
        hz = hero["actor_state"]["location"]["z"]
        if self.transform_camp2_to_camp1 and hx != 100000 and hz != 100000:
            hx, hz = -hx, -hz

        # 如果主英雄死亡或坐标无效，全部记作不在射程内
        if hero["actor_state"]["hp"] <= 0 or hx == 100000 or hz == 100000:
            for _ in self.enemy_camp_hero_dict:
                vector_feature.append(0.0)
            return

        for _, enemy in self.main_camp_hero_dict.items():
            if enemy["actor_state"]["hp"] <= 0:
                vector_feature.append(0.0)
                continue

            ex = enemy["actor_state"]["location"]["x"]
            ez = enemy["actor_state"]["location"]["z"]
            if ex == 100000 or ez == 100000:
                vector_feature.append(0.0)
                continue

            if self.transform_camp2_to_camp1:
                ex, ez = -ex, -ez

            dx, dz = ex - hx, ez - hz
            dist = math.sqrt(dx * dx + dz * dz)

            # 在射程内 -> 1.0，否则 0.0
            vector_feature.append(1.0 if dist <= atk_range else 0.0)

    # enemy_in_attack_range
    def is_hero_in_attack_range(self, hero, vector_feature, feature_name):
        hero = next(iter(self.main_camp_hero_dict.values()))
        atk_range = hero["actor_state"]["attack_range"]
        hx = hero["actor_state"]["location"]["x"]
        hz = hero["actor_state"]["location"]["z"]
        if self.transform_camp2_to_camp1 and hx != 100000 and hz != 100000:
            hx, hz = -hx, -hz

        # 如果主英雄死亡或坐标无效，全部记作不在射程内
        if hero["actor_state"]["hp"] <= 0 or hx == 100000 or hz == 100000:
            for _ in self.enemy_camp_hero_dict:
                vector_feature.append(0.0)
            return

        for _, enemy in self.enemy_camp_hero_dict.items():
            if enemy["actor_state"]["hp"] <= 0:
                vector_feature.append(0.0)
                continue

            ex = enemy["actor_state"]["location"]["x"]
            ez = enemy["actor_state"]["location"]["z"]
            if ex == 100000 or ez == 100000:
                vector_feature.append(0.0)
                continue

            if self.transform_camp2_to_camp1:
                ex, ez = -ex, -ez

            dx, dz = ex - hx, ez - hz
            dist = math.sqrt(dx * dx + dz * dz)

            # 在射程内 -> 1.0，否则 0.0
            vector_feature.append(1.0 if dist <= atk_range else 0.0)


    #英雄的攻击目标
    def get_hero_attack_target(self, hero, vector_feature, feature_name):
        main_hero = next(iter(self.main_camp_hero_dict.values()))
        main_attack_target = main_hero["actor_state"]["attack_target"]*1e-3
        vector_feature.append(main_attack_target)

    def get_enemy_attack_target(self, hero, vector_feature, feature_name):
        enemy_hero = next(iter(self.enemy_camp_hero_dict.values()))
        enemy_attack_target = enemy_hero["actor_state"]["attack_target"]*1e-3
        vector_feature.append(enemy_attack_target)
