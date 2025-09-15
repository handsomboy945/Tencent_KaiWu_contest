#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math
from collections import OrderedDict


class GameStateProcess:
    def __init__(self, player_camp):
        self.normalizer = FeatureNormalizer()
        self.player_camp = player_camp  # "PLAYERCAMP_1" or "PLAYERCAMP_2"
        self.get_gamestate_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.gamestate_feature_config)
        self.m_each_level_max_exp = {}
        self.init_max_exp_of_each_hero()

    def get_gamestate_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "gamestate_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        self.gamestate_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.gamestate_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_gamestate(self, frame_state):
        vector_feature = []
        
        # Generate each specific feature through feature_func_map
        for feature_name, feature_func in self.feature_func_map.items():
            value = []
            self.feature_func_map[feature_name](frame_state, value)
            # Normalize the specific features
            if feature_name not in self.map_feature_to_norm:
                raise ValueError(f"Normalization method for feature '{feature_name}' not found")
            
            for k in value:
                norm_func, *params = self.map_feature_to_norm[feature_name]
                normalized_value = norm_func(k, *params)
                if isinstance(normalized_value, list):
                    vector_feature.extend(normalized_value)
                else:
                    vector_feature.append(normalized_value)
                    
        return vector_feature

    def get_game_time(self, frame_state, vector_feature):
        # Calculate game time in seconds (assuming 30 frames per second)
        value = frame_state["frameNo"] / 30.0
        vector_feature.append(value)

    def get_frame_count(self, frame_state, vector_feature):
        value = frame_state["frameNo"]
        vector_feature.append(value)

    def get_ally_gold_total(self, frame_state, vector_feature):
        total_gold = 0
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.player_camp:
                total_gold += hero.get("moneyCnt", 0)
        vector_feature.append(total_gold)

    def get_enemy_gold_total(self, frame_state, vector_feature):
        total_gold = 0
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] != self.player_camp:
                total_gold += hero.get("moneyCnt", 0)
        vector_feature.append(total_gold)

    # 用智能体等级和当前经验值，计算获得经验值的总量
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def get_ally_exp_total(self, frame_state, vector_feature):
        total_exp = 0
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.player_camp:
                exp_sum = 0.0
                for i in range(1, hero["level"]):
                    exp_sum += self.m_each_level_max_exp[i]
                exp_sum += hero["exp"]
        vector_feature.append(exp_sum)

    def get_enemy_exp_total(self, frame_state, vector_feature):
        total_exp = 0
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] != self.player_camp:
                exp_sum = 0.0
                for i in range(1, hero["level"]):
                    exp_sum += self.m_each_level_max_exp[i]
                exp_sum += hero["exp"]
        vector_feature.append(exp_sum)

    def get_ally_tower_count(self, frame_state, vector_feature):
        count = 0
        for npc in frame_state["npc_states"]:
            if (npc["sub_type"] == "ACTOR_SUB_TOWER" and 
                npc["camp"] == self.player_camp and 
                npc["hp"] > 0):
                count += 1
        vector_feature.append(count)

    def get_enemy_tower_count(self, frame_state, vector_feature):
        count = 0
        for npc in frame_state["npc_states"]:
            if (npc["sub_type"] == "ACTOR_SUB_TOWER" and 
                npc["camp"] != self.player_camp and 
                npc["hp"] > 0):
                count += 1
        vector_feature.append(count)

    def get_ally_crystal_hp(self, frame_state, vector_feature):
        for npc in frame_state["npc_states"]:
            if (npc["sub_type"] == "ACTOR_SUB_TOWER" and 
                npc["camp"] == self.player_camp):
                vector_feature.append(npc["hp"])
                return
        vector_feature.append(0)  # If crystal not found

    def get_enemy_crystal_hp(self, frame_state, vector_feature):
        for npc in frame_state["npc_states"]:
            if (npc["sub_type"] == "ACTOR_SUB_TOWER" and 
                npc["camp"] != self.player_camp):
                vector_feature.append(npc["hp"])
                return
        vector_feature.append(0)  # If crystal not found

    def get_ally_kill_count(self, frame_state, vector_feature):
        total_kills = 0
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.player_camp:
                total_kills += hero.get("killCnt", 0)
        vector_feature.append(total_kills)

    def get_enemy_kill_count(self, frame_state, vector_feature):
        total_kills = 0
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] != self.player_camp:
                total_kills += hero.get("killCnt", 0)
        vector_feature.append(total_kills)

    def get_game_phase(self, frame_state, vector_feature):
        # Determine game phase based on time and other factors
        game_time = frame_state["frameNo"] / 30.0  # Convert frames to seconds
        
        if game_time < 300:  # First 5 minutes
            phase = 1  # Early game
        elif game_time < 900:  # 5-15 minutes
            phase = 2  # Mid game
        else:
            phase = 3  # Late game
            
        vector_feature.append(phase)