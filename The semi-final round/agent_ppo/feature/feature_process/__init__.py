#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.enemy_process import EnemyProcess
from agent_ppo.feature.feature_process.friend_soilder_feature_process import FriendlySoilderProcess
from agent_ppo.feature.feature_process.enemy_soilder_feature_process import EnemySoilderProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.image_process import ImageProcess
from agent_ppo.feature.feature_process.gamestate_process import GameStateProcess
from agent_ppo.feature.feature_process.monster_process import MonsterProcess



class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.enemy_process = EnemyProcess(camp)
        self.friend_soilder_process = FriendlySoilderProcess(camp)
        self.enemy_soilder_process = EnemySoilderProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.image_process = ImageProcess(camp)
        self.gamestate_process = GameStateProcess(camp)
        self.monster_process = MonsterProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.enemy_process = EnemyProcess(camp)
        self.friend_soilder_process = FriendlySoilderProcess(camp)
        self.enemy_soilder_process = EnemySoilderProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.image_process = ImageProcess(camp)
        self.gamestate_process = GameStateProcess(camp)
        self.monster_process = MonsterProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_enemy_feature(self, frame_state):
        return self.enemy_process.process_vec_hero(frame_state)

    def process_friend_soilder_feature(self, frame_state):
        return self.friend_soilder_process.process_vec_hero(frame_state)

    def process_enemy_soilder_feature(self, frame_state):
        return self.enemy_soilder_process.process_vec_hero(frame_state)

    def process_image_feature(self, frame_state):
        return self.image_process.process_vec_image(frame_state)

    def process_gamestate_feature(self, frame_state):
        return self.gamestate_process.process_vec_gamestate(frame_state)

    def process_monster_feature(self, frame_state):
        return self.monster_process.process_vec_organ(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        hero_feature = self.process_hero_feature(frame_state)
        enemy_feature = self.process_enemy_feature(frame_state)
        friend_soilder_feature = self.process_friend_soilder_feature(frame_state)
        enemy_soilder_feature = self.process_enemy_soilder_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        monster_feature = self.process_monster_feature(frame_state)
        unit_feature = hero_feature + enemy_feature + friend_soilder_feature + enemy_soilder_feature + organ_feature + monster_feature
        
        game_state_feature = self.process_gamestate_feature(frame_state)
        feature = unit_feature + game_state_feature 

        return feature
