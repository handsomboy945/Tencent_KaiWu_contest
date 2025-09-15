#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from agent_ppo.conf.conf import GameConfig


# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1

        # 三套“逐帧统计量”map（不等于最终奖励值）
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()

        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}

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

    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    def result(self, frame_data):
        # 每帧入口
        frame_no = frame_data["frameNo"]
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        # 时间退火（统一乘系数，不破坏各子项与总和的一致性）
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # 计算每帧的每个奖励子项的“统计量”
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        # 1) 获取双方英雄（主视角为 camp）
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        # 主英雄常用字段
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # 2) 获取双方防御塔 / 水晶
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":
                    enemy_spring = organ

        # 3) 写入每个子项的“当前帧统计量”（cur_frame_value），并迁移 last
        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero["moneyCnt"]

            elif reward_name == "hp_point":
                # 双重开方：低血更敏感，高血压扁
                if main_hero_max_hp > 0:
                    reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
                else:
                    reward_struct.cur_frame_value = 0.0

            elif reward_name == "ep_rate":
                # 蓝量百分比；死亡/无蓝上限时记 0
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0.0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)

            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]

            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]

            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]

            elif reward_name == "last_hit":
                # 事件型：本帧+1/-1，鼓励我方补兵，抑制敌方补我方兵
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data.get("frame_action", {})
                if "dead_action" in frame_action and enemy_hero is not None:
                    dead_actions = frame_action["dead_action"]
                    my_id = main_hero["actor_state"]["runtime_id"]
                    enemy_id = enemy_hero["actor_state"]["runtime_id"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == my_id
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_id
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0

            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)

            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

            else:
                # 未配置的键不做处理（保持 0）
                reward_struct.cur_frame_value = reward_struct.cur_frame_value

    # 用智能体到双方防御塔的距离，计算前进奖励（即时值，非差分）
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0.0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        # 满血但英雄位置在“两塔距离之外”（更靠后）时，给惩罚（负值）
        if main_hero["actor_state"]["hp"] / max(1, main_hero["actor_state"]["max_hp"]) > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # 用帧数据来计算两边的奖励子项信息（逐帧统计量）
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]

        # 我方/敌方各算一份统计量
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # 用每一帧得到的奖励子项统计量来计算对应的“即时奖励值”
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0

        # 为 exp/hp_point/ep_rate/last_hit/forward 做到与“老版本语义”一致
        # 其余 money/kill/death/tower_hp_point 使用“零和+差分”通用分支
        # ---------------------------------------------------------------
        # 先找到主英雄（用于 exp 满级判断）
        main_hero = None
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_hero = hero
                break

        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():

            if reward_name == "hp_point":
                # 4 情况健壮处理（避免复活/死亡的 0 值带来错误差分）
                m_last = self.m_main_calc_frame_map[reward_name].last_frame_value
                e_last = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                m_cur  = self.m_main_calc_frame_map[reward_name].cur_frame_value
                e_cur  = self.m_enemy_calc_frame_map[reward_name].cur_frame_value

                if m_last == 0.0 and e_last == 0.0:
                    reward_struct.last_frame_value = 0.0
                    reward_struct.cur_frame_value  = 0.0
                elif m_last == 0.0:
                    reward_struct.last_frame_value = -e_last
                    reward_struct.cur_frame_value  = -e_cur
                elif e_last == 0.0:
                    reward_struct.last_frame_value = m_last
                    reward_struct.cur_frame_value  = m_cur
                else:
                    reward_struct.last_frame_value = m_last - e_last
                    reward_struct.cur_frame_value  = m_cur  - e_cur

                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            elif reward_name == "ep_rate":
                # 只看我方；仅当上帧>0时记差分（鼓励蓝量回升，不惩罚放技能）
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.cur_frame_value  = self.m_main_calc_frame_map[reward_name].cur_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0.0

            elif reward_name == "exp":
                # 满级（>=15）后不再给经验奖励
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0.0
                else:
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            elif reward_name == "last_hit":
                # 事件通道：直接用本帧统计值（正负皆可）
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "forward":
                # 即时值：直接取本帧（通常为非正，配合权重体现惩罚/奖励）
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            else:
                # money / kill / death / tower_hp_point 等：零和 + 差分
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # 线性加权汇总
            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value

        reward_dict["reward_sum"] = reward_sum
