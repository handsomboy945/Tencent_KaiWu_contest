#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import time
import random
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
)
from kaiwu_agent.utils.common_func import attached
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.learn(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.predict_success_count = 0
        self.load_model_success_count = 0
        self.last_report_monitor_time = 0

        # ================= Round-robin 轮换赛程配置 =================
        # 三个英雄的ID：169:后羿，173:李元芳，174:虞姬
        self.hero_ids = [169, 173, 174]

        # 每个红方英雄对战的局数（例如 100 局）
        self.red_block_games = 10

        # 蓝方顺序（每完成 3*red_block_games 局后切换到下一位蓝方英雄）
        self.blue_order = self.hero_ids[:]  # [169, 173, 174]

        # 红方顺序（在一个大块内，红方每 red_block_games 局切换一次，依次三位）
        self.red_order = self.hero_ids[:]   # [169, 173, 174]

        # 调度指针：只在训练对局推进（评估对局不影响训练计划）
        self.match_index = 0

        # 是否启用轮换；如需从 toml 读取，可自行扩展 EnvConfManager
        self.enable_round_robin = True
        # ==========================================================

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config()

            # === 轮换赛程：仅训练局启用，覆盖 usr_conf 的英雄ID ===
            if self.enable_round_robin and not is_eval:
                blue_hero, red_hero = self._current_blue_red_heroes()
                usr_conf = self._override_usr_conf_heroes(usr_conf, blue_hero, red_hero)
                self.logger.info(
                    f"[RR] Episode plan -> BLUE:{blue_hero} vs RED:{red_hero} (match_index={self.match_index})"
                )

            # Start a new environment
            # 启动新对局，返回初始环境状态
            observation, extra_info = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if self._handle_disaster_recovery(extra_info):
                break

            # Reset agents
            # 重置智能体
            self.reset_agents(observation)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[i]["frame_state"])
                    observation[i]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[index])
                        else:
                            actions[index] = agent.exploit(observation[index])
                        self.predict_success_count += 1

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[index])
                            frame_collector.save_frame(frame, agent_id=index)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                frame_no, observation, terminated, truncated, extra_info = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if self._handle_disaster_recovery(extra_info):
                    break

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[i]["frame_state"])
                        observation[i]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]

                now = time.time()
                if now - self.last_report_monitor_time >= 60:
                    monitor_data = {
                        "actor_predict_succ_cnt": self.predict_success_count,
                        "actor_load_last_model_succ_cnt": self.load_model_success_count,
                    }
                    self.monitor.put_data({os.getpid(): monitor_data})
                    self.last_report_monitor_time = now

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[i]["reward"]["reward_sum"],
                            )

                    monitor_data = {}
                    if self.monitor and is_eval:
                        monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                        self.monitor.put_data({os.getpid(): monitor_data})

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples

                    # === 仅训练局推进轮换调度指针（评估对局不影响训练计划） ===
                    if self.enable_round_robin and not is_eval:
                        self.match_index += 1
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                    self.load_model_success_count += 1
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[i])

    def _handle_disaster_recovery(self, extra_info):
        # Handle disaster recovery logic
        # 处理容灾逻辑
        result_code = extra_info.get("result_code", 0)
        result_message = extra_info.get("result_message", "")
        if result_code < 0:
            self.logger.error(f"Env run error, result_code: {result_code}, result_message: {result_message}")
            raise RuntimeError(result_message)
        elif result_code > 0:
            return True
        return False

    # ===================== 轮换赛程：辅助方法 =====================

    def _current_blue_red_heroes(self):
        """
        根据 self.match_index 计算当前蓝/红方 hero_id。
        规则：
          - 每个大块 block_size = 3 * red_block_games 局，蓝方换一次英雄
          - 在一个大块内，红方每 red_block_games 局换一次英雄，依次三位
        """
        block_size = 3 * self.red_block_games
        block_id = self.match_index // block_size
        offset_in_block = self.match_index % block_size

        blue_idx = block_id % len(self.blue_order)
        red_idx = (offset_in_block // self.red_block_games) % len(self.red_order)

        blue_hero = self.blue_order[blue_idx]
        red_hero = self.red_order[red_idx]
        return blue_hero, red_hero

    def _override_usr_conf_heroes(self, usr_conf, blue_hero, red_hero):
        """
        覆盖 update_config() 返回的 usr_conf 里的阵容英雄ID。
        结构假定：
          usr_conf["lineups"]["blue_camp"] / ["red_camp"] 是列表，每个元素包含 {"hero_id": int}
        这里只保留 1v1：将第一个条目的 hero_id 替换，其余裁剪掉。
        """
        try:
            # 蓝方
            if "lineups" in usr_conf and "blue_camp" in usr_conf["lineups"]:
                if len(usr_conf["lineups"]["blue_camp"]) == 0:
                    usr_conf["lineups"]["blue_camp"].append({"hero_id": blue_hero})
                else:
                    usr_conf["lineups"]["blue_camp"][0]["hero_id"] = blue_hero
                usr_conf["lineups"]["blue_camp"] = usr_conf["lineups"]["blue_camp"][:1]

            # 红方
            if "lineups" in usr_conf and "red_camp" in usr_conf["lineups"]:
                if len(usr_conf["lineups"]["red_camp"]) == 0:
                    usr_conf["lineups"]["red_camp"].append({"hero_id": red_hero})
                else:
                    usr_conf["lineups"]["red_camp"][0]["hero_id"] = red_hero
                usr_conf["lineups"]["red_camp"] = usr_conf["lineups"]["red_camp"][:1]

        except Exception as e:
            self.logger.error(f"Override usr_conf heroes failed: {e}, usr_conf={usr_conf}")
        return usr_conf
