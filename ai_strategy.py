"""AI策略模块：基于多时间窗统计的可解释自适应难度控制。"""

from __future__ import annotations

import math
import threading
import time
from typing import Any
import json

import requests


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class AIStrategyManager:
    """基于 20s / 2min / 10min 的专注度统计，输出稳定且可解释的策略。"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.focus_history: list[tuple[float, float]] = []  # (timestamp_sec, focus_0_3)
        self.current_strategy: dict[str, Any] = {
            "speed": 3,
            "obstacle_height": 1.5,
            "rhythm": "stable",
            "confidence": 0.6,
        }
        self.last_ai_update = time.time()
        self.ai_update_interval = 12.0
        self.last_ai_response = ""
        self.last_ai_report: dict[str, Any] = {
            "title": "等待数据",
            "summary": "正在收集专注度曲线，稍后会生成更科学的难度策略。",
            "reasoning": [],
            "metrics": {},
        }
        self.lock = threading.Lock()

    def add_focus_data(self, focus_level: float) -> None:
        with self.lock:
            now = time.time()
            self.focus_history.append((now, _clamp(float(focus_level), 0.0, 3.0)))
            # 保留最近 10 分钟，兼顾长期趋势和内存稳定
            cutoff = now - 600.0
            self.focus_history = [(t, f) for t, f in self.focus_history if t >= cutoff]

    @staticmethod
    def _slice_window(data: list[tuple[float, float]], now: float, sec: float) -> list[tuple[float, float]]:
        lo = now - sec
        return [(t, f) for t, f in data if t >= lo]

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        m = sum(values) / len(values)
        var = sum((x - m) ** 2 for x in values) / len(values)
        return m, math.sqrt(var)

    @staticmethod
    def _slope_per_sec(data: list[tuple[float, float]]) -> float:
        if len(data) < 3:
            return 0.0
        t0 = data[0][0]
        xs = [t - t0 for t, _ in data]
        ys = [f for _, f in data]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom <= 1e-9:
            return 0.0
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        return num / denom

    def _compute_strategy(self, data: list[tuple[float, float]]) -> tuple[dict[str, Any], dict[str, Any], str]:
        now = time.time()
        w_short = self._slice_window(data, now, 20.0)
        w_mid = self._slice_window(data, now, 120.0)
        w_long = self._slice_window(data, now, 600.0)

        v_short = [f for _, f in w_short]
        v_mid = [f for _, f in w_mid]
        v_long = [f for _, f in w_long]
        m_short, s_short = self._mean_std(v_short)
        m_mid, s_mid = self._mean_std(v_mid)
        m_long, _ = self._mean_std(v_long)
        slope_short = self._slope_per_sec(w_short)
        slope_mid = self._slope_per_sec(w_mid)
        slope_long = self._slope_per_sec(w_long)

        # 核心：短期反应 + 中期稳态 + 长期疲劳补偿
        weighted_focus = 0.50 * m_short + 0.35 * m_mid + 0.15 * m_long
        trend_term = _clamp(18.0 * slope_mid + 8.0 * slope_short, -0.40, 0.40)
        volatility_penalty = _clamp((s_short - 0.35) * 0.9, 0.0, 0.45)
        fatigue_penalty = 0.0
        if m_long < 1.3 and slope_long < -0.0015:
            fatigue_penalty = 0.25

        target_obstacle = _clamp(
            weighted_focus * 1.12 + trend_term - volatility_penalty - fatigue_penalty,
            0.20,
            3.00,
        )

        with self.lock:
            prev_ob = float(self.current_strategy.get("obstacle_height", 1.5))
            prev_sp = int(self.current_strategy.get("speed", 3))

        # 限制单次变化，避免策略大跳
        obstacle = prev_ob + _clamp(target_obstacle - prev_ob, -0.35, 0.35)
        obstacle = round(_clamp(obstacle, 0.0, 3.0), 2)
        target_speed = int(round(_clamp(1.0 + 4.0 * (obstacle / 3.0), 1.0, 5.0)))
        if target_speed > prev_sp + 1:
            speed = prev_sp + 1
        elif target_speed < prev_sp - 1:
            speed = prev_sp - 1
        else:
            speed = target_speed
        speed = int(_clamp(float(speed), 1.0, 5.0))

        if fatigue_penalty > 0.0:
            rhythm = "relaxing"
        elif trend_term > 0.15 and volatility_penalty < 0.12:
            rhythm = "rising"
        else:
            rhythm = "stable"

        confidence = _clamp(0.45 + min(len(v_long), 250) / 400.0 - s_short * 0.08, 0.30, 0.95)

        strategy: dict[str, Any] = {
            "speed": speed,
            "obstacle_height": obstacle,
            "rhythm": rhythm,
            "confidence": round(confidence, 2),
        }
        metrics = {
            "short_mean": round(m_short, 2),
            "mid_mean": round(m_mid, 2),
            "long_mean": round(m_long, 2),
            "short_std": round(s_short, 3),
            "slope_short": round(slope_short, 4),
            "slope_mid": round(slope_mid, 4),
            "slope_long": round(slope_long, 4),
            "fatigue_penalty": round(fatigue_penalty, 2),
        }
        explanation = self._build_human_explanation(strategy, metrics, len(v_long))
        report = self._build_report(strategy, metrics, len(v_long))
        return strategy, report, explanation

    @staticmethod
    def _build_human_explanation(strategy: dict[str, Any], metrics: dict[str, Any], sample_count: int) -> str:
        rhythm_map = {"stable": "平稳节奏", "rising": "递增节奏", "relaxing": "放松节奏"}
        rhythm_text = rhythm_map.get(str(strategy["rhythm"]), str(strategy["rhythm"]))
        return (
            f"我正在用 20秒/2分钟/10分钟 三层曲线联合调参。当前建议："
            f"速度 {strategy['speed']}，障碍高度 {strategy['obstacle_height']:.2f}（0-3），节奏 {rhythm_text}。"
            f" 关键依据：短期均值 {metrics['short_mean']:.2f}，中期均值 {metrics['mid_mean']:.2f}，"
            f"长期均值 {metrics['long_mean']:.2f}，短期波动 {metrics['short_std']:.3f}，"
            f"中期趋势斜率 {metrics['slope_mid']:.4f}。当前累计样本 {sample_count}。"
        )

    @staticmethod
    def _build_report(strategy: dict[str, Any], metrics: dict[str, Any], sample_count: int) -> dict[str, Any]:
        rhythm_map = {"stable": "平稳节奏", "rising": "递增节奏", "relaxing": "放松节奏"}
        rhythm_text = rhythm_map.get(str(strategy["rhythm"]), str(strategy["rhythm"]))
        reasons = [
            f"短期(20s)/中期(2min)/长期(10min)均值：{metrics['short_mean']:.2f} / {metrics['mid_mean']:.2f} / {metrics['long_mean']:.2f}",
            f"短期波动(标准差)：{metrics['short_std']:.3f}，波动越大则加难越保守。",
            f"趋势斜率(短/中/长)：{metrics['slope_short']:.4f} / {metrics['slope_mid']:.4f} / {metrics['slope_long']:.4f}",
        ]
        if metrics["fatigue_penalty"] > 0:
            reasons.append("检测到长期下行且总体偏低，触发疲劳保护，适当降压。")
        else:
            reasons.append("未触发疲劳保护，按稳定增益策略更新难度。")
        return {
            "title": "AI 难度控制报告",
            "summary": (
                f"策略输出：速度 {strategy['speed']}，障碍高度 {strategy['obstacle_height']:.2f}/3，"
                f"节奏 {rhythm_text}，置信度 {strategy['confidence']:.2f}。"
            ),
            "reasoning": reasons,
            "metrics": {
                "sample_count": sample_count,
                "short_mean": metrics["short_mean"],
                "mid_mean": metrics["mid_mean"],
                "long_mean": metrics["long_mean"],
                "short_std": metrics["short_std"],
            },
        }

    def update_strategy_if_needed(self) -> bool:
        current_time = time.time()
        if current_time - self.last_ai_update < self.ai_update_interval:
            return False

        with self.lock:
            data = self.focus_history.copy()
            if len(data) < 8:
                self.last_ai_response = "数据仍在积累（至少需要 8 个点）"
                self.last_ai_report = {
                    "title": "AI 难度控制报告",
                    "summary": "数据仍在积累，先保持当前难度，避免误判。",
                    "reasoning": [f"当前样本数 {len(data)}，建议至少 8 个样本后再进行稳定调参。"],
                    "metrics": {"sample_count": len(data)},
                }
                return False

        strategy, report, explanation = self._compute_strategy(data)
        with self.lock:
            self.current_strategy = strategy
            self.last_ai_update = current_time
            self.last_ai_report = report
            self.last_ai_response = explanation
        return True

    def get_current_strategy(self) -> dict[str, Any]:
        with self.lock:
            return self.current_strategy.copy()

    def get_last_ai_response(self) -> str:
        with self.lock:
            return self.last_ai_response if self.last_ai_response else "暂无AI回复"

    def get_last_ai_report(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.last_ai_report)

    def _build_local_game_summary(
        self,
        *,
        end_reason: str,
        duration_sec: float,
        target_duration_sec: float,
        mercy_count: int,
        values: list[float],
    ) -> dict[str, Any]:
        if not values:
            return {
                "title": "AI 专注度总结",
                "summary": "本局可用数据较少，建议保持正视屏幕并稳定采样后再进行评估。",
                "bullets": [
                    f"结束原因：{end_reason}",
                    "专注数据不足，无法形成稳定趋势结论。",
                    "建议：下局将目标时长缩短到 3-5 分钟，先建立稳定专注区间。",
                ],
            }
        avg = sum(values) / len(values)
        vmin = min(values)
        vmax = max(values)
        focused_ratio = sum(1 for v in values if v >= 1.5) / len(values)
        high_ratio = sum(1 for v in values if v >= 2.3) / len(values)
        stability = sum((v - avg) ** 2 for v in values) / len(values)
        mm = int(duration_sec // 60)
        ss = int(duration_sec % 60)
        target_min = int(round(target_duration_sec / 60.0))
        return {
            "title": "AI 专注度总结",
            "summary": (
                f"本局平均专注度 {avg:.2f}/3，整体处于"
                f"{'较稳定' if stability < 0.35 else '波动偏大'}状态。"
            ),
            "bullets": [
                f"结束原因：{end_reason}",
                f"本局时长：{mm:02d}:{ss:02d}（目标 {target_min} 分钟）",
                f"专注范围：最低 {vmin:.2f} / 最高 {vmax:.2f} / 平均 {avg:.2f}",
                f"中高专注占比：{focused_ratio * 100:.1f}%；高专注占比：{high_ratio * 100:.1f}%",
                f"AI 仁慈机制触发：{mercy_count} 次",
                "建议：采用 50 分钟专注 + 10 分钟休息节奏，并减少频繁低头动作。",
            ],
        }

    def generate_game_summary(
        self,
        *,
        end_reason: str,
        duration_sec: float,
        target_duration_sec: float,
        mercy_count: int,
        focus_values: list[float],
    ) -> dict[str, Any]:
        values = [_clamp(float(v), 0.0, 3.0) for v in focus_values if v is not None]
        local_summary = self._build_local_game_summary(
            end_reason=end_reason,
            duration_sec=duration_sec,
            target_duration_sec=target_duration_sec,
            mercy_count=mercy_count,
            values=values,
        )

        # 若可用，使用大模型把总结润色成更自然的“人话”。
        if not self.api_key:
            return local_summary
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "你是专注训练游戏的AI教练。请根据给定统计生成总结，"
                            "返回严格JSON格式："
                            '{"title":"...","summary":"...","bullets":["...","...","...","...","..."]}。'
                            "语言要求：中文、简洁、鼓励式、可执行建议。"
                            f"统计数据：{json.dumps(local_summary, ensure_ascii=False)}"
                        ),
                    }
                ],
                "temperature": 0.6,
                "max_tokens": 260,
            }
            response = requests.post(url, headers=headers, json=payload, timeout=12)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict) and "summary" in parsed:
                parsed.setdefault("title", "AI 专注度总结")
                bullets = parsed.get("bullets")
                if not isinstance(bullets, list):
                    parsed["bullets"] = local_summary["bullets"]
                return parsed
        except Exception:
            pass
        return local_summary
