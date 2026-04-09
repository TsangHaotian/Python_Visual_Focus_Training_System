(function () {
  const FOCUS_COLORS = ["#ff4444", "#ffcc00", "#88dd88", "#228822"];
  const CHART_SEC = 120;

  const gameCanvas = document.getElementById("game-canvas");
  const chartCanvas = document.getElementById("chart-canvas");
  const focusBadge = document.getElementById("focus-badge");
  const aiLine = document.getElementById("ai-line");
  const statusLine = document.getElementById("status-line");
  const aiResponse = document.getElementById("ai-response");
  const summaryBox = document.getElementById("game-summary");
  const durationInput = document.getElementById("duration-minutes");
  const applyDurationBtn = document.getElementById("apply-duration-btn");
  const mercyModal = document.getElementById("mercy-modal");
  const mercyModalText = document.getElementById("mercy-modal-text");
  const mercyModalClose = document.getElementById("mercy-modal-close");

  const gctx = gameCanvas.getContext("2d");
  const cctx = chartCanvas.getContext("2d");

  function resizeChart() {
    const rect = chartCanvas.parentElement.getBoundingClientRect();
    chartCanvas.width = Math.max(400, Math.floor(rect.width - 8));
  }
  window.addEventListener("resize", resizeChart);
  resizeChart();

  /** 整体节奏：越小越慢（前进与系统缩放） */
  const PACE = 0.38;
  /** 专注度 0–3 与障碍高度等级 0–3 共用同一像素标尺（相对地面） */
  const MAX_FOCUS_LIFT = 72;
  function liftForFocusLevel(level03) {
    const f = Math.max(0, Math.min(3, level03));
    return (f / 3) * MAX_FOCUS_LIFT;
  }
  function pixelsFromObstacleLevel(level03) {
    const L = Math.max(0, Math.min(3, level03));
    return (L / 3) * MAX_FOCUS_LIFT;
  }
  function clamp03(x) {
    if (x == null || x === undefined || Number.isNaN(Number(x))) return null;
    return Math.max(0, Math.min(3, Number(x)));
  }

  const game = {
    w: 480,
    h: 220,
    groundY: 184,
    playerX: 72,
    playerR: 14,
    playerY: 0,
    vy: 0,
    obstacles: [],
    spawnTimer: 0,
    speed: 1.1,
    obsH: 36,
    gameStart: performance.now() / 1000,
    stalled: false,
  };
  game.playerY = game.groundY - game.playerR;

  const history = [];
  const sessionHistory = [];
  let focusTarget = 0;
  let focusDisplay = 0;
  let lastHistoryPushSec = 0;
  let lastAiResponseText = "";
  let sessionDurationSec = 5 * 60;
  let unfocusedAccumSec = 0;
  let mercyCount = 0;
  const maxMercyCount = 3;
  const mercyTriggerSec = 14;
  let gameEnded = false;
  let gameEndReason = "";

  function escapeHtml(text) {
    return String(text ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function resetGame() {
    game.playerY = game.groundY - game.playerR;
    game.vy = 0;
    game.obstacles = [];
    game.spawnTimer = 0;
    game.gameStart = performance.now() / 1000;
    game.stalled = false;
    unfocusedAccumSec = 0;
    mercyCount = 0;
    gameEnded = false;
    gameEndReason = "";
    sessionHistory.length = 0;
    hideMercyModal();
    summaryBox.innerHTML = '<div class="ai-response-placeholder">游戏进行中，结束后自动生成总结...</div>';
  }

  function showMercyModal(msg) {
    mercyModalText.textContent = msg;
    mercyModal.classList.remove("hidden");
  }

  function hideMercyModal() {
    mercyModal.classList.add("hidden");
  }

  function triggerMercy() {
    mercyCount += 1;
    unfocusedAccumSec = 0;
    const skipCount = Math.min(3, game.obstacles.length);
    if (skipCount > 0) {
      game.obstacles.splice(0, skipCount);
    }
    game.spawnTimer = Math.max(game.spawnTimer, 0.55 / PACE);
    const msg =
      `检测到你长时间未专注，AI 已帮你跳过 ${skipCount} 个障碍。` +
      `\n仁慈次数：${mercyCount}/${maxMercyCount}。请继续保持专注！`;
    showMercyModal(msg);
  }

  function buildSessionSummaryFallback(nowSec) {
    const values = sessionHistory.map(([, v]) => Number(v)).filter((x) => Number.isFinite(x));
    const elapsed = Math.max(0, nowSec - game.gameStart);
    if (!values.length) {
      return "本局缺少有效专注度数据，建议下次保持摄像头稳定并正视屏幕。";
    }
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const focusedRatio = values.filter((v) => v >= 1.5).length / values.length;
    const highRatio = values.filter((v) => v >= 2.3).length / values.length;
    const mm = Math.floor(elapsed / 60);
    const ss = Math.floor(elapsed % 60);
    return [
      `<div><strong>结束原因：</strong>${escapeHtml(gameEndReason || "完成目标时长")}</div>`,
      `<div><strong>本局时长：</strong>${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}（目标 ${Math.round(sessionDurationSec / 60)} 分钟）</div>`,
      `<div><strong>核心指标：</strong>平均专注 ${avg.toFixed(2)} / 3，最低 ${min.toFixed(2)}，最高 ${max.toFixed(2)}</div>`,
      `<div><strong>行为表现：</strong>中高专注(>=1.5) 占比 ${(focusedRatio * 100).toFixed(1)}%，高专注(>=2.3) 占比 ${(highRatio * 100).toFixed(1)}%</div>`,
      `<div><strong>AI仁慈机制：</strong>触发 ${mercyCount} 次（上限 ${maxMercyCount}）</div>`,
      "<ul><li>建议1：每 60-90 秒做一次微调坐姿，减少低头和侧脸。</li><li>建议2：若连续分心，先停 10 秒深呼吸再继续。</li><li>建议3：下局将目标设置为略短时长，先把稳定性练起来。</li></ul>",
    ].join("");
  }

  function renderAiSummary(summary, fallbackHtml) {
    if (!summary || typeof summary !== "object") {
      summaryBox.innerHTML = fallbackHtml;
      return;
    }
    const title = escapeHtml(summary.title || "AI 专注度总结");
    const summaryText = escapeHtml(summary.summary || "");
    const bullets = Array.isArray(summary.bullets) ? summary.bullets : [];
    const bulletHtml = bullets
      .slice(0, 8)
      .map((x) => `<li>${escapeHtml(x)}</li>`)
      .join("");
    summaryBox.innerHTML = `
      <div class="ai-response-timestamp">${title}</div>
      <div class="ai-response-content"><strong>结论：</strong>${summaryText}</div>
      ${bulletHtml ? `<div class="ai-response-content"><strong>要点：</strong><ul>${bulletHtml}</ul></div>` : ""}
    `;
  }

  async function finishGame(reason, nowSec) {
    if (gameEnded) return;
    gameEnded = true;
    gameEndReason = reason;
    const fallbackHtml = buildSessionSummaryFallback(nowSec);
    summaryBox.innerHTML = '<div class="ai-response-placeholder">AI 正在生成本局总结...</div>';
    try {
      const durationSec = Math.max(0, nowSec - game.gameStart);
      const focusValues = sessionHistory.map(([, v]) => Number(v)).filter((x) => Number.isFinite(x));
      const resp = await fetch("/api/game_summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          end_reason: reason,
          duration_sec: durationSec,
          target_duration_sec: sessionDurationSec,
          mercy_count: mercyCount,
          focus_values: focusValues,
        }),
      });
      const data = await resp.json();
      renderAiSummary(data.summary, fallbackHtml);
    } catch (e) {
      console.warn("AI summary failed, fallback used", e);
      summaryBox.innerHTML = fallbackHtml;
    }
    showMercyModal(`游戏结束：${reason}\n已在页面底部生成本局专注度总结。`);
  }

  /** speed：1–5；obstacleLevel：障碍高度等级 0–3，与曲线 / 小球刻度同源，像素 = (L/3)*MAX_FOCUS_LIFT */
  function applyDifficulty(speed, obstacleLevel03) {
    const base = Math.max(0.5, Number(speed) * 0.42);
    game.speed = Math.max(0.45, base * PACE);
    const L = clamp03(obstacleLevel03);
    if (L !== null) game.obsH = pixelsFromObstacleLevel(L);
  }

  function gameTick(dt, nowSec, cls, face) {
    if (gameEnded) return;
    const px = game.playerX,
      pr = game.playerR;

    let clsN = 0;
    if (face && cls != null && cls !== undefined && cls >= 0 && cls <= 3) {
      clsN = cls;
    }

    const baseLift = liftForFocusLevel(clsN);
    const targetY = game.groundY - pr - baseLift;

    const spring = 1.55;
    const damp = 0.68;
    const ay = (targetY - game.playerY) * spring - game.vy * damp;
    game.vy += ay * dt;
    game.playerY += game.vy * dt * 60;
    const maxVy = 2.4;
    if (game.vy > maxVy) game.vy = maxVy;
    if (game.vy < -maxVy) game.vy = -maxVy;

    if (game.playerY + pr > game.groundY) {
      game.playerY = game.groundY - pr;
      if (game.vy > 0) game.vy = 0;
    }

    const py = game.playerY;
    let stalled = false;
    for (const o of game.obstacles) {
      const top = game.groundY - o.h;
      const horiz = px + pr > o.x + 1 && px - pr < o.x + o.w - 1;
      const clears = py + pr <= top + 1;
      if (horiz && !clears) {
        stalled = true;
        break;
      }
    }
    game.stalled = stalled;

    const move = stalled ? 0 : game.speed * dt * 60;
    for (const o of game.obstacles) {
      o.x -= move;
      const hf = o.hitFlash || 0;
      if (hf > 0) o.hitFlash = Math.max(0, hf - dt);
      else o.hitFlash = 0;
    }
    game.obstacles = game.obstacles.filter((o) => o.x + o.w > 0);
    if (!stalled) game.spawnTimer -= dt;
    if (game.spawnTimer <= 0) {
      game.obstacles.push({ x: game.w + 20, w: 22, h: game.obsH, hitFlash: 0 });
      game.spawnTimer =
        1.85 / PACE + (10 - Math.min(game.speed * 2.2, 10)) * (0.35 / PACE);
    }

    for (const o of game.obstacles) {
      const top = game.groundY - o.h;
      const horiz = px + pr > o.x + 2 && px - pr < o.x + o.w - 2;
      const clearsOver = py + pr <= top + 1;
      if (clearsOver || !horiz || stalled) continue;
      const ballBottom = py + pr;
      if (ballBottom > top && py - pr < game.groundY) {
        o.hitFlash = 0.15;
        game.vy = Math.max(game.vy, 2.2);
      }
    }
  }

  // 用 Canvas 矢量绘制“朝右老鹰”，避免 emoji 在不同平台朝向不一致
  function drawEagleRight(ctx, x, y, size) {
    const s = Math.max(12, size);
    ctx.save();
    ctx.translate(x, y);

    // 翅膀/身体（朝右）
    ctx.fillStyle = "#8b5e3c";
    ctx.beginPath();
    ctx.moveTo(-0.6 * s, 0.0 * s);
    ctx.quadraticCurveTo(-0.2 * s, -0.75 * s, 0.25 * s, -0.08 * s);
    ctx.quadraticCurveTo(0.55 * s, 0.05 * s, 0.1 * s, 0.32 * s);
    ctx.quadraticCurveTo(-0.25 * s, 0.48 * s, -0.6 * s, 0.0 * s);
    ctx.fill();

    // 头部
    ctx.fillStyle = "#d9c2a5";
    ctx.beginPath();
    ctx.arc(0.32 * s, -0.1 * s, 0.19 * s, 0, Math.PI * 2);
    ctx.fill();

    // 喙（向右）
    ctx.fillStyle = "#f2b31f";
    ctx.beginPath();
    ctx.moveTo(0.5 * s, -0.1 * s);
    ctx.lineTo(0.82 * s, -0.03 * s);
    ctx.lineTo(0.52 * s, 0.03 * s);
    ctx.closePath();
    ctx.fill();

    // 眼睛
    ctx.fillStyle = "#111";
    ctx.beginPath();
    ctx.arc(0.33 * s, -0.13 * s, 0.028 * s, 0, Math.PI * 2);
    ctx.fill();

    // 尾羽
    ctx.fillStyle = "#6b4428";
    ctx.beginPath();
    ctx.moveTo(-0.6 * s, 0.0 * s);
    ctx.lineTo(-0.86 * s, -0.1 * s);
    ctx.lineTo(-0.82 * s, 0.11 * s);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  function drawGame(nowSec) {
    const w = game.w,
      h = game.h;
    gctx.clearRect(0, 0, w, h);
    gctx.fillStyle = "#e8e8e8";
    gctx.fillRect(0, game.groundY, w, h - game.groundY);
    gctx.strokeStyle = "#666";
    gctx.lineWidth = 2;
    gctx.beginPath();
    gctx.moveTo(0, game.groundY);
    gctx.lineTo(w, game.groundY);
    gctx.stroke();

    gctx.font = "10px sans-serif";
    for (let b = 0; b <= 3; b++) {
      const ly = game.groundY - (b / 3) * MAX_FOCUS_LIFT;
      gctx.strokeStyle = "rgba(200,200,200,0.5)";
      gctx.lineWidth = 1;
      gctx.setLineDash([4, 4]);
      gctx.beginPath();
      gctx.moveTo(0, ly);
      gctx.lineTo(56, ly);
      gctx.stroke();
      gctx.beginPath();
      gctx.moveTo(w - 56, ly);
      gctx.lineTo(w, ly);
      gctx.stroke();
      gctx.setLineDash([]);
      gctx.fillStyle = "#999";
      gctx.textAlign = "left";
      gctx.fillText(String(b), 4, ly - 2);
      gctx.textAlign = "right";
      gctx.fillText(String(b), w - 4, ly - 2);
    }
    gctx.textAlign = "left";
    gctx.fillStyle = "#777";
    gctx.font = "9px sans-serif";
    gctx.fillText("专注度 0–3", 2, game.groundY - MAX_FOCUS_LIFT - 4);
    gctx.textAlign = "right";
    gctx.fillText("专注度 0–3", w - 2, game.groundY - MAX_FOCUS_LIFT - 4);
    gctx.textAlign = "left";

    // 障碍改为 emoji 小山（视觉替换，不改变碰撞盒）
    for (const o of game.obstacles) {
      const mountainSize = Math.max(20, o.h + 6);
      const x = o.x - 2;
      const y = game.groundY + 2;
      gctx.font = `${mountainSize}px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif`;
      gctx.textAlign = "left";
      gctx.textBaseline = "bottom";
      if ((o.hitFlash || 0) > 0) {
        gctx.save();
        gctx.shadowColor = "rgba(255, 200, 60, 0.9)";
        gctx.shadowBlur = 10;
        gctx.fillText("⛰️", x, y);
        gctx.restore();
      } else {
        gctx.fillText("⛰️", x, y);
      }
    }

    // 玩家改为固定“朝右”老鹰（跨平台一致）
    drawEagleRight(gctx, game.playerX, game.playerY, game.playerR * 1.6);

    gctx.fillStyle = "#555";
    gctx.font = "11px sans-serif";
    gctx.textAlign = "left";
    const hint = game.stalled
      ? "专注不足 — 场景停滞，提高专注后小球会缓慢上升"
      : "高度仅随专注度缓慢变化；无自动越障";
    gctx.fillText(hint, 8, 18);
  }

  function drawChart(nowSec) {
    const w = chartCanvas.width;
    const h = chartCanvas.height;
    const ml = 36,
      mr = 10,
      mt = 8,
      mb = 22;
    const pw = w - ml - mr;
    const ph = h - mt - mb;
    cctx.fillStyle = "#fff";
    cctx.fillRect(0, 0, w, h);
    cctx.strokeStyle = "#e0e0e0";
    
    // Y 轴：与游戏/摄像头同一套 0–3 专注度标尺（主刻度 0,1,2,3；内分 0.25）
    for (let i = 0; i <= 12; i++) {
      const y = mt + ph - (i / 12.0) * ph;
      cctx.beginPath();
      cctx.moveTo(ml, y);
      cctx.lineTo(w - mr, y);
      cctx.stroke();
      const value = i * 0.25;
      if (i % 4 === 0) {
        cctx.fillStyle = "#666";
        cctx.font = "11px sans-serif";
        cctx.fillText(String(value.toFixed(0)), 8, y + 4);
      } else {
        cctx.fillStyle = "#ccc";
        cctx.font = "9px sans-serif";
        cctx.fillText(value.toFixed(2), 8, y + 3);
      }
    }
    cctx.fillStyle = "#888";
    cctx.font = "10px sans-serif";
    cctx.fillText("专注度", 2, mt + 12);
    
    const pts = history.filter(([t]) => nowSec - t <= CHART_SEC);
    if (pts.length >= 2) {
      cctx.strokeStyle = "#1a73e8";
      cctx.lineWidth = 2;
      cctx.beginPath();
      const toXY = (p) => {
        const [t, v] = p;
        const x = ml + ((t - (nowSec - CHART_SEC)) / CHART_SEC) * pw;
        const v_clamped = Math.max(0.0, Math.min(3.0, parseFloat(v)));
        const y = mt + ph - (v_clamped / 3.0) * ph;
        return [x, y];
      };
      const [x0, y0] = toXY(pts[0]);
      cctx.moveTo(x0, y0);
      for (let i = 1; i < pts.length - 1; i++) {
        const [x, y] = toXY(pts[i]);
        const [xn, yn] = toXY(pts[i + 1]);
        const cx = (x + xn) / 2;
        const cy = (y + yn) / 2;
        cctx.quadraticCurveTo(x, y, cx, cy);
      }
      if (pts.length > 2) {
        const [xLast1, yLast1] = toXY(pts[pts.length - 2]);
        const [xLast, yLast] = toXY(pts[pts.length - 1]);
        cctx.quadraticCurveTo(xLast1, yLast1, xLast, yLast);
      } else {
        const [x1, y1] = toXY(pts[1]);
        cctx.lineTo(x1, y1);
      }
      cctx.stroke();
    } else if (pts.length === 1) {
      const [t, v] = pts[0];
      const x = ml + ((t - (nowSec - CHART_SEC)) / CHART_SEC) * pw;
      const v_clamped = Math.max(0.0, Math.min(3.0, parseFloat(v)));
      const y = mt + ph - (v_clamped / 3.0) * ph;
      cctx.fillStyle = "#1a73e8";
      cctx.beginPath();
      cctx.arc(x, y, 3, 0, Math.PI * 2);
      cctx.fill();
    }
    cctx.fillStyle = "#999";
    cctx.font = "11px sans-serif";
    cctx.textAlign = "center";
    cctx.fillText("0 ← 时间(秒) → 120", w / 2, 12);
    for (let sec = 0; sec <= CHART_SEC; sec += 20) {
      const x = ml + (sec / CHART_SEC) * pw;
      cctx.fillText(String(sec), x, h - 6);
    }
    cctx.textAlign = "left";
  }

  let lastApi = {
    cls_idx: null,
    speed: 3,
    obstacle_h: 3,
    face_detected: false,
  };

  async function pollState() {
    try {
      const r = await fetch("/api/state", { cache: "no-store" });
      const j = await r.json();
      lastApi = j

      const cls = j.cls_idx;
      const now = performance.now() / 1000;
      
      // 与后端一致：期望等级 ∈ [0,3]（∑i·p_i），无人脸为 0
      const preciseFocus = j.precise_focus;
      if (j.face_detected && preciseFocus !== null && preciseFocus !== undefined) {
        const target = clamp03(preciseFocus);
        focusTarget = target === null ? 0 : target;
      } else {
        focusTarget = 0;
      }

      if (j.face_detected && cls !== null && cls !== undefined && cls >= 0 && cls <= 3) {
        focusBadge.textContent =
          "专注度等级 " + cls + " / 3（软值 " + (typeof preciseFocus === "number" ? preciseFocus.toFixed(2) : "—") + "）";
        focusBadge.style.background = FOCUS_COLORS[cls];
      } else {
        focusBadge.textContent =
          "专注度等级 0 / 3（无人脸，与曲线一致）";
        focusBadge.style.background = "#666";
      }

      // 更新AI策略显示
      const aiStrategy = j.ai_strategy;
      if (aiStrategy) {
        const rhythmMap = {
          "stable": "平稳节奏",
          "rising": "递增节奏", 
          "relaxing": "放松节奏"
        };
        const rhythmText = rhythmMap[aiStrategy.rhythm] || aiStrategy.rhythm;
        aiLine.textContent = `AI 策略：${rhythmText} | 速度：${aiStrategy.speed} | 障碍高度(0–3)：${Number(aiStrategy.obstacle_height).toFixed(2)}`;
      } else {
        aiLine.textContent =
          "AI 策略：" +
          (j.strategy || "—") +
          " | 速度：" +
          (j.speed ?? "—") +
          " | 障碍高度(0–3)：" +
          (j.obstacle_h != null ? Number(j.obstacle_h).toFixed(2) : "—");
      }

      // 更新AI回复显示
      const aiReport = j.ai_report;
      if (aiReport && aiReport.summary) {
        const reportKey = JSON.stringify(aiReport);
        if (reportKey !== lastAiResponseText) {
          const timestamp = new Date().toLocaleTimeString();
          const reasons = Array.isArray(aiReport.reasoning) ? aiReport.reasoning : [];
          const reasonHtml = reasons
            .slice(0, 6)
            .map((x) => `<li>${escapeHtml(x)}</li>`)
            .join("");
          const metrics = aiReport.metrics || {};
          aiResponse.innerHTML = `
            <div class="ai-response-timestamp">[${timestamp}] ${escapeHtml(aiReport.title || "AI 策略解释")}</div>
            <div class="ai-response-content"><strong>结论：</strong>${escapeHtml(aiReport.summary)}</div>
            ${
              reasonHtml
                ? `<div class="ai-response-content"><strong>依据：</strong><ul>${reasonHtml}</ul></div>`
                : ""
            }
            <div class="ai-response-content"><strong>样本数：</strong>${escapeHtml(metrics.sample_count ?? "—")} | <strong>短/中/长均值：</strong>${escapeHtml(metrics.short_mean ?? "—")} / ${escapeHtml(metrics.mid_mean ?? "—")} / ${escapeHtml(metrics.long_mean ?? "—")}</div>
          `;
          lastAiResponseText = reportKey;
        }
      } else if (j.ai_response && j.ai_response !== "暂无AI回复") {
        if (j.ai_response !== lastAiResponseText) {
          const timestamp = new Date().toLocaleTimeString();
          aiResponse.innerHTML = `
            <div class="ai-response-timestamp">[${timestamp}] AI最新回复：</div>
            <div class="ai-response-content">${escapeHtml(j.ai_response)}</div>
          `;
          lastAiResponseText = j.ai_response;
        }
      } else if (!j.ai_response || j.ai_response === "暂无AI回复") {
        if (lastAiResponseText !== "__placeholder__") {
          aiResponse.innerHTML = '<div class="ai-response-placeholder">等待AI策略生成...</div>';
          lastAiResponseText = "__placeholder__";
        }
      }

      let obsLv = null;
      if (aiStrategy && aiStrategy.obstacle_height != null && aiStrategy.speed != null) {
        obsLv = clamp03(aiStrategy.obstacle_height);
        applyDifficulty(aiStrategy.speed, obsLv);
      } else if (j.speed != null && j.obstacle_h != null) {
        obsLv = clamp03(j.obstacle_h);
        applyDifficulty(j.speed, obsLv);
      } else {
        const sp = j.speed != null ? j.speed : 3;
        if (j.face_detected && cls !== null && cls !== undefined && cls >= 0 && cls <= 3) {
          obsLv = clamp03(3 - cls);
        } else {
          obsLv = 3;
        }
        applyDifficulty(sp, obsLv);
      }

      const elapsed = now - game.gameStart;
      const mm = Math.floor(elapsed / 60);
      const ss = Math.floor(elapsed % 60);
      const diff =
        j.face_detected && cls !== null && cls !== undefined && cls >= 0 && cls <= 3
          ? cls
          : 0;
      const st = gameEnded ? "已结束" : (game.stalled ? "停滞（待专注）" : "进行中");
      statusLine.textContent =
        "【游戏状态】 " +
        st +
        " | 时间：" +
        String(mm).padStart(2, "0") +
        ":" +
        String(ss).padStart(2, "0") +
        " | 专注度(0–3)：" +
        diff +
        ` | 仁慈次数：${mercyCount}/${maxMercyCount}` +
        ` | 终点：${Math.round(sessionDurationSec / 60)}分钟`;
    } catch (e) {
      console.warn(e);
    }
  }

  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      if (gameEnded) return;
      game.vy -= 3.2;
    }
    if (e.key === "r" || e.key === "R") resetGame();
  });

  mercyModalClose.addEventListener("click", hideMercyModal);
  mercyModal.addEventListener("click", (e) => {
    if (e.target === mercyModal) hideMercyModal();
  });

  applyDurationBtn.addEventListener("click", () => {
    const min = Number(durationInput.value);
    const clamped = Math.max(1, Math.min(60, Number.isFinite(min) ? min : 5));
    durationInput.value = String(Math.round(clamped));
    sessionDurationSec = Math.round(clamped) * 60;
    resetGame();
    showMercyModal(`已应用新的终点时长：${Math.round(clamped)} 分钟，并已重开游戏。`);
  });

  let lastT = performance.now() / 1000;
  function loop(t) {
    const now = t / 1000;
    const dt = Math.min(0.05, now - lastT);
    lastT = now;

    // 曲线使用连续时间平滑：先限速，再低通，避免 0/1/2/3 离散值导致跳变
    const maxStepPerSec = 3.2;
    const maxStep = maxStepPerSec * dt;
    const targetDelta = focusTarget - focusDisplay;
    const limitedTarget = focusDisplay + Math.max(-maxStep, Math.min(maxStep, targetDelta));
    const tau = 0.35;
    const alpha = 1 - Math.exp(-dt / tau);
    focusDisplay += (limitedTarget - focusDisplay) * alpha;
    focusDisplay = Math.max(0, Math.min(3, focusDisplay));

    // 以固定采样频率写入历史，绘图更平顺
    if (now - lastHistoryPushSec >= 1 / 15) {
      history.push([now, focusDisplay]);
      sessionHistory.push([now, focusDisplay]);
      lastHistoryPushSec = now;
      while (history.length && now - history[0][0] > CHART_SEC) history.shift();
    }

    const cls = lastApi.cls_idx;
    const face = lastApi.face_detected;
    const isFocused = face && focusDisplay >= 1.1;
    if (!gameEnded) {
      if (!isFocused) unfocusedAccumSec += dt;
      else unfocusedAccumSec = Math.max(0, unfocusedAccumSec - dt * 2.0);

      if (unfocusedAccumSec >= mercyTriggerSec) {
        if (mercyCount >= maxMercyCount) {
          finishGame("不专注触发仁慈机制超过 3 次，提前结束本局", now);
        } else {
          triggerMercy();
        }
      }
      if (now - game.gameStart >= sessionDurationSec) {
        finishGame("达到用户设置的游戏终点时长", now);
      }
    }

    gameTick(dt, now, cls, face);
    drawGame(now);
    drawChart(now);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  setInterval(pollState, 200);
  pollState();
})();
