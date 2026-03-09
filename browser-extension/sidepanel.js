/**
 * AI 购物助手 — Sidepanel
 * LangGraph 驱动的智能导购界面
 */
(function () {
  'use strict';

  // ─── 状态 ──────────────────────────────────────────────────────
  let currentProduct = null;
  let conversationId = null;

  const CURRENCY = { CNY:'¥', HKD:'HK$', USD:'$', EUR:'€', GBP:'£', JPY:'JP¥', AUD:'A$', SGD:'S$', CAD:'C$' };
  const INTENT_LABEL = { search:'🔍 搜索', compare:'⚖️ 对比', chat:'💬 对话' };
  const ACTION_LABEL = {
    buy_now:'💚 立即购买', wait:'⏳ 等待降价',
    consider:'🤔 谨慎考虑', avoid:'❌ 不建议购买',
    neutral:'➡️ 保持观望', cautious:'⚠️ 需要考量'
  };
  const RISK_LABEL  = { low:'低风险', medium:'中等风险', high:'高风险', critical:'危险', unknown:'未知' };

  // ─── 工具 ──────────────────────────────────────────────────────
  function fmtPrice(price, cur = 'CNY') {
    const sym = CURRENCY[cur] || '¥';
    const val = parseFloat(price || 0).toFixed(2);
    return `${sym}${val}`;
  }

  function esc(str) {
    return String(str || '')
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;');
  }

  // 简易 markdown → html（仅用于分析卡片内容）
  function md(str) {
    return esc(str)
      // ## 标题
      .replace(/^#{1,3}\s+(.+)$/gm, '<strong style="font-size:12px;display:block;margin:8px 0 2px">$1</strong>')
      // **粗体**
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // *斜体*
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // - 列表项
      .replace(/^[-•]\s+(.+)$/gm, '<span style="display:block;padding-left:10px;margin:2px 0">• $1</span>')
      // 数字列表
      .replace(/^\d+\.\s+(.+)$/gm, '<span style="display:block;padding-left:10px;margin:2px 0">$1</span>')
      // 换行
      .replace(/\n/g, '<br>');
  }

  function errMsg(e) {
    if (!e) return '未知错误';
    if (typeof e === 'string') return e;
    return e.message || e.detail || JSON.stringify(e).substring(0, 120);
  }

  function show(el)  { el && el.classList.remove('hidden'); }
  function hide(el)  { el && el.classList.add('hidden'); }
  function toggle(el){ el && el.classList.toggle('hidden'); }

  // ─── 初始化 ────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initChat();
    initAnalysis();
    initPrice();
    initVisualSearch();
    initSettings();
    initHeader();

    // 监听来自 background 的消息
    chrome.runtime && chrome.runtime.onMessage && chrome.runtime.onMessage.addListener(
      (req, _sender, sendResp) => {
        if (req.action === 'startAnalysis' || req.action === 'updateProduct') {
          if (req.productData) {
            currentProduct = req.productData;
            renderBanner();
            renderChips();
            autofillPriceSearch();
            // 收到分析请求时自动跳转到分析 tab
            if (req.action === 'startAnalysis') switchTab('analysis');
          }
        }
        sendResp && sendResp({ success: true });
        return true;
      }
    );

    await loadProduct();
    checkStatus();
  });

  // ─── Tab 切换 ──────────────────────────────────────────────────
  function initTabs() {
    document.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.pane));
    });
  }

  function switchTab(pane) {
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.pane === pane));
    document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === `pane-${pane}`));
  }

  // ─── Header ────────────────────────────────────────────────────
  function initHeader() {
    document.getElementById('refreshBtn').addEventListener('click', async () => {
      await loadProduct(true);
    });
    document.getElementById('settingsBtn').addEventListener('click', () => {
      show(document.getElementById('settingsOverlay'));
    });
  }

  async function checkStatus() {
    const dot = document.getElementById('statusDot');
    try {
      await window.apiClient.healthCheck();
      dot.className = 'status-dot online';
      dot.title = '已连接';
    } catch {
      dot.className = 'status-dot offline';
      dot.title = '未连接 — 请确认后端已启动';
    }
  }

  // ─── 商品加载 ──────────────────────────────────────────────────
  async function loadProduct(forceExtract = false) {
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tabs[0]) return;

      const tabId = tabs[0].id;
      const stored = await chrome.storage.local.get([`product_${tabId}`, 'product_current']);
      const p = stored[`product_${tabId}`] || stored.product_current;

      if (p && !forceExtract) {
        currentProduct = p;
      } else {
        // 尝试从页面重新提取
        chrome.tabs.sendMessage(tabId, { action: 'extractProductInfo' }, () => {});
        // 等 1.5s 后重新读取
        await new Promise(r => setTimeout(r, 1500));
        const fresh = await chrome.storage.local.get([`product_${tabId}`, 'product_current']);
        currentProduct = fresh[`product_${tabId}`] || fresh.product_current || null;
      }
    } catch (e) {
      console.warn('loadProduct:', e);
    }

    renderBanner();
    renderChips();
    autofillPriceSearch();
    loadPriceHistory();
  }

  function renderBanner() {
    const banner = document.getElementById('productBanner');
    if (!currentProduct) { hide(banner); return; }
    show(banner);
    const img = document.getElementById('bannerImg');
    img.src = currentProduct.image || '';
    img.style.display = currentProduct.image ? '' : 'none';
    document.getElementById('bannerName').textContent = currentProduct.name || currentProduct.title || '商品';
    document.getElementById('bannerPrice').textContent = fmtPrice(currentProduct.price, currentProduct.currency);
    document.getElementById('bannerPlatform').textContent = currentProduct.platform || '';
  }

  function renderChips() {
    const chips = document.getElementById('quickChips');
    currentProduct ? show(chips) : hide(chips);
  }

  function autofillPriceSearch() {
    if (!currentProduct) return;
    const inp = document.getElementById('priceSearchInput');
    if (inp && !inp.value) inp.value = currentProduct.name || currentProduct.title || '';
  }

  // ─── 聊天 ──────────────────────────────────────────────────────
  function initChat() {
    // "询问 AI" banner 按钮
    document.getElementById('bannerAskBtn').addEventListener('click', () => switchTab('chat'));

    // 快速 prompt 芯片
    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const base = chip.dataset.prompt;
        const msg = currentProduct
          ? `[商品：${currentProduct.name || ''}，价格：${fmtPrice(currentProduct.price, currentProduct.currency)}]\n${base}`
          : base;
        sendChat(msg);
      });
    });

    // 图片上传（聊天区域）
    const imgBtn  = document.getElementById('imgChatBtn');
    const imgFile = document.getElementById('imgChatFile');
    imgBtn.addEventListener('click', () => imgFile.click());
    imgFile.addEventListener('change', async () => {
      if (!imgFile.files[0]) return;
      await handleChatImage(imgFile.files[0]);
      imgFile.value = '';
    });

    // 发送按钮 & Enter
    const input   = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');

    sendBtn.addEventListener('click', () => {
      const msg = input.value.trim();
      if (!msg) return;
      input.value = '';
      resizeTextarea(input);
      sendChat(msg);
    });

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
      }
    });

    // 自动调整高度
    input.addEventListener('input', () => resizeTextarea(input));
  }

  function resizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 100) + 'px';
  }

  async function sendChat(message) {
    appendMsg('user', message);
    showTyping(true);

    const useRag = document.getElementById('ragToggle').checked;

    // 注入商品上下文（如果有）
    const fullMsg = currentProduct && !message.startsWith('[商品：')
      ? `[正在查看商品：${currentProduct.name || ''}，价格：${fmtPrice(currentProduct.price, currentProduct.currency)}，平台：${currentProduct.platform || ''}]\n\n${message}`
      : message;

    try {
      const res = await window.apiClient.sendEnhancedChat({
        message:      fullMsg,
        message_type: 'text',
        model:        'glm-4-0520',
        use_memory:   true,
        use_rag:      useRag,
        knowledge_base_ids: null,
      });

      showTyping(false);

      const meta = {
        intent:    res.agent_collaboration?.intent,
        confidence: res.agent_collaboration?.confidence,
        docs:      res.rag_results?.length || 0,
        time:      res.processing_time,
      };
      appendMsg('assistant', res.response || '抱歉，无法回答这个问题。', meta);

    } catch (e) {
      showTyping(false);
      const txt = errMsg(e).includes('connect')
        ? '⚠️ 无法连接后端，请确认 http://localhost:8000 已启动。'
        : `⚠️ 出错了：${errMsg(e)}`;
      appendMsg('assistant', txt);
    }
  }

  async function handleChatImage(file) {
    const url = URL.createObjectURL(file);
    appendMsg('user', `<img src="${url}" style="max-width:160px;border-radius:8px;margin-top:4px">`, null, true);
    showTyping(true);
    try {
      const res = await window.apiClient.visualSearch(file);
      showTyping(false);
      const items = res.data?.similar_products || res.results || res.products || [];
      if (items.length) {
        const desc = res.data?.image_description ? `<p style="font-size:12px;color:var(--text-2);margin:0 0 8px">🔍 图像识别：${esc(res.data.image_description.slice(0, 80))}…</p>` : '';
        const html = items.slice(0, 5).map(it => `<div class="visual-result-item">
          ${it.image_url || it.image ? `<img class="visual-result-img" src="${esc(it.image_url || it.image)}" onerror="this.style.display='none'">` : ''}
          <div class="visual-result-info">
            <div class="visual-result-name">${esc(it.title || it.name || '商品')}</div>
            <div class="visual-result-price">${fmtPrice(it.price, it.currency)}</div>
            ${it.product_url || it.url ? `<a href="${esc(it.product_url || it.url)}" target="_blank" style="font-size:11px;color:var(--primary)">查看详情 →</a>` : ''}
          </div>
        </div>`).join('');
        appendMsg('assistant', `为您找到 ${items.length} 个相似商品：${desc}<div style="margin-top:8px">${html}</div>`, null, true);
      } else {
        const imgDesc = res.data?.image_description;
        appendMsg('assistant', imgDesc
          ? `已识别图像：${imgDesc}\n\n暂未找到相似商品，数据库中可能没有匹配的美妆产品，请尝试其他图片。`
          : '未找到相似商品，请尝试其他图片。');
      }
    } catch (e) {
      showTyping(false);
      appendMsg('assistant', `图片搜索失败：${errMsg(e)}`);
    }
  }

  function appendMsg(role, content, meta = null, raw = false) {
    const box = document.getElementById('messages');

    let metaHtml = '';
    if (meta && role === 'assistant') {
      const intentKey   = meta.intent || 'chat';
      const intentLabel = INTENT_LABEL[intentKey] || `💬 ${intentKey}`;
      const confPct     = meta.confidence != null ? `${(meta.confidence * 100).toFixed(0)}%` : null;
      const docsStr     = meta.docs > 0 ? `📄 ${meta.docs} 文档` : '';
      const timeStr     = meta.time != null ? `${meta.time.toFixed(1)}s` : '';

      metaHtml = `<div class="msg-meta">
        <span class="meta-badge ${intentKey}">${intentLabel}</span>
        ${confPct ? `<span class="meta-badge conf">置信 ${confPct}</span>` : ''}
        ${docsStr ? `<span class="meta-badge docs">${docsStr}</span>` : ''}
        ${timeStr ? `<span class="meta-time">${timeStr}</span>` : ''}
      </div>`;
    }

    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `
      <div class="msg-bubble">${raw ? content : esc(content).replace(/\n/g, '<br>')}</div>
      ${metaHtml}
    `;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  function showTyping(visible) {
    const el = document.getElementById('typingIndicator');
    visible ? show(el) : hide(el);
    if (visible) document.getElementById('messages').scrollTop = 9999;
  }

  // ─── 分析 ──────────────────────────────────────────────────────
  function initAnalysis() {
    document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);
  }

  async function runAnalysis() {
    const btn = document.getElementById('analyzeBtn');
    const out = document.getElementById('analysisResult');

    if (!currentProduct) {
      out.innerHTML = `<div class="error-state">
        <p>❌ 未检测到商品信息</p>
        <p class="error-hint">请先访问商品详情页，或点击刷新按钮重新提取。</p>
      </div>`;
      return;
    }

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>分析中…`;
    out.innerHTML = skeletonCards();

    try {
      const data = {
        name:       currentProduct.name || currentProduct.title || '',
        price:      currentProduct.price || 0,
        currency:   currentProduct.currency || 'CNY',
        platform:   currentProduct.platform || 'unknown',
        productId:  currentProduct.productId || currentProduct.id || '',
        image:      currentProduct.image || '',
        url:        currentProduct.url || '',
        description: currentProduct.description || '',
        parameters:  currentProduct.parameters || {},
      };

      const res  = await window.apiClient.analyzeProduct(data);
      const info = res.data || res;

      if (!info || (info.error && !info.comprehensive_analysis)) {
        throw new Error(info?.error || '分析返回空结果');
      }

      out.innerHTML = renderAnalysisCards(info);

    } catch (e) {
      out.innerHTML = `<div class="error-state">
        <p>❌ 分析失败：${esc(errMsg(e))}</p>
        <p class="error-hint">请确认后端服务已启动，并检查 API Key 配置。</p>
      </div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = '⚡ 一键全面分析';
    }
  }

  function skeletonCards() {
    return ['综合评估', '购买建议', '价格分析', '风险评估'].map(t => `
      <div class="a-card">
        <div class="a-card-hdr">${t}</div>
        <div class="a-card-body">
          <div class="skeleton" style="width:90%"></div>
          <div class="skeleton" style="width:70%;margin-top:8px"></div>
          <div class="skeleton" style="width:80%;margin-top:8px"></div>
        </div>
      </div>`).join('');
  }

  function renderAnalysisCards(info) {
    const parts = [];

    // 综合评估
    if (info.comprehensive_analysis || info.analysis) {
      parts.push(`<div class="a-card">
        <div class="a-card-hdr">📋 综合评估</div>
        <div class="a-card-body" style="font-size:12px;line-height:1.7">${md(info.comprehensive_analysis || info.analysis)}</div>
      </div>`);
    }

    // 购买建议
    if (info.recommendation) {
      const rec    = info.recommendation;
      const action = rec.action || 'neutral';
      const conf   = rec.confidence != null ? (rec.confidence * 100).toFixed(0) : null;
      parts.push(`<div class="a-card">
        <div class="a-card-hdr">💡 购买建议</div>
        <div class="a-card-body">
          <div class="action-badge ${esc(action)}">${ACTION_LABEL[action] || action}</div>
          ${conf ? `<div class="conf-bar-wrap">
            <div class="conf-bar-label"><span>置信度</span><span>${conf}%</span></div>
            <div class="conf-bar"><div class="conf-bar-fill" style="width:${conf}%"></div></div>
          </div>` : ''}
          ${rec.reason ? `<p style="margin-top:8px;font-size:12px;color:var(--text-2)">${esc(rec.reason)}</p>` : ''}
        </div>
      </div>`);
    }

    // 价格分析
    if (info.price_analysis && !info.price_analysis.error) {
      const pa = info.price_analysis;
      parts.push(`<div class="a-card">
        <div class="a-card-hdr">💰 价格分析</div>
        <div class="a-card-body">
          <div class="price-row">
            <span class="price-row-name">当前价格</span>
            <span class="price-row-val">${fmtPrice(pa.current_price || currentProduct.price, currentProduct.currency)}</span>
          </div>
          ${pa.lowest_found_price ? `<div class="price-row">
            <span class="price-row-name">历史最低</span>
            <span class="price-row-val" style="color:var(--success)">${fmtPrice(pa.lowest_found_price, 'CNY')}</span>
          </div>` : ''}
          ${pa.savings_potential > 0 ? `<div style="margin-top:6px;font-size:12px;color:var(--success)">
            💸 潜在节省 ${fmtPrice(pa.savings_potential, 'CNY')}
          </div>` : ''}
          ${pa.platform ? `<div style="margin-top:6px;font-size:11px;color:var(--text-3)">平台：${esc(pa.platform)}</div>` : ''}
        </div>
      </div>`);
    }

    // 风险评估
    if (info.risk_analysis && !info.risk_analysis.error) {
      const ra    = info.risk_analysis;
      const level = ra.overall_risk_level || 'unknown';
      const risks = ra.detailed_risks || [];
      const suggs = ra.mitigation_suggestions || [];
      parts.push(`<div class="a-card">
        <div class="a-card-hdr">⚠️ 风险评估</div>
        <div class="a-card-body">
          <span class="risk-badge ${esc(level)}">${RISK_LABEL[level] || level}</span>
          <span style="font-size:12px;color:var(--text-2);margin-left:8px">共发现 ${ra.risk_count || risks.length} 个风险</span>
          ${risks.length ? `<div style="margin-top:10px">
            ${risks.map(r => `<div class="risk-item">
              <strong>${esc(r.risk_type || '未知')}</strong>
              <div style="color:var(--text-2)">${esc(r.evidence || '')}</div>
              ${r.severity != null ? `<div class="risk-sev">严重度 ${(r.severity * 100).toFixed(0)}%</div>` : ''}
            </div>`).join('')}
          </div>` : ''}
          ${suggs.length ? `<div style="margin-top:10px">
            <div style="font-size:11px;font-weight:700;color:var(--text-2);margin-bottom:4px">💡 建议</div>
            <ul class="suggestion-list">${suggs.map(s => `<li>${esc(s)}</li>`).join('')}</ul>
          </div>` : ''}
        </div>
      </div>`);
    }

    return parts.length ? parts.join('') : `<div class="empty-state"><p>分析完成，无详细数据。</p></div>`;
  }

  // ─── 比价 ──────────────────────────────────────────────────────
  function initPrice() {
    // 搜索
    document.getElementById('priceSearchBtn').addEventListener('click', runPriceSearch);
    document.getElementById('priceSearchInput').addEventListener('keydown', e => {
      if (e.key === 'Enter') runPriceSearch();
    });

    // 折叠历史
    document.getElementById('historyToggle').addEventListener('click', () => {
      const body    = document.getElementById('historyBody');
      const chevron = document.getElementById('historyChevron');
      toggle(body); chevron.classList.toggle('open');
    });

    // 折叠预警
    document.getElementById('alertToggle').addEventListener('click', () => {
      const body    = document.getElementById('alertBody');
      const chevron = document.getElementById('alertChevron');
      toggle(body); chevron.classList.toggle('open');
    });

    // 设置预警
    document.getElementById('setAlertBtn').addEventListener('click', runSetAlert);
  }

  async function runPriceSearch() {
    const query = document.getElementById('priceSearchInput').value.trim();
    if (!query) return;

    const out = document.getElementById('comparisonResult');
    out.innerHTML = `<div class="empty-state-sm"><span class="spinner" style="border-color:rgba(108,92,231,.3);border-top-color:var(--primary)"></span> 搜索中…</div>`;

    try {
      const res = await window.apiClient.comparePrices(query, ['jd','taobao','pdd']);
      out.innerHTML = renderComparison(res, query);
    } catch (e) {
      out.innerHTML = `<div class="error-state">❌ 比价失败：${esc(errMsg(e))}</div>`;
    }
  }

  function renderComparison(res, query) {
    // 兼容多种后端返回格式
    let platformPrices = {};

    if (res.all_products?.platform_prices) {
      platformPrices = res.all_products.platform_prices;
    } else if (res[query]?.platform_prices) {
      platformPrices = res[query].platform_prices;
    } else {
      // 尝试直接用对象的 value
      const val = Object.values(res)[0];
      if (val?.platform_prices) platformPrices = val.platform_prices;
    }

    const platforms = Object.keys(platformPrices);
    if (!platforms.length) return `<div class="empty-state-sm">未找到比价数据</div>`;

    // 找最低价
    let minPrice = Infinity, bestPlatform = '';
    platforms.forEach(plat => {
      const items = Array.isArray(platformPrices[plat]) ? platformPrices[plat] : [platformPrices[plat]];
      items.forEach(item => {
        const p = parseFloat(item.price || 0);
        if (p > 0 && p < minPrice) { minPrice = p; bestPlatform = plat; }
      });
    });

    const PLAT_NAME = { jd:'京东', taobao:'淘宝/天猫', pdd:'拼多多', xiaohongshu:'小红书', douyin:'抖音' };

    return platforms.map(plat => {
      const items = Array.isArray(platformPrices[plat]) ? platformPrices[plat] : [platformPrices[plat]];
      const label = PLAT_NAME[plat] || plat.toUpperCase();
      const rows  = items.slice(0, 3).map(item => {
        const isBest = plat === bestPlatform && parseFloat(item.price) === minPrice;
        return `<div class="price-row${isBest ? ' best' : ''}">
          <span class="price-row-name">${esc(item.title || item.name || label)}</span>
          <span class="price-row-val">${fmtPrice(item.price, item.currency || 'CNY')}</span>
          ${isBest ? '<span class="best-badge">最低</span>' : ''}
          ${item.product_url ? `<a href="${esc(item.product_url)}" target="_blank">→</a>` : ''}
        </div>`;
      }).join('');
      return `<div class="platform-group">
        <div class="platform-label">${label}</div>
        ${rows}
      </div>`;
    }).join('');
  }

  async function loadPriceHistory() {
    // 历史价格功能暂未接入数据，隐藏此板块
    const block = document.getElementById('historyBlock');
    if (block) block.style.display = 'none';
  }

  async function runSetAlert() {
    const target = parseFloat(document.getElementById('targetPriceInput').value);
    const out    = document.getElementById('alertResult');

    if (!target || target <= 0) {
      out.innerHTML = `<div class="error-state">请输入有效的目标价格</div>`; return;
    }
    if (!currentProduct?.productId) {
      out.innerHTML = `<div class="error-state">当前商品没有 ID，无法设置预警</div>`; return;
    }

    try {
      await window.apiClient.trackPrice(currentProduct.productId, target);
      out.innerHTML = `<div class="success-state">
        ✅ 已设置预警：当价格降至 ${fmtPrice(target, currentProduct.currency)} 时通知你
      </div>`;
    } catch (e) {
      out.innerHTML = `<div class="error-state">设置失败：${esc(errMsg(e))}</div>`;
    }
  }

  // ─── 图搜 ──────────────────────────────────────────────────────
  function initVisualSearch() {
    const fileInput = document.getElementById('visualFileInput');
    const zone      = document.getElementById('uploadZone');
    const preview   = document.getElementById('visualPreview');
    const previewImg = document.getElementById('previewImg');
    let selectedFile = null;

    // 文件选择
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) showVisualPreview(fileInput.files[0]);
    });

    // 拖放
    zone.addEventListener('dragover', e => { e.preventDefault(); document.querySelector('.upload-label').classList.add('dragover'); });
    zone.addEventListener('dragleave', () => document.querySelector('.upload-label').classList.remove('dragover'));
    zone.addEventListener('drop', e => {
      e.preventDefault();
      document.querySelector('.upload-label').classList.remove('dragover');
      if (e.dataTransfer.files[0]) showVisualPreview(e.dataTransfer.files[0]);
    });

    function showVisualPreview(file) {
      selectedFile = file;
      previewImg.src = URL.createObjectURL(file);
      show(preview);
      document.getElementById('uploadZone').style.display = 'none';
    }

    // 搜索
    document.getElementById('doVisualSearchBtn').addEventListener('click', async () => {
      if (!selectedFile) return;
      const btn = document.getElementById('doVisualSearchBtn');
      const out = document.getElementById('visualResult');

      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span>搜索中…`;
      out.innerHTML = '';

      try {
        const res   = await window.apiClient.visualSearch(selectedFile);
        const items = res.data?.similar_products || res.results || res.products || [];

        if (items.length) {
          out.innerHTML = `<div style="font-size:12px;color:var(--text-2);margin-bottom:8px">找到 ${items.length} 个相似商品：</div>` +
            items.slice(0, 10).map(it => `<div class="visual-result-item">
              ${it.image_url || it.image ? `<img class="visual-result-img" src="${esc(it.image_url || it.image)}" onerror="this.style.display='none'">` : ''}
              <div class="visual-result-info">
                <div class="visual-result-name">${esc(it.title || it.name || '商品')}</div>
                <div class="visual-result-price">${fmtPrice(it.price, it.currency)}</div>
                ${it.product_url || it.url ? `<a href="${esc(it.product_url || it.url)}" target="_blank">查看详情 →</a>` : ''}
              </div>
            </div>`).join('');
        } else {
          out.innerHTML = `<div class="empty-state"><div class="empty-icon">🔍</div><p>未找到相似商品，换张图片试试？</p></div>`;
        }
      } catch (e) {
        out.innerHTML = `<div class="error-state">❌ 搜索失败：${esc(errMsg(e))}</div>`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '🔍 搜索同款';
      }
    });

    // 清除
    document.getElementById('clearVisualBtn').addEventListener('click', () => {
      selectedFile = null;
      previewImg.src = '';
      fileInput.value = '';
      hide(preview);
      document.getElementById('uploadZone').style.display = '';
      document.getElementById('visualResult').innerHTML = '';
    });
  }

  // ─── 设置 ──────────────────────────────────────────────────────
  function initSettings() {
    // 读取已保存配置
    chrome.storage.sync.get(['config'], res => {
      const cfg = res.config || {};
      document.getElementById('apiUrlInput').value = cfg.apiUrl || 'http://localhost:8000';
      document.getElementById('autoExtractToggle').checked = cfg.autoExtract !== false;
    });

    document.getElementById('closeSettingsBtn').addEventListener('click', () => {
      hide(document.getElementById('settingsOverlay'));
    });

    document.getElementById('saveSettingsBtn').addEventListener('click', () => {
      const cfg = {
        apiUrl:      document.getElementById('apiUrlInput').value.trim() || 'http://localhost:8000',
        autoExtract: document.getElementById('autoExtractToggle').checked,
        enabled:     true,
      };
      chrome.storage.sync.set({ config: cfg }, () => {
        hide(document.getElementById('settingsOverlay'));
        checkStatus();
      });
    });

    // 点击遮罩关闭
    document.getElementById('settingsOverlay').addEventListener('click', function (e) {
      if (e.target === this) hide(this);
    });
  }

})();
