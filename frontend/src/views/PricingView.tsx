import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { useAutoSave } from "../hooks/useAutoSave";
import { fmtNum } from "../lib/format";
import type {
  CandidateBrief,
  MatchedDefault,
  PriceEntry,
  PriceSelection,
  PricingUnpriced,
  ProviderModelInfo,
  SourceStatus,
  UserPricingEntry,
} from "../lib/types";
import { Panel } from "../components/Panel";
import { Button } from "../components/Button";
import { SaveToast } from "../components/SaveToast";
import { Segmented } from "../components/Segmented";
import { Loading, ErrorBox } from "../components/Feedback";
import {
  DraftEntry,
  ProviderPricingCard,
  draftToEntry,
  entryToDraft,
  isDraftEmpty,
} from "../components/ProviderPricingCard";
import { SourceStatusBar } from "../components/price/SourceStatusBar";

// 卡片筛选（F6.7）：按生效价格来源归类
type PricingFilter = "all" | "unpriced" | "candidates" | "selected" | "manual";

const FILTER_OPTIONS: { value: PricingFilter; label: string }[] = [
  { value: "all", label: "全部" },
  { value: "unpriced", label: "未定价" },
  { value: "candidates", label: "待选候选" },
  { value: "selected", label: "已选源" },
  { value: "manual", label: "手工定价" },
];

export function PricingView({ refreshNonce }: { refreshNonce: number }) {
  const res = useApi(() => api.getPricing(), [refreshNonce]);
  const data = res.data;
  const [drafts, setDrafts] = useState<Record<string, DraftEntry>>({});
  const [resetResult, setResetResult] = useState("");
  // 两段式确认：首次点击武装，4 秒内再次点击执行
  const [resetArmed, setResetArmed] = useState(false);
  const resetArmTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [ready, setReady] = useState(false);
  // 跳转信号：点击未定价告警行时递增，传给对应 provider 卡片触发脉冲动画
  const [highlightTarget, setHighlightTarget] = useState<string | null>(null);
  const [highlightSignal, setHighlightSignal] = useState(0);
  // 局部未定价告警覆盖：保存后单独刷新，避免整页 refetch 导致闪烁
  const [unpricedOverride, setUnpricedOverride] = useState<
    PricingUnpriced[] | null
  >(null);
  const [filter, setFilter] = useState<PricingFilter>("all");
  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState("");
  const [selecting, setSelecting] = useState(false);
  const [autoSync, setAutoSync] = useState(false);

  useEffect(() => {
    if (!data) return;
    const next: Record<string, DraftEntry> = {};
    const userPricing = data.user_pricing || {};
    for (const [pid, entry] of Object.entries(userPricing)) {
      next[pid] = entryToDraft(entry);
    }
    setDrafts(next);
    setReady(true);
    setUnpricedOverride(null); // 新数据到达时清除覆盖
  }, [data]);

  useEffect(() => {
    let mounted = true;
    void api
      .getConfig()
      .then((config) => {
        const priceSync = config.price_sync;
        const enabled =
          typeof priceSync === "object" &&
          priceSync !== null &&
          (priceSync as Record<string, unknown>).auto_enabled === true;
        if (mounted) setAutoSync(enabled);
      })
      .catch(() => {
        // 读取失败时保持默认关闭；不影响手动同步。
      });
    return () => {
      mounted = false;
    };
  }, [refreshNonce]);
  const providerModels: ProviderModelInfo[] = data?.provider_models || [];
  const defaults: Record<string, PriceEntry> = data?.defaults || {};
  const unpriced = unpricedOverride ?? data?.unpriced ?? [];
  const sources: Record<string, SourceStatus> = data?.sources || {};
  const selections: Record<string, Record<string, PriceSelection>> =
    data?.selections || {};
  const candidateMap: Record<string, Record<string, CandidateBrief[]>> =
    data?.candidates || {};
  const autoSelected: Record<string, Record<string, string>> =
    data?.auto_selected || {};

  // 当前配置中的 provider ID 集合
  const configIds = useMemo(
    () => new Set(providerModels.map((p) => p.id)),
    [providerModels],
  );

  // 已设过定价但不在当前配置中的「孤儿」provider
  const orphanIds = Object.keys(drafts).filter(
    (pid) => !configIds.has(pid),
  );

  // 从 unpriced 中提取不在当前配置中、也未在 drafts 中的「历史」provider
  const historicalIds = useMemo(() => {
    const seen = new Set([...configIds, ...orphanIds]);
    const ids: string[] = [];
    for (const u of unpriced) {
      const pid = u.provider_id || "";
      if (pid && !seen.has(pid)) {
        seen.add(pid);
        ids.push(pid);
      }
    }
    return ids;
  }, [unpriced, configIds, orphanIds]);

  // 短名：取最后一个 / 后面的部分（如 newapi/image-ocr → image-ocr）
  const shortName = (id: string) => {
    const i = id.lastIndexOf("/");
    return i >= 0 ? id.slice(i + 1) : id;
  };

  // provider 是否已有生效的价格来源（已选或自动匹配）
  const hasSourceSelection = (pid: string) => {
    const sel = selections[pid];
    if (sel && Object.keys(sel).length > 0) return true;
    const auto = autoSelected[pid];
    return !!auto && Object.keys(auto).length > 0;
  };

  // provider 是否存在待选候选
  const hasCandidates = (pid: string) => {
    const m = candidateMap[pid];
    return !!m && Object.values(m).some((arr) => arr && arr.length > 0);
  };

  // 原始显示列表（去重前）
  const rawList: {
    id: string;
    displayId: string;
    type?: string;
    candidates: string[];
    matchedDefault: MatchedDefault | null;
    isHistorical?: boolean;
  }[] = [
    ...providerModels.map((p) => ({
      id: p.id,
      displayId: p.id,
      type: p.type,
      candidates: p.candidates,
      matchedDefault: p.matched_default ?? null,
    })),
    ...orphanIds.map((id) => ({
      id,
      displayId: id,
      type: undefined,
      candidates: [],
      matchedDefault: null,
    })),
    ...historicalIds.map((id) => ({
      id,
      displayId: id,
      type: undefined,
      candidates: [],
      matchedDefault: null,
      isHistorical: true,
    })),
  ];

  // 去重：按短名分组，如果多个 ID 共享同一短名（如 newapi/image-ocr 和 image-ocr），
  // 保留最长 ID（更具体），displayId 用短名，删除较短的重复项
  const displayList = useMemo(() => {
    const byShort = new Map<string, typeof rawList>();
    for (const item of rawList) {
      const sn = shortName(item.displayId);
      const existing = byShort.get(sn);
      if (!existing) {
        byShort.set(sn, [item]);
      } else {
        existing.push(item);
      }
    }
    const result: typeof rawList = [];
    for (const [, group] of byShort) {
      if (group.length === 1) {
        result.push(group[0]);
      } else {
        // 保留最长 ID，displayId 用短名
        group.sort((a, b) => b.id.length - a.id.length);
        const kept = { ...group[0], displayId: shortName(group[0].displayId) };
        result.push(kept);
      }
    }
    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providerModels, orphanIds, historicalIds]);

  // 筛选后的卡片列表（F6.7）
  const filteredList = useMemo(() => {
    if (filter === "all") return displayList;
    return displayList.filter((p) => {
      const manual = !isDraftEmpty(ensureDraft(p.id));
      switch (filter) {
        case "manual":
          return manual;
        case "selected":
          return !manual && hasSourceSelection(p.id);
        case "candidates":
          return hasCandidates(p.id);
        case "unpriced":
          return !manual && !p.matchedDefault && !hasSourceSelection(p.id);
        default:
          return true;
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayList, filter, drafts, selections, autoSelected, candidateMap]);

  // 未定价告警按 provider_id 分组（去重后），用于可点击跳转
  const unpricedByProvider = useMemo(() => {
    type UGroup = { models: typeof unpriced; totalTokens: number };
    const map = new Map<string, UGroup>();
    for (const u of unpriced) {
      const pid = u.provider_id || "(未知)";
      const group = map.get(pid) || { models: [], totalTokens: 0 };
      group.models.push(u);
      group.totalTokens += u.tokens || 0;
      map.set(pid, group);
    }
    // 按短名合并：newapi/image-ocr 和 image-ocr 合并为一行
    const byShort = new Map<string, UGroup & { fullId: string }>();
    for (const [pid, group] of map) {
      const sn = shortName(pid);
      const existing = byShort.get(sn);
      if (!existing) {
        byShort.set(sn, { ...group, fullId: pid });
      } else {
        existing.models.push(...group.models);
        existing.totalTokens += group.totalTokens;
        if (pid.length > existing.fullId.length) existing.fullId = pid;
      }
    }
    return Array.from(byShort.entries())
      .map(([, g]) => [g.fullId, g] as [string, UGroup])
      .sort((a, b) => b[1].totalTokens - a[1].totalTokens);
  }, [unpriced]);

  // 存在未定价用量的 provider ID 及短名集合，用于卡片背景色判定
  const unpricedIdSet = useMemo(() => {
    const s = new Set<string>();
    for (const u of unpriced) {
      const pid = u.provider_id || "";
      if (pid) {
        s.add(pid);
        s.add(shortName(pid));
      }
    }
    return s;
  }, [unpriced]);

  const hasUnpricedUsage = (pid: string) =>
    unpricedIdSet.has(pid) || unpricedIdSet.has(shortName(pid));

  const updateDraft = (pid: string, patch: Partial<DraftEntry>) =>
    setDrafts((prev) => {
      const cur = prev[pid] ?? entryToDraft(undefined);
      return { ...prev, [pid]: { ...cur, ...patch } };
    });
  const clearDraft = (pid: string) =>
    setDrafts((prev) => {
      const next = { ...prev };
      delete next[pid];
      return next;
    });
  const ensureDraft = (pid: string): DraftEntry =>
    drafts[pid] ?? entryToDraft(undefined);

  const collect = (): Record<string, UserPricingEntry> => {
    const out: Record<string, UserPricingEntry> = {};
    for (const [pid, d] of Object.entries(drafts)) {
      if (isDraftEmpty(d)) continue;
      const entry = draftToEntry(d);
      if (entry) out[pid] = entry;
    }
    return out;
  };

  const payload = useMemo<{
    pricing: Record<string, UserPricingEntry> | null;
    error?: string;
  }>(() => {
    try {
      return { pricing: collect() };
    } catch (e) {
      return { pricing: null, error: e instanceof Error ? e.message : String(e) };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drafts]);

  const { status, error } = useAutoSave(
    payload,
    async (p) => {
      if (p.error) throw new Error(p.error);
      await api.postSaveConfig({ pricing: p.pricing });
      // 局部刷新未定价告警，避免整页 refetch 导致闪烁
      try {
        const fresh = await api.getPricing();
        setUnpricedOverride(fresh.unpriced ?? []);
      } catch {
        // 刷新失败不影响保存成功
      }
    },
    { enabled: ready },
  );

  // ---- 价格源操作（F1/F2） ----

  const doSync = async (sourceIds?: string[]) => {
    setSyncing(true);
    setSyncMsg("同步中…");
    try {
      const r = await api.postPricingSync(sourceIds);
      const failed = r.results.filter((s) => s.status === "error");
      const updated = r.results.reduce((sum, s) => sum + s.models, 0);
      const tail = failed.length > 0 ? `；${failed.length} 个源失败` : "";
      setSyncMsg(`✅ 已同步 ${updated} 个模型${tail}`);
      res.refetch();
    } catch (e) {
      setSyncMsg(`❌ 同步失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSyncing(false);
    }
  };
  const toggleAutoSync = async (enabled: boolean) => {
    try {
      const cfg = await api.getConfig();
      const priceSync = { ...((cfg.price_sync as Record<string, unknown>) ?? {}) };
      priceSync.auto_enabled = enabled;
      await api.postSaveConfig({ price_sync: priceSync });
      setAutoSync(enabled);
      setSyncMsg(enabled ? "已启用每日自动同步" : "已关闭每日自动同步");
    } catch (e) {
      setSyncMsg(`❌ 更新自动同步失败：${e instanceof Error ? e.message : String(e)}`);
    }
  };

  // 源开关：读最新 config → 改 price_sources → 保存 → 启用时立即拉该源
  const toggleSource = async (sourceId: string, enabled: boolean) => {
    try {
      const cfg = await api.getConfig();
      const ps = { ...((cfg.price_sources as Record<string, unknown>) ?? {}) };
      const entry = { ...((ps[sourceId] as Record<string, unknown>) ?? {}) };
      entry.enabled = enabled;
      if (sourceId.startsWith("newapi:") && !entry.provider_id) {
        entry.provider_id = sourceId.slice(7);
        entry.use_provider_key = true;
      }
      ps[sourceId] = entry;
      await api.postSaveConfig({ price_sources: ps });
      if (enabled) {
        await doSync([sourceId]);
      } else {
        res.refetch();
      }
    } catch (e) {
      setSyncMsg(`❌ 更新价格源失败：${e instanceof Error ? e.message : String(e)}`);
    }
  };

  // 探测并启用 New API 源（F1.2/F6.2）
  const enableNewApiSource = async (pid: string) => {
    setSelecting(true);
    setSyncMsg(`探测 ${pid} …`);
    try {
      const det = await api.postPricingDetect(pid);
      if (!det.is_newapi) {
        setSyncMsg(`❌ ${pid} 未检测到 New API /api/pricing 接口`);
        return;
      }
      const cfg = await api.getConfig();
      const ps = { ...((cfg.price_sources as Record<string, unknown>) ?? {}) };
      ps[`newapi:${pid}`] = {
        enabled: true,
        provider_id: pid,
        base_url: det.base_url ?? "",
        use_provider_key: true,
      };
      await api.postSaveConfig({ price_sources: ps });
      setSyncMsg(`✅ ${pid} 已启用为 New API 源，正在拉取价格…`);
      await doSync([`newapi:${pid}`]);
    } catch (e) {
      setSyncMsg(`❌ 探测失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSelecting(false);
    }
  };

  const disableNewApiSource = async (pid: string) => {
    await toggleSource(`newapi:${pid}`, false);
    setSyncMsg(`已停用 New API 源 ${pid}（已确认的选择不受影响）`);
  };

  const selectCandidate = async (pid: string, model: string, priceKey: string) => {
    setSelecting(true);
    try {
      await api.postPricingSelect({ provider_id: pid, model, price_key: priceKey });
      res.refetch();
    } catch (e) {
      setSyncMsg(`❌ 选用失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSelecting(false);
    }
  };

  const resetSelection = async (pid: string, model: string) => {
    setSelecting(true);
    try {
      await api.postPricingSelectReset({ provider_id: pid, model });
      res.refetch();
    } catch (e) {
      setSyncMsg(`❌ 清除失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSelecting(false);
    }
  };

  // provider 对应的 New API 源状态（有配置才显示开关行）
  const newApiSourceFor = (pid: string) => {
    const sid = `newapi:${pid}`;
    const st = sources[sid];
    if (st) return { sourceId: sid, enabled: !!st.enabled };
    return undefined;
  };

  if (res.loading && !data) return <Loading />;
  if (res.error) return <ErrorBox message={`加载定价失败：${res.error}`} />;

  const reset = async () => {
    // 两段式确认：首次点击武装，4 秒内再次点击执行（替代 confirm，兼容嵌入式 webview）
    if (!resetArmed) {
      setResetArmed(true);
      setResetResult("⚠ 再次点击以确认重置");
      if (resetArmTimer.current) clearTimeout(resetArmTimer.current);
      resetArmTimer.current = setTimeout(() => {
        setResetArmed(false);
        setResetResult("");
      }, 4000);
      return;
    }
    if (resetArmTimer.current) clearTimeout(resetArmTimer.current);
    setResetArmed(false);
    setResetResult("重置中…");
    try {
      await api.postSaveConfig({ pricing: {} });
      setResetResult("✅ 已重置，立即生效");
      res.refetch();
    } catch (e) {
      setResetResult(`❌ 重置失败：${e instanceof Error ? e.message : String(e)}`);
    }
  };

  // 点击未定价告警行 → 跳转到对应 provider 卡片
  const jumpToProvider = (pid: string) => {
    setHighlightTarget(pid);
    setHighlightSignal((s) => s + 1);
  };

  const defaultKeys = Object.keys(defaults).sort();

  // 统计概要数字
  const totalProviders = displayList.length;
  const unmatchedCount = displayList.filter(
    (p) => !p.matchedDefault && !hasSourceSelection(p.id),
  ).length;
  const candidateProviderCount = displayList.filter((p) => hasCandidates(p.id)).length;

  return (
    <div>
      <Panel className="source-panel">
        <h2>价格源</h2>
        <SourceStatusBar
          sources={sources}
          syncing={syncing}
          onSync={() => void doSync()}
          onToggleSource={(sid, en) => void toggleSource(sid, en)}
          autoSync={autoSync}
          onToggleAutoSync={(enabled) => void toggleAutoSync(enabled)}
        />
        {syncMsg && <div className="muted small source-sync-msg">{syncMsg}</div>}
      </Panel>

      {unpriced.length > 0 && (
        <Panel className="alert-panel">
          <h2>未定价告警（{unpricedByProvider.length} 个 Provider）</h2>
          <div className="alert-body">
            以下 Provider 有用量但无定价匹配，成本被计为 <strong>$0</strong>。
            点击行可快速跳转到对应 Provider 定价卡片。
          </div>
          <div className="unpriced-groups">
            {unpricedByProvider.map(([pid, group]) => {
              const isHistorical = !configIds.has(pid) && !drafts[pid];
              return (
                <div
                  key={pid}
                  className={`unpriced-group-row ${isHistorical ? "is-historical" : ""}`}
                  onClick={() => jumpToProvider(pid)}
                  title={isHistorical ? "该 Provider 已不在当前配置中，点击仍可设置定价" : "点击跳转到定价卡片"}
                >
                  <span className="mono unpriced-pid">{shortName(pid) || "(未知)"}</span>
                  {isHistorical && <span className="unpriced-historical-tag">历史</span>}
                  <span className="unpriced-models">
                    {group.models.length} 个模型
                  </span>
                  <span className="unpriced-tokens">
                    {fmtNum(group.totalTokens)} token
                  </span>
                  <span className="unpriced-jump-hint">点击跳转 ▸</span>
                </div>
              );
            })}
          </div>
        </Panel>
      )}

      <Panel>
        <div className="pricing-header">
          <h2>Provider 定价</h2>
          <div className="pricing-header-stats">
            <span className="muted small">{totalProviders} 个 Provider</span>
            {unmatchedCount > 0 && (
              <span className="pricing-unmatched-count">
                {unmatchedCount} 个无价格匹配
              </span>
            )}
            {candidateProviderCount > 0 && (
              <span className="pricing-unmatched-count">
                {candidateProviderCount} 个有待选候选
              </span>
            )}
          </div>
        </div>
        <div className="muted small" style={{ marginBottom: 8 }}>
          按 <strong>provider_id</strong> 设置定价。未设置时依次使用：已选价格源 → 自动匹配 → 内置默认。
          修改后自动保存、即时热生效，告警同步更新。
        </div>
        <div style={{ marginBottom: 8 }}>
          <Segmented
            options={FILTER_OPTIONS}
            value={filter}
            onChange={(v) => setFilter(v as PricingFilter)}
            variant="weak"
          />
        </div>
        {displayList.length === 0 && (
          <div className="muted small" style={{ margin: "8px 0" }}>
            未获取到当前 AstrBot 的 provider 配置。可在 AstrBot 主配置添加 provider 后重载插件。
          </div>
        )}
        <div className="overrides-list">
          {filteredList.map((p) => {
            return (
              <ProviderPricingCard
                key={p.id}
                providerId={p.id}
                displayId={p.displayId}
                type={p.type}
                candidates={p.candidates}
                draft={ensureDraft(p.id)}
                matchedDefault={p.matchedDefault}
                hasUserOverride={!isDraftEmpty(ensureDraft(p.id))}
                isHistorical={p.isHistorical}
                hasUsage={hasUnpricedUsage(p.id)}
                highlightSignal={
                  highlightTarget === p.id ? highlightSignal : undefined
                }
                selections={selections[p.id]}
                autoPriceKeys={autoSelected[p.id]}
                priceCandidatesByModel={candidateMap[p.id]}
                newApiSource={p.isHistorical ? undefined : newApiSourceFor(p.id)}
                selecting={selecting}
                onSelectCandidate={(m, pk) => selectCandidate(p.id, m, pk)}
                onResetSelection={(m) => resetSelection(p.id, m)}
                onEnableNewApiSource={
                  p.isHistorical ? undefined : () => enableNewApiSource(p.id)
                }
                onDisableNewApiSource={
                  p.isHistorical ? undefined : () => disableNewApiSource(p.id)
                }
                onChange={(patch) => updateDraft(p.id, patch)}
                onClear={() => clearDraft(p.id)}
              />
            );
          })}
          {filteredList.length === 0 && displayList.length > 0 && (
            <div className="muted small" style={{ margin: "8px 0" }}>
              当前筛选条件下没有 Provider。
            </div>
          )}
        </div>
        <div className="row" style={{ marginTop: 8, gap: 10, alignItems: "center" }}>
          <Button
            onClick={reset}
            title="清空自定义定价，恢复内置默认匹配"
            variant={resetArmed ? "danger" : "default"}
          >
            {resetArmed ? "⚠ 确认重置" : "重置全部"}
          </Button>
          <span className="muted">{resetResult}</span>
        </div>
      </Panel>

      {defaultKeys.length > 0 && (
        <details className="panel">
          <summary>
            内置默认单价（参考 OpenRouter，共 {defaultKeys.length} 个模型，per_token，只读）
          </summary>
          <div className="muted small" style={{ margin: "6px 0" }}>
            随插件版本更新；按模型名模糊匹配（前缀 / 子串），作为未设置 provider 定价时的回退基准。
          </div>
          <table>
            <thead>
              <tr>
                <th>模型</th>
                <th>输入</th>
                <th>缓存命中</th>
                <th>输出</th>
                <th>缓存写入</th>
              </tr>
            </thead>
            <tbody>
              {defaultKeys.map((k) => {
                const p = defaults[k] || {};
                return (
                  <tr key={k}>
                    <td className="mono">{k}</td>
                    <td>{p.input != null ? p.input : "-"}</td>
                    <td>{p.input_cached != null ? p.input_cached : "-"}</td>
                    <td>{p.output != null ? p.output : "-"}</td>
                    <td>{p.cache_creation != null ? p.cache_creation : "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </details>
      )}

      <SaveToast status={status} error={error} />
    </div>
  );
}
