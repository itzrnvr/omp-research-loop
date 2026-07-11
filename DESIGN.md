# omp-research-loop — Full Design Rationale

## The Problem

The user works on GPU kernel optimization (CUDA/Triton on an RTX 3070 Ti Laptop 8GB, sm_86) — llama.cpp tuning, the DSpark speculator, custom int4 engines. The iterative kernel-optimization process is **manual**: propose a strategy → implement → benchmark → analyze the failure → propose a better strategy → repeat. This is tedious, the findings are lost on context compaction, and there's no structured record of what was tried + what worked.

The user wants an **autonomous loop** that does this — a "step ladder" that iteratively improves — and **persists everything** so the findings survive across sessions + compactions. And it should be **generic** (not kernel-specific) — usable for prompt optimization, hyperparameter search, algorithm discovery, anything that follows a generate→implement→bench→analyze→iterate pattern.

## The Goals

1. **Autonomous iterative improvement** — generate strategies → analyze → implement → bench → analyze failure → next step. The loop runs itself; the user supplies the goal + the test + the winning criteria.
2. **Generic** — any goal + test + winning criteria. Not hardcoded for kernels. The extension is domain-agnostic.
3. **Enforced workflow** — the agent can't skip steps. Tool gates enforce the step order + the strategy count.
4. **Persistence that survives compaction** — an append-only git repo (NOTES.md, PROGRESS.md, bench.json). The agent reads them to recover state after a context compaction.
5. **The extension as the orchestrator** — the extension drives the loop (makes its own LLM calls for the review/guidance), the agent is the implementer (the "hands").
6. **A configurable model (like omp-supervisor)** — the extension has its own model for the orchestrator's LLM calls.
7. **Interactive setup** — infer the goal/test/criteria from the session + an ask-like wizard (confirm/edit). Minimal typing.
8. **A statusline widget** — the 🔄 Research · iter N/budget · step · best indicator (mirrors the omp-supervisor's footer).

## The Design Decisions (the reasoning)

### Why this flow: generate → synthesize → plan → work → submit → bench/review → loop

| Step | Why |
|---|---|
| **generate_strategies (N)** | Diversity — N distinct strategies per iteration increases the search space. The LLM proposes multiple approaches (not just one), each with a name, approach, expected gain, + risk. |
| **synthesize_strategy (ONE)** | Convergence — combine the best ideas from the N into one coherent final strategy. This avoids the agent picking a random strategy + ensures the best ideas are combined. |
| **create_plan** | Structure — a concrete implementation plan from the strategy. Grounds the strategy in actionable steps (the files to write, the approach, the steps). |
| **work (implement)** | The agent implements the plan. Open-ended (the agent uses its file tools). The extension doesn't control the implementation — the agent is the "hands." |
| **submit_work (agent-triggered)** | The agent decides when it's done implementing. This gives the agent autonomy over the implementation pace (it might take 1 turn or 10). |
| **bench + review + guidance (extension-run)** | The extension runs the bench (NOT the agent — so the agent can't fake a result). The extension evaluates the winning criteria + reviews the result via its own model + produces guidance for the next iteration. |
| **loop** | The next iteration's generate is informed by the guidance + NOTES.md. Self-improvement — each iteration builds on the prior findings. |

### Why tool gates (not a soft prompt)?

If the workflow is just a system prompt ("follow these steps"), the agent might:
- Skip steps (jump straight to "done").
- Not enforce N (submit 1 strategy instead of 5).
- Fake the bench result (claim it passed without running it).
- Decide to stop early ("this is good enough").

Tool gates **hard-enforce** the step order + the count. The agent must call `generate_strategies` (with N) → `synthesize_strategy` → `create_plan` → `submit_work` in order. The extension validates each gate (the current step). The extension runs the bench itself (on `submit_work`) — the agent can't fake a result. The extension controls the loop (advance/stop) — the agent can't decide to stop early.

### Why the extension-as-orchestrator (not the agent)?

The agent is good at the **open-ended work** (writing the kernel, implementing the strategy). The extension is good at the **structured workflow** (enforcing the steps, running the bench, recording findings, looping).

The extension makes its own LLM calls (via `Agent`/`AgentSession`, a stateless one-shot) for:
- **Inferring the task inputs** (goal/test/criteria) from the session.
- **Inferring the test command** from the test description + the repo context.
- **Reviewing the bench result** + producing guidance for the next iteration.
- **Evaluating a natural-language winning criteria** against the bench output.

These are focused, stateless calls (not the agent's full context) — cheaper + more targeted. The orchestrator model is configurable (`/research model`), separate from the agent's chat model.

### Why the append-only git repo (NOTES.md, PROGRESS.md, bench.json)?

The findings **survive compaction** — the agent reads NOTES.md + PROGRESS.md to recover the state after a context compaction. The markdown IS the long-term memory. The git repo version-controls the findings (each iteration is a commit). Cross-session recovery.

Append-only (never overwrite) — the history is preserved. The user can review what was tried + what worked at any time.

### Why for GLM-5.2?

- **The user's hardware**: RTX 3070 Ti Laptop 8GB (sm_86, CUDA). The user does GPU kernel work (llama.cpp tuning, DSpark speculator, int4 engine).
- **The user's default model**: `firepass-openai/accounts/fireworks/routers/glm-5p2-fast` (GLM-5.2 via Fireworks). The extension uses the session model (GLM-5.2) as the orchestrator model by default.
- **The KV caching**: the orchestrator's system prompt (the review/guidance prompt) is expanded to ≥1024 tokens for the provider's prefix cache (GLM-5.2 via Fireworks supports automatic prefix caching). The user-prompt tail (the bench output, the strategies) is where the changing content goes — zero cache impact from the changing parts. The cacheable prefix (system prompt + the stable user-prompt fields) stays valid across calls.
- **The mixed toolchain** (Triton for exploration, CUDA for max-perf) — the user's kernel work uses both. The extension is language-agnostic (the test command is whatever the user provides; the agent writes Triton OR CUDA per the strategy).
- **The cost**: GLM-5.2 (via Fireworks) is a fast, cheap model — suitable for the frequent orchestrator calls (the review/guidance on every iteration). The KV caching keeps the cost low (the system prompt is cached).

### GLM-5.2 — the strengths + weaknesses

#### Strengths
- Fast, cheap, good for the frequent orchestrator calls (the review/guidance on every iteration).
- KV caching (Fireworks prefix cache) — the system prompt (≥1024 tokens) is cached → the per-call cost is low.
- Sufficient for the orchestrator role (the review/guidance is a structured JSON analysis, not deep reasoning).
- The user's default model (the session model) — no extra config needed.

#### Weaknesses
- Weaker reasoning — the review/guidance might miss nuanced failure analysis (e.g. a subtle correctness bug, a perf bottleneck that requires deep architecture knowledge).
- The strategy generation might be less creative (fewer novel approaches).
- The natural-language criteria evaluation might be less precise.
- The model may hallucinate metrics or misjudge the bench output.
- The agent can sometimes talk the orchestrator into "done" (the anti-gaming rule in the system prompt mitigates this, but a weaker model is more susceptible).

### Why the interactive setup (infer + wizard)?

The user doesn't want to type everything. The extension:
1. **Infers** the goal/test/criteria from the session (the orchestrator model reads the session history + proposes them).
2. **Wizard** — an ask-like `ctx.ui.custom` overlay that shows the inferred inputs + asks to confirm (Enter) or cancel (Esc). Minimal typing.
3. **Inline args** (`/research --goal ... --test ... --win ...`) skip the wizard for scripting.

### Why the statusline widget?

The user wants to see the loop status (the iteration, the step, the best metric) at a glance — like the omp-supervisor's footer (🎯 ◉ Supervising · Goal · model · steers · action). The 🔄 Research · iter N/budget · step · best indicator mirrors the supervisor's. The widget updates at each transition (generate → synthesize → plan → work → bench → review → next iter).

### Why the model picker?

The user has 1581 models configured. `/research model` (no arg) opens an interactive picker (search + ↑/↓ + Enter) — the user can search + select the orchestrator model. The picker uses `matchesKey` (the omp-standard key matching) + a cached filtered list + a cached render (handles 1581 models without lag, even during streaming).

## The Research + Findings (the omp discoveries)

### The omp extension APIs (discovered during the build)
- `pi.registerCommand(name, { description, getArgumentCompletions?, handler })` — slash command.
- `getArgumentCompletions: (prefix) => Array<{value, label, description, hint?}> | null` — autocomplete (no ctx param — use a module-level `currentCtx` for model lists).
- `pi.registerTool({ name, parameters: z.object({...}), execute })` — `z = pi.zod`; `params` typed by the schema; return `{ content: [{type:"text", text}], details: undefined }`.
- `pi.on("session_start"|"session_switch"|"turn_start"|"turn_end"|"agent_end", async (event: unknown, ctx: ExtensionContext) => {})`.
- `pi.sendUserMessage(msg, { deliverAs: "steer" })` — steer the agent (kicks off an agent turn → queues later slash commands).
- `pi.appendEntry(customType, data)` → `session.sessionManager.appendCustomEntry` → a `{type:"custom", customType, data, id, parentId, timestamp}` entry (the omp-documented state-persistence API).
- `ctx.ui.notify(msg, "info"|"warning")` — toast (fades in ~1-2s).
- `ctx.ui.setStatus(id, emoji)` / `ctx.ui.setWidget(id, (tui, theme) => ({render, invalidate}))` — statusline badge + footer.
- `ctx.ui.custom<T>((tui, theme, kb, done) => ({render, invalidate, handleInput}))` → `Promise<T>` — interactive overlay. Cache the render output. Use `matchesKey` in handleInput.
- `ctx.modelRegistry.find(provider, modelId)` + `getApiKeyForProvider(provider)`; `ctx.model` (`.provider`, `.id`); `ctx.models.list()` (1500+ models).
- `ctx.sessionManager.getBranch()` — the session entries (includes the custom entries from appendEntry).
- The one-shot LLM call: `new Agent({getApiKey, initialState:{model, systemPrompt, tools:[], messages:[]}, streamFn})` + `new AgentSession({agent, sessionManager: SessionManager.inMemory(tempDir), settings: Settings.isolated({"compaction.enabled":false}), modelRegistry, toolRegistry: new Map()})` + `session.subscribe((e: AgentEvent) => ...)` + `await session.prompt(userPrompt)` + `session.dispose()` + `rmSync(tempDir)`.

### The gotchas (what didn't work + the fixes)
- **`matchesKey` (NOT raw `\u001b[A`/`\u001b[B`)** for the `ctx.ui.custom` handleInput. The raw escape sequences worked in the omp-supervisor's submenu context but NOT in a standalone widget. Fix: `import { matchesKey } from "@oh-my-pi/pi-tui";` + `matchesKey(data, "up"|"down"|"enter"|"escape"|"backspace"|...)`.
- **The module re-evaluation** — omp re-evaluates the extension module on session events (resetting module-level singletons). The in-memory state + a module-level singleton both didn't survive. The omp-documented way: `pi.appendEntry` + an always-run `loadFromSession` (matching the omp-supervisor). The file-based persistence is a fallback.
- **The `appendEntry` "0 found"** — a test artifact (checked the wrong session dir). The appendEntry DOES write a custom entry (to the in-memory getBranch + the JSONL on flush).
- **The `ModelSelectorComponent` crash** — theme undefined in the plugin context. Fix: `ctx.ui.select` or a custom `ctx.ui.custom` picker.
- **The `ctx.models` type error** — a pre-existing type error (not in the type defs), ignored at load.
- **The fire-and-forget tool execute** — the bench + the LLM review can exceed the ~30s handler timeout. Fix: `void runBenchAndAdvance(...).catch(...); return textResult("running…");` + deliver the result via `pi.sendUserMessage(..., {deliverAs:"steer"})`.
- **The render-output cache** — the `ctx.ui.custom` widget re-renders every frame during streaming. Fix: cache the render output (return cached `string[]` when state unchanged; invalidate on input).
- **The `pi.sendUserMessage` (steer) queues later commands** — the steer kicks off an agent turn → later slash commands are queued. Fix (testing): send the verification command immediately (sub-second) or C-c first.
- **The TS rules** (enforced by the harness): `ts-no-any` (use `unknown` + type guards), `ts-import-type` (separate `import type`), `ts-no-inline-cast-access` (use `in`/`typeof` OR named-const casts), `ts-no-dynamic-import` (static imports only).
- **The git commit** — a local repo may have no git identity. Fix: `ensureRepo` sets a default git identity.
- **The omp-supervisor anti-gaming** — the agent can talk the supervisor into "done" ("this is the limit" / "can't be improved"). Fix: the "done" criteria requires VERIFIABLE + COMPLETE achievement; the anti-gaming rule rejects the agent's claims; the stagnation is tightened (≥90% + a genuine effort).

### The psmux testing gotchas
- The welcome screen (wait for the MCP to connect before sending commands).
- The slow/hanging MCP (colab-mcp blocks the session for 30s+).
- The steer's agent turn (queues later commands).
- The toast fade (capture quickly).
- The throwaway cwd (don't touch the user's session).
- The session JSONL (check the right session dir; the entries are in the in-memory getBranch first).

## The Architecture (the files)

| File | Role |
|---|---|
| `index.ts` | The main entry — the `/research` command + the 4 tool gates + the session hooks + the autocomplete. |
| `loop-driver.ts` | The bench/review/advance (runs on `submit_work`, fire-and-forget). |
| `orchestrator.ts` | The infer/review/eval logic (the extension's own LLM calls). |
| `model-client.ts` | The one-shot LLM call (Agent/AgentSession, stateless, disposed after). |
| `wizard.ts` | The interactive setup (ctx.ui.custom — confirm/cancel). |
| `status-widget.ts` | The statusline widget (setStatus badge + setWidget footer). |
| `bench.ts` | The bench runner (run the test command + extract the metric via regex/jsonpath). |
| `loop-state.ts` | The step-state manager (+ session persistence via appendEntry). |
| `persistence.ts` | The append-only git repo (NOTES.md, PROGRESS.md, bench.json, git commits). |
| `workspace-config.ts` | The orchestrator model config (.omp/research-config.json). |
| `model-picker.ts` | The interactive model selector (ctx.ui.custom with search + cached render). |
| `types.ts` | The data model (LoopState, TaskConfig, WinningCriteria, Strategy, BenchResult). |
| `package.json` | The manifest (omp.extensions → index.ts). |

## The Data Flow

1. **`/research`** → infer (goal, test, criteria) from the session → wizard (confirm) → create_test (infer the test command) → start the loop (ensureRepo, NOTES.md, git commit, nextIteration, updateUI, steer the agent to generate strategies).
2. **The loop** (each iteration):
   - `generate_strategies` (N strategies) → the extension validates N + records to NOTES.md → advance to synthesize.
   - `synthesize_strategy` (ONE final strategy) → record → advance to plan.
   - `create_plan` (the implementation plan) → record → advance to work.
   - **work** (the agent implements) → `submit_work` (the agent calls this when ready).
   - **bench** (the extension runs the test command) → **evaluate** (the winning criteria — metric parse OR the orchestrator model) → **review** (the orchestrator model produces guidance) → **record** (NOTES.md, bench.json, PROGRESS.md, git commit) → **return the details to the agent** (steer).
   - **advance** (nextIteration → generate) OR **stop** (criteria met / budget exhausted).
3. **Compaction recovery** — the agent reads PROGRESS.md + the task's NOTES.md to recover the state (the findings survive).

## The Future (the follow-ups)

1. **The state persistence** — the `appendEntry` + the always-run `loadFromSession` (matching the omp-supervisor). The `if (this.state) return;` guard + the singleton should be reverted. The file-based fallback is a last resort.
2. **The agent-created test** (the `create_test` full freedom) — a `submit_test` gate (the agent creates the bench script + submits the command).
3. **The wizard edit** — allow editing each field (goal/test/criteria) inline (not just confirm/cancel).
4. **The settings panel** — a mid-loop overlay (like the omp-supervisor's) to view/edit the goal/test/criteria/model/N/budget.
5. **The full end-to-end loop test** — the agent calling the tool gates → bench → review → loop (blocked by the state persistence + the agent turn from the steer).
6. **The KV-cache `cache_control`** — for Anthropic (the 1-hour TTL), if the orchestrator model is Anthropic. The omp pi-ai supports `cache_control`/`cacheRetention:"long"` via `buildAnthropicSystemBlocks(...,{cacheControl})`.
7. **The metric "best" tracking** — the `recordBench` should track the best according to the comparator (≥/→ max, ≤/→ min), not just the max.
8. **The statusline enhancements** — add the current strategy count + the last bench result to the footer widget.

## The omp-supervisor findings (the companion extension)

The omp-supervisor (the companion extension) taught us the patterns + the gotchas:
- The `convertToLlm` removal (omp v16.3.11).
- The `ModelSelectorComponent` crash (theme undefined in the plugin context).
- The `tool_result` event shape (`event.content`, not `event.output.content`).
- The `pi.sendMessage` content (string, not array).
- The bare-keyword guard (`/supervise high` → the goal became "high").
- The model picker cache (the double requestRender + the 1581-model filter allocation).
- The model-update `updateValue` (the explicit update + the forward declaration).
- The system prompt expansion for KV caching (603 → 1428 tokens).
- The `/supervise add` feature (the tail placement for KV caching).
- The `.pi`→`.omp` porting leftover.
- The supervisor "does nothing" = silent model-call failure (the deepseek 401).
- The anti-gaming (the agent talks the supervisor into "done").
- The persistent "done" indicator (the ✅ Supervision complete widget).
- The `matchesKey` key matching (the omp-standard).
- The `appendEntry` state persistence (the omp-documented way).
- The module re-evaluation (omp re-evaluates the extension module on session events).
- The TS rules (no-any, import-type, no-inline-cast-access, no-dynamic-import).
- The psmux testing gotchas.

These findings are captured in the `building-omp-extensions` managed skill + the FINDINGS.md in both repos.
