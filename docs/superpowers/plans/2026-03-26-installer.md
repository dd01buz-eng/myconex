# MYCONEX Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross-platform installer (`install.sh` + `install.ps1`) that sets up any machine as a hub, full mesh node, or lightweight Ollama-only node, with TUI/SSH/unattended modes and fleet answer-file support.

**Architecture:** `install.sh` is the canonical entry point; all logic lives in sourced library files under `install/lib/`. `install.ps1` is a thin Windows wrapper that detects WSL and delegates to `install.sh` inside it, or runs a native PowerShell fallback. Each install step is an idempotent function in its own file under `install/lib/steps/`.

**Tech Stack:** bash 4+, bats-core (testing), whiptail (TUI), PowerShell 5+ (Windows wrapper), yq (YAML parsing in shell), docker compose v2, systemd, ollama CLI.

---

## File Map

| File | Create / Modify | Purpose |
|---|---|---|
| `install.sh` | Create | Main entry point — parses flags, sources libs, runs flow |
| `install.ps1` | Create | Windows wrapper — WSL detection + native PS fallback |
| `install/lib/ui.sh` | Create | TUI abstraction: ui_menu, ui_checklist, ui_input, ui_confirm |
| `install/lib/detect.sh` | Create | OS, package manager, hardware tier, GPU, Docker, display/SSH |
| `install/lib/profiles.sh` | Create | Component matrix, tier→model mapping, step sequencer |
| `install/lib/steps/core.sh` | Create | Python 3.11+, pip deps, git submodules |
| `install/lib/steps/hub_services.sh` | Create | Docker install + docker compose --profile full |
| `install/lib/steps/llm.sh` | Create | Ollama install + model pull by tier |
| `install/lib/steps/gpu.sh` | Create | NVIDIA Container Toolkit |
| `install/lib/steps/hermes.sh` | Create | hermes-agent + flash-moe submodule init |
| `install/lib/steps/registration_agent.sh` | Create | Lightweight node: Ollama + announce agent |
| `install/lib/steps/discord.sh` | Create | discord.py deps + token collection |
| `install/lib/steps/integrations.sh` | Create | Gmail/YouTube/RSS/podcast deps + API keys |
| `install/lib/steps/dashboard.sh` | Create | Dashboard deps |
| `install/lib/steps/config.sh` | Create | Writes config/node.yaml + .env |
| `install/lib/steps/systemd.sh` | Create | Generates + installs systemd units (or start.sh fallback) |
| `install/answers.yaml.example` | Create | Documented answer file template |
| `core/classifier/hardware.py` | Modify | Sync TIER_DEFINITIONS models to match mesh_config.yaml |
| `tests/installer/test_helper.bash` | Create | Shared bats helpers + command mocks |
| `tests/installer/test_detect.bats` | Create | Tests for detect.sh functions |
| `tests/installer/test_ui.bats` | Create | Tests for ui.sh backend selection |
| `tests/installer/test_profiles.bats` | Create | Tests for component matrix lookups |
| `tests/installer/test_config.bats` | Create | Tests for config.sh file generation |
| `tests/installer/test_install_flags.bats` | Create | Tests for install.sh CLI flag parsing |

---

## Task 0: Sync hardware.py tier models (prerequisite)

**Files:**
- Modify: `core/classifier/hardware.py:74-99`

The spec calls this a prerequisite. `hardware.py:TIER_DEFINITIONS` still has old models (llama3.1, llama3.2, phi3). `mesh_config.yaml` was updated 2026-03-24 to qwen3. Update `hardware.py` to match.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hardware_tiers.py`:
```python
import ast, pathlib

def _load_tier_defs():
    src = pathlib.Path("core/classifier/hardware.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "TIER_DEFINITIONS":
                    return ast.literal_eval(node.value)
    raise ValueError("TIER_DEFINITIONS not found")

def test_tier_models_match_mesh_config():
    defs = _load_tier_defs()
    assert defs["T1"]["ollama_model"] == "llama3.1:70b"
    assert defs["T2"]["ollama_model"] == "qwen3:8b"
    assert defs["T3"]["ollama_model"] == "qwen3:4b"
    assert defs["T4"]["ollama_model"] == "qwen3:0.6b"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_hardware_tiers.py -v
```
Expected: FAIL — T2/T3/T4 show old model names.

- [ ] **Step 3: Update TIER_DEFINITIONS in hardware.py**

In `core/classifier/hardware.py`, lines 81–98, change:
```python
    "T2": {
        "label": "Mid-GPU",
        "roles": ["medium-model", "inference", "embedding", "fine-tuning"],
        "description": "Mid-range GPU node with 8–24 GB VRAM. Runs 7B–30B models.",
        "ollama_model": "qwen3:8b",
    },
    "T3": {
        "label": "CPU-Heavy",
        "roles": ["orchestration", "embedding", "lightweight-inference", "relay"],
        "description": "High-core-count CPU or large RAM without significant GPU.",
        "ollama_model": "qwen3:4b",
    },
    "T4": {
        "label": "Edge / Embedded",
        "roles": ["sensor", "relay", "lightweight-inference"],
        "description": "Raspberry Pi or low-resource node. Minimal inference only.",
        "ollama_model": "qwen3:0.6b",
    },
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_hardware_tiers.py -v
```
Expected: PASS

- [ ] **Step 5: Record decision and commit**

Append to `docs/DECISIONS.md`:
```markdown
## Sync hardware.py tier models to qwen3
**Decision:** Updated TIER_DEFINITIONS in hardware.py to use qwen3 models (T2: qwen3:8b, T3: qwen3:4b, T4: qwen3:0.6b).
**Why:** mesh_config.yaml was updated 2026-03-24 but hardware.py was not synced. Installer reads hardware.py as the single source of truth.
**Alternatives:** Keep old models; rejected because mesh_config.yaml explicitly updated them.
**Consequences:** Nodes installing fresh will pull qwen3 models, not llama3.1/phi3.
```

```bash
git add core/classifier/hardware.py docs/DECISIONS.md tests/test_hardware_tiers.py
git commit -m "fix: sync hardware.py tier models to qwen3, matching mesh_config.yaml"
```

---

## Task 1: Test infrastructure scaffold

**Files:**
- Create: `tests/installer/test_helper.bash`
- Create: `tests/installer/.gitkeep`

bats-core is used for all shell tests. Install it as a git submodule so tests are self-contained.

- [ ] **Step 1: Add bats-core as submodule**

```bash
git submodule add https://github.com/bats-core/bats-core tests/lib/bats-core
git submodule add https://github.com/bats-core/bats-support tests/lib/bats-support
git submodule add https://github.com/bats-core/bats-assert tests/lib/bats-assert
```

- [ ] **Step 2: Create test helper**

Create `tests/installer/test_helper.bash`:
```bash
# Load bats helpers
load '../lib/bats-support/load'
load '../lib/bats-assert/load'

# Repo root (two levels up from tests/installer/)
REPO_ROOT="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"

# Source a lib file from the installer
source_lib() {
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/install/lib/${1}"
}

# Mock a command: mock_cmd <name> <exit_code> [stdout]
# Creates a temporary function override visible in the test scope
mock_cmd() {
    local name="$1" exit_code="$2" output="${3:-}"
    eval "${name}() { echo \"${output}\"; return ${exit_code}; }"
    export -f "${name}"
}
```

- [ ] **Step 3: Verify bats runs**

```bash
tests/lib/bats-core/bin/bats --version
```
Expected: `Bats 1.x.x`

- [ ] **Step 4: Commit scaffold**

```bash
git add tests/installer/ tests/lib/
git commit -m "chore: add bats-core test infrastructure for installer"
```

---

## Task 2: detect.sh — OS and package manager detection

**Files:**
- Create: `install/lib/detect.sh`
- Create: `tests/installer/test_detect.bats`

- [ ] **Step 1: Write failing tests**

Create `tests/installer/test_detect.bats`:
```bash
#!/usr/bin/env bats
load 'test_helper'

setup() { source_lib "detect.sh"; }

@test "detect_pkg_manager returns apt on Ubuntu" {
    mock_cmd apt-get 0
    run detect_pkg_manager
    assert_success
    assert_output "apt"
}

@test "detect_pkg_manager returns dnf on Fedora" {
    apt-get() { return 1; }; export -f apt-get
    mock_cmd dnf 0
    run detect_pkg_manager
    assert_success
    assert_output "dnf"
}

@test "detect_pkg_manager returns pacman on Arch" {
    apt-get() { return 1; }; export -f apt-get
    dnf() { return 1; }; export -f dnf
    mock_cmd pacman 0
    run detect_pkg_manager
    assert_success
    assert_output "pacman"
}

@test "detect_wsl returns true inside WSL" {
    MYCONEX_TEST_KERNEL="Linux-5.15-microsoft-standard-WSL2"
    run detect_wsl
    assert_success
}

@test "detect_wsl returns false outside WSL" {
    MYCONEX_TEST_KERNEL="Linux-6.1.0-generic"
    run detect_wsl
    assert_failure
}

@test "detect_display returns tui when DISPLAY is set" {
    DISPLAY=":0"
    mock_cmd whiptail 0
    run detect_display_mode
    assert_output "tui"
}

@test "detect_display returns plain over SSH" {
    unset DISPLAY
    SSH_CONNECTION="10.0.0.1 12345 10.0.0.2 22"
    run detect_display_mode
    assert_output "plain"
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_detect.bats
```
Expected: all fail with "source_lib: detect.sh not found"

- [ ] **Step 3: Create detect.sh**

Create `install/lib/detect.sh`:
```bash
#!/usr/bin/env bash
# detect.sh — OS, hardware, and environment detection

# --- Package manager ---
detect_pkg_manager() {
    if command -v apt-get &>/dev/null; then echo "apt"
    elif command -v dnf &>/dev/null;     then echo "dnf"
    elif command -v pacman &>/dev/null;  then echo "pacman"
    elif command -v apk &>/dev/null;     then echo "apk"
    else echo "unknown"; return 1
    fi
}

# --- WSL detection ---
# Testable via MYCONEX_TEST_KERNEL override
detect_wsl() {
    local kernel="${MYCONEX_TEST_KERNEL:-$(uname -r)}"
    [[ "$kernel" == *microsoft* ]]
}

# --- Display / UI mode ---
detect_display_mode() {
    # --no-tui and --unattended are handled before this is called
    if [[ -n "${SSH_CONNECTION:-}" ]] || [[ -z "${DISPLAY:-}" ]]; then
        echo "plain"
    elif command -v whiptail &>/dev/null; then
        echo "tui"
    else
        echo "plain"
    fi
}

# --- Python version check ---
detect_python() {
    local py
    for py in python3.11 python3.12 python3; do
        if command -v "$py" &>/dev/null; then
            local ver; ver=$("$py" -c "import sys; print(sys.version_info[:2])")
            if [[ "$ver" > "(3, 10)" ]]; then echo "$py"; return 0; fi
        fi
    done
    return 1
}

# --- Docker check ---
detect_docker() {
    command -v docker &>/dev/null && docker info &>/dev/null
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_detect.bats
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add install/lib/detect.sh tests/installer/test_detect.bats
git commit -m "feat(installer): detect.sh — OS, package manager, WSL, display mode"
```

---

## Task 3: detect.sh — Hardware tier classification

**Files:**
- Modify: `install/lib/detect.sh`
- Modify: `tests/installer/test_detect.bats`

- [ ] **Step 1: Add failing tier tests**

Append to `tests/installer/test_detect.bats`:
```bash
@test "detect_tier returns T1 for >24GB VRAM" {
    # nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits outputs a bare integer
    mock_cmd nvidia-smi 0 "24564"
    run detect_tier
    assert_output "T1"
}

@test "detect_tier returns T2 for 8-24GB VRAM" {
    mock_cmd nvidia-smi 0 "12288"
    run detect_tier
    assert_output "T2"
}

@test "detect_tier returns T3 for no GPU, >8 cores" {
    nvidia-smi() { return 1; }; export -f nvidia-smi
    MYCONEX_TEST_CPU_CORES=20
    MYCONEX_TEST_RAM_GB=32
    run detect_tier
    assert_output "T3"
}

@test "detect_tier returns T4 for no GPU, <=4 cores" {
    nvidia-smi() { return 1; }; export -f nvidia-smi
    MYCONEX_TEST_CPU_CORES=4
    MYCONEX_TEST_RAM_GB=4
    run detect_tier
    assert_output "T4"
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_detect.bats
```
Expected: new tests fail with "detect_tier: command not found"

- [ ] **Step 3: Add detect_tier to detect.sh**

Append to `install/lib/detect.sh`:
```bash
# --- GPU VRAM (MB) via nvidia-smi ---
_detect_gpu_vram_mb() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d '[:space:]'
}

# --- CPU core count (testable override via MYCONEX_TEST_CPU_CORES) ---
_detect_cpu_cores() {
    echo "${MYCONEX_TEST_CPU_CORES:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"
}

# --- RAM in GB (testable override via MYCONEX_TEST_RAM_GB) ---
_detect_ram_gb() {
    if [[ -n "${MYCONEX_TEST_RAM_GB:-}" ]]; then echo "$MYCONEX_TEST_RAM_GB"; return; fi
    local kb; kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
    echo $(( ${kb:-0} / 1024 / 1024 ))
}

# --- Tier classification ---
detect_tier() {
    local vram_mb; vram_mb=$(_detect_gpu_vram_mb)
    if [[ -n "$vram_mb" && "$vram_mb" -gt 0 ]]; then
        if   [[ "$vram_mb" -ge 24000 ]]; then echo "T1"
        elif [[ "$vram_mb" -ge 7000  ]]; then echo "T2"
        else                                  echo "T3"
        fi
        return
    fi
    # CPU-only path
    local cores ram
    cores=$(_detect_cpu_cores)
    ram=$(_detect_ram_gb)
    if [[ "$cores" -ge 8 && "$ram" -ge 16 ]]; then echo "T3"
    else                                           echo "T4"
    fi
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_detect.bats
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add install/lib/detect.sh tests/installer/test_detect.bats
git commit -m "feat(installer): detect_tier — T1-T4 hardware classification"
```

---

## Task 4: ui.sh — Interface abstraction

**Files:**
- Create: `install/lib/ui.sh`
- Create: `tests/installer/test_ui.bats`

- [ ] **Step 1: Write failing tests**

Create `tests/installer/test_ui.bats`:
```bash
#!/usr/bin/env bats
load 'test_helper'

setup() {
    source_lib "ui.sh"
    export UI_MODE="plain"   # default to plain in tests
}

@test "ui_confirm returns 0 on 'y' input" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=plain
        echo y | ui_confirm 'Proceed?'
    "
    assert_success
}

@test "ui_confirm returns 1 on 'n' input" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=plain
        echo n | ui_confirm 'Proceed?'
    "
    assert_failure
}

@test "ui_confirm in unattended mode returns 0 (auto-yes)" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=unattended
        ui_confirm 'Proceed?'
    "
    assert_success
}

@test "ui_input in unattended mode reads from MYCONEX_UNATTENDED_VALUES" {
    run bash -c "
        source '${REPO_ROOT}/install/lib/ui.sh'
        UI_MODE=unattended
        MYCONEX_VAL_NODE_NAME='test-hub'
        ui_input 'Node name:' NODE_NAME
        echo \"\$NODE_NAME\"
    "
    assert_output "test-hub"
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_ui.bats
```
Expected: fail

- [ ] **Step 3: Create ui.sh**

Create `install/lib/ui.sh`:
```bash
#!/usr/bin/env bash
# ui.sh — TUI abstraction layer
# UI_MODE: tui | plain | unattended  (set by install.sh before sourcing)

# Single-choice menu. Sets REPLY to chosen value.
# Usage: ui_menu "Title" "item1" "item2" ...
ui_menu() {
    local title="$1"; shift
    local items=("$@")
    case "${UI_MODE:-plain}" in
        tui)
            local menu_items=()
            for i in "${!items[@]}"; do menu_items+=("$i" "${items[$i]}"); done
            REPLY=$(whiptail --title "$title" --menu "" 15 60 "${#items[@]}" \
                "${menu_items[@]}" 3>&1 1>&2 2>&3)
            REPLY="${items[$REPLY]}"
            ;;
        unattended)
            REPLY="${items[0]}"  # first option; override via answer file
            ;;
        *)
            echo "$title"
            for i in "${!items[@]}"; do echo "  $((i+1))) ${items[$i]}"; done
            read -r -p "Choice [1]: " choice
            REPLY="${items[$(( ${choice:-1} - 1 ))]}"
            ;;
    esac
}

# Multi-select checklist. Sets SELECTED as space-separated values.
# Usage: ui_checklist "Title" "item1:on" "item2:off" ...
ui_checklist() {
    local title="$1"; shift
    SELECTED=()
    case "${UI_MODE:-plain}" in
        tui)
            local check_items=()
            for item in "$@"; do
                local name="${item%%:*}" state="${item##*:}"
                check_items+=("$name" "" "$state")
            done
            local result
            result=$(whiptail --title "$title" --checklist "" 20 60 "${#@}" \
                "${check_items[@]}" 3>&1 1>&2 2>&3)
            # shellcheck disable=SC2206
            IFS=' ' read -r -a SELECTED <<< "${result//\"/}"
            ;;
        unattended)
            for item in "$@"; do
                [[ "${item##*:}" == "on" ]] && SELECTED+=("${item%%:*}")
            done
            ;;
        *)
            echo "$title (space to toggle, enter to confirm)"
            for item in "$@"; do
                local name="${item%%:*}" state="${item##*:}"
                read -r -p "  Include ${name}? [${state}]: " ans
                ans="${ans:-$state}"
                [[ "$ans" =~ ^(on|yes|y|1)$ ]] && SELECTED+=("$name")
            done
            ;;
    esac
}

# Text / secret input. Stores result in named variable.
# Usage: ui_input "Prompt" VAR_NAME [secret]
ui_input() {
    local prompt="$1" varname="$2" secret="${3:-}"
    # Unattended: read from MYCONEX_VAL_<VARNAME>
    if [[ "${UI_MODE:-plain}" == "unattended" ]]; then
        local envkey="MYCONEX_VAL_${varname}"
        printf -v "$varname" '%s' "${!envkey:-}"
        return
    fi
    local val
    if [[ -n "$secret" ]]; then
        read -r -s -p "$prompt " val; echo
    else
        read -r -p "$prompt " val
    fi
    printf -v "$varname" '%s' "$val"
}

# Yes/no confirmation. Returns 0=yes, 1=no.
# Usage: ui_confirm "Proceed?"
ui_confirm() {
    local prompt="$1"
    case "${UI_MODE:-plain}" in
        tui)
            whiptail --yesno "$prompt" 8 40
            ;;
        unattended)
            return 0
            ;;
        *)
            read -r -p "$prompt [Y/n]: " ans
            [[ "${ans:-y}" =~ ^[Yy] ]]
            ;;
    esac
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_ui.bats
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add install/lib/ui.sh tests/installer/test_ui.bats
git commit -m "feat(installer): ui.sh — tui/plain/unattended abstraction"
```

---

## Task 5: profiles.sh — Component matrix and tier→model mapping

**Files:**
- Create: `install/lib/profiles.sh`
- Create: `tests/installer/test_profiles.bats`

The tier→model mapping in bash must stay in sync with `hardware.py:TIER_DEFINITIONS`.

- [ ] **Step 1: Write failing tests**

Create `tests/installer/test_profiles.bats`:
```bash
#!/usr/bin/env bats
load 'test_helper'

setup() { source_lib "profiles.sh"; }

@test "hub requires core" {
    run profile_requires "hub" "core"
    assert_success
}

@test "hub requires hub_services" {
    run profile_requires "hub" "hub_services"
    assert_success
}

@test "lightweight skips hub_services" {
    run profile_requires "lightweight" "hub_services"
    assert_failure
}

@test "T4 full-node skips hermes_moe" {
    run profile_requires_for_tier "full-node" "T4" "hermes_moe"
    assert_failure
}

@test "T2 full-node requires hermes_moe" {
    run profile_requires_for_tier "full-node" "T2" "hermes_moe"
    assert_success
}

@test "model_for_tier T2 returns qwen3:8b" {
    run model_for_tier "T2"
    assert_output "qwen3:8b"
}

@test "model_for_tier T4 returns qwen3:0.6b" {
    run model_for_tier "T4"
    assert_output "qwen3:0.6b"
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_profiles.bats
```
Expected: fail

- [ ] **Step 3: Create profiles.sh**

Create `install/lib/profiles.sh`:
```bash
#!/usr/bin/env bash
# profiles.sh — component matrix and tier→model mapping
# Must stay in sync with core/classifier/hardware.py:TIER_DEFINITIONS

# Tier → Ollama model (mirrors hardware.py TIER_DEFINITIONS)
declare -A _TIER_MODELS=(
    [T1]="llama3.1:70b"
    [T2]="qwen3:8b"
    [T3]="qwen3:4b"
    [T4]="qwen3:0.6b"
)

model_for_tier() { echo "${_TIER_MODELS[${1:?}]}"; }

# Hub mandatory components
declare -A _HUB_REQUIRED=(
    [core]=1 [hub_services]=1 [llm_backends]=1
    [gpu]=1  [hermes_moe]=1   [systemd]=1 [ollama]=1
)
# Hub optional components
declare -A _HUB_OPTIONAL=([discord]=1 [integrations]=1 [dashboard]=1)

# Full-node per tier: lists components that ARE required (not skipped)
# gpu and hermes_moe are absent for T3/T4
declare -A _NODE_T1=([core]=1 [llm_backends]=1 [gpu]=1 [hermes_moe]=1 [systemd]=1 [ollama]=1)
declare -A _NODE_T2=([core]=1 [llm_backends]=1 [gpu]=1 [hermes_moe]=1 [systemd]=1 [ollama]=1)
declare -A _NODE_T3=([core]=1 [llm_backends]=1 [systemd]=1 [ollama]=1)
declare -A _NODE_T4=([core]=1 [systemd]=1 [ollama]=1)
declare -A _NODE_OPT=([discord]=1 [integrations]=1 [dashboard]=1)

# Lightweight
declare -A _LIGHT_REQUIRED=([core]=1 [registration_agent]=1 [systemd]=1 [ollama]=1)

# profile_requires <role> <component>   → 0 if required, 1 if not
profile_requires() {
    local role="$1" comp="$2"
    case "$role" in
        hub)         [[ -n "${_HUB_REQUIRED[$comp]:-}" ]] ;;
        lightweight) [[ -n "${_LIGHT_REQUIRED[$comp]:-}" ]] ;;
        full-node)   [[ -n "${_NODE_T2[$comp]:-}" ]]  # default T2; use _for_tier variant
            ;;
    esac
}

# profile_requires_for_tier <role> <tier> <component>
profile_requires_for_tier() {
    local role="$1" tier="$2" comp="$3"
    [[ "$role" != "full-node" ]] && { profile_requires "$role" "$comp"; return; }
    local -n _ref="_NODE_${tier}"
    [[ -n "${_ref[$comp]:-}" ]]
}

# profile_optional <role> <component>   → 0 if optional, 1 if not
profile_optional() {
    local role="$1" comp="$2"
    case "$role" in
        hub|full-node) [[ -n "${_HUB_OPTIONAL[$comp]:-}" ]] ;;
        *)             return 1 ;;
    esac
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_profiles.bats
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add install/lib/profiles.sh tests/installer/test_profiles.bats
git commit -m "feat(installer): profiles.sh — component matrix, tier→model mapping"
```

---

## Task 6: steps/core.sh — Python, pip, submodules

**Files:**
- Create: `install/lib/steps/core.sh`

- [ ] **Step 1: Create core.sh**

Create `install/lib/steps/core.sh`:
```bash
#!/usr/bin/env bash
# core.sh — Python 3.11+, pip deps, git submodules
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"

STEP_NAME="core"
SENTINEL="${HOME}/.myconex/.installed_core"

step_core() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[core] already installed, skipping"; return 0; }

    log_step "Installing core: Python, pip deps, submodules"

    # 1. Python 3.11+
    local py; py=$(detect_python || true)
    if [[ -z "$py" ]]; then
        local pkg; pkg=$(detect_pkg_manager)
        log_step "Installing Python 3.11"
        case "$pkg" in
            apt)    sudo apt-get install -y python3.11 python3.11-venv python3-pip ;;
            dnf)    sudo dnf install -y python3.11 ;;
            pacman) sudo pacman -Sy --noconfirm python ;;
            apk)    sudo apk add python3 py3-pip ;;
        esac
        py=$(detect_python)
    fi
    log_step "Python: $py"

    # 2. pip deps from project root requirements.txt
    local req="${MYCONEX_REPO_ROOT:?}/requirements.txt"
    if [[ "${REINSTALL:-}" ]]; then
        "$py" -m pip install --upgrade -r "$req"
    else
        "$py" -m pip install -r "$req"
    fi

    # 3. Git submodules (hermes-agent, flash-moe)
    git -C "$MYCONEX_REPO_ROOT" submodule update --init --recursive

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "core: done"
}
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n install/lib/steps/core.sh
```
Expected: no output (no syntax errors)

- [ ] **Step 3: Commit**

```bash
git add install/lib/steps/core.sh
git commit -m "feat(installer): steps/core.sh — Python, pip, submodules"
```

---

## Task 7: steps/hub_services.sh — Docker + Compose stack

**Files:**
- Create: `install/lib/steps/hub_services.sh`

- [ ] **Step 1: Create hub_services.sh**

Create `install/lib/steps/hub_services.sh`:
```bash
#!/usr/bin/env bash
# hub_services.sh — Docker install + docker compose --profile full
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"

SENTINEL="${HOME}/.myconex/.installed_hub_services"

step_hub_services() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[hub_services] already installed"; return 0; }

    log_step "Installing hub services (Docker + compose stack)"

    # Install Docker if missing
    if ! detect_docker; then
        log_step "Installing Docker"
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "$USER"
        log_step "Docker installed. You may need to log out and back in for group changes."
    fi

    # On reinstall: pull latest images
    if [[ -f "$SENTINEL" ]]; then
        log_step "Pulling latest images"
        docker compose -f "${MYCONEX_REPO_ROOT}/services/docker-compose.yml" \
            --profile full pull
    fi

    log_step "Starting hub services"
    docker compose -f "${MYCONEX_REPO_ROOT}/services/docker-compose.yml" \
        --profile full up -d

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "hub_services: done"
}
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n install/lib/steps/hub_services.sh
```
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add install/lib/steps/hub_services.sh
git commit -m "feat(installer): steps/hub_services.sh — Docker + compose stack"
```

---

## Task 8: steps/llm.sh — Ollama install and model pull

**Files:**
- Create: `install/lib/steps/llm.sh`

- [ ] **Step 1: Create llm.sh**

Create `install/lib/steps/llm.sh`:
```bash
#!/usr/bin/env bash
# llm.sh — Ollama install + model pull by tier
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../profiles.sh"

SENTINEL="${HOME}/.myconex/.installed_llm"

step_llm() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[llm] already installed"; return 0; }

    local tier="${MYCONEX_DETECTED_TIER:?tier not set}"
    local model="${MYCONEX_OLLAMA_MODEL:-$(model_for_tier "$tier")}"

    log_step "Installing LLM backend (Ollama, model: $model)"

    # Install Ollama if missing
    if ! command -v ollama &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # Pull model (idempotent — ollama pull is a no-op if already present)
    log_step "Pulling $model (this may take a while)"
    ollama pull "$model"

    # Always pull embedding model
    ollama pull nomic-embed-text

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "llm: done"
}
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n install/lib/steps/llm.sh
```
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add install/lib/steps/llm.sh
git commit -m "feat(installer): steps/llm.sh — Ollama install and tier model pull"
```

---

## Task 9: steps/gpu.sh — NVIDIA Container Toolkit

**Files:**
- Create: `install/lib/steps/gpu.sh`

- [ ] **Step 1: Create gpu.sh**

Create `install/lib/steps/gpu.sh`:
```bash
#!/usr/bin/env bash
# gpu.sh — NVIDIA Container Toolkit (skipped if no NVIDIA GPU)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../detect.sh"

SENTINEL="${HOME}/.myconex/.installed_gpu"

step_gpu() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && { echo "[gpu] already installed"; return 0; }

    # Auto-skip if no NVIDIA GPU
    if ! command -v nvidia-smi &>/dev/null; then
        log_step "No NVIDIA GPU detected — skipping GPU setup"
        touch "$SENTINEL"
        return 0
    fi

    log_step "Installing NVIDIA Container Toolkit"
    local pkg; pkg=$(detect_pkg_manager)
    case "$pkg" in
        apt)
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update -q
            sudo apt-get install -y nvidia-container-toolkit
            ;;
        dnf)
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
                | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            sudo dnf install -y nvidia-container-toolkit
            ;;
        *)
            log_step "WARNING: cannot auto-install NVIDIA toolkit for package manager: $pkg"
            log_step "Install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            return 0
            ;;
    esac

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    mkdir -p "$(dirname "$SENTINEL")"
    touch "$SENTINEL"
    log_step "gpu: done"
}
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n install/lib/steps/gpu.sh
```
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add install/lib/steps/gpu.sh
git commit -m "feat(installer): steps/gpu.sh — NVIDIA Container Toolkit"
```

---

## Task 10: Remaining step scripts (hermes, registration_agent, discord, integrations, dashboard)

**Files:**
- Create: `install/lib/steps/hermes.sh`
- Create: `install/lib/steps/registration_agent.sh`
- Create: `install/lib/steps/discord.sh`
- Create: `install/lib/steps/integrations.sh`
- Create: `install/lib/steps/dashboard.sh`

These five steps follow the same sentinel-guard + idempotent pattern.

- [ ] **Step 1: Create hermes.sh**

Create `install/lib/steps/hermes.sh`:
```bash
#!/usr/bin/env bash
# hermes.sh — hermes-agent + flash-moe submodules
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_hermes"

step_hermes() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Initialising hermes-agent and flash-moe submodules"
    git -C "${MYCONEX_REPO_ROOT:?}" submodule update --init --recursive \
        integrations/hermes-agent integrations/flash-moe
    local py; py=$(command -v python3.11 || command -v python3)
    "$py" -m pip install --quiet -e "${MYCONEX_REPO_ROOT}/integrations/hermes-agent"
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "hermes: done"
}
```

- [ ] **Step 2: Create registration_agent.sh**

Create `install/lib/steps/registration_agent.sh`:
```bash
#!/usr/bin/env bash
# registration_agent.sh — lightweight node: Ollama + mesh announce agent
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_registration_agent"

step_registration_agent() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Installing mesh registration agent"
    # The registration agent is a thin Python script bundled with myconex
    local py; py=$(command -v python3.11 || command -v python3)
    "$py" -m pip install --quiet -r "${MYCONEX_REPO_ROOT}/requirements.txt"
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "registration_agent: done"
}
```

- [ ] **Step 3: Create discord.sh**

Create `install/lib/steps/discord.sh`:
```bash
#!/usr/bin/env bash
# discord.sh — discord.py deps + token collection
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_discord"

step_discord() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Configuring Discord gateway"
    # discord.py is already in requirements.txt — just collect the token
    if [[ -z "${DISCORD_BOT_TOKEN:-}" ]]; then
        if [[ "${UI_MODE:-plain}" == "unattended" ]]; then
            echo "ERROR: DISCORD_BOT_TOKEN env var required for unattended Discord setup" >&2
            exit 1
        fi
        source "$(dirname "${BASH_SOURCE[0]}")/../ui.sh"
        ui_input "Discord bot token:" DISCORD_BOT_TOKEN secret
    fi
    # Token written to .env by config.sh — just export for this session
    export DISCORD_BOT_TOKEN
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "discord: done"
}
```

- [ ] **Step 4: Create integrations.sh**

Create `install/lib/steps/integrations.sh`:
```bash
#!/usr/bin/env bash
# integrations.sh — optional ingester API keys (Gmail, YouTube, RSS, podcast)
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_integrations"

step_integrations() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Configuring integrations"
    # Deps already in requirements.txt; collect optional API keys
    for var in GMAIL_CLIENT_ID GMAIL_CLIENT_SECRET; do
        if [[ -z "${!var:-}" ]]; then
            if [[ "${UI_MODE:-plain}" == "unattended" ]]; then
                echo "ERROR: ${var} env var required for unattended integrations setup" >&2
                exit 1
            fi
            source "$(dirname "${BASH_SOURCE[0]}")/../ui.sh"
            ui_input "${var}:" "$var" secret
        fi
        export "${var?}"
    done
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "integrations: done"
}
```

- [ ] **Step 5: Create dashboard.sh**

Create `install/lib/steps/dashboard.sh`:
```bash
#!/usr/bin/env bash
# dashboard.sh — web dashboard deps
set -euo pipefail
SENTINEL="${HOME}/.myconex/.installed_dashboard"

step_dashboard() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Installing dashboard"
    local py; py=$(command -v python3.11 || command -v python3)
    # Dashboard lives at dashboard/app.py; its deps are in requirements.txt
    "$py" -m pip install --quiet gradio 2>/dev/null || true
    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "dashboard: done"
}
```

- [ ] **Step 6: Verify all syntax**

```bash
for f in install/lib/steps/hermes.sh install/lib/steps/registration_agent.sh \
          install/lib/steps/discord.sh install/lib/steps/integrations.sh \
          install/lib/steps/dashboard.sh; do
    bash -n "$f" && echo "OK: $f"
done
```
Expected: `OK:` for each file

- [ ] **Step 7: Commit**

```bash
git add install/lib/steps/hermes.sh install/lib/steps/registration_agent.sh \
        install/lib/steps/discord.sh install/lib/steps/integrations.sh \
        install/lib/steps/dashboard.sh
git commit -m "feat(installer): steps — hermes, registration_agent, discord, integrations, dashboard"
```

---

## Task 11: steps/config.sh — Write node.yaml and .env

**Files:**
- Create: `install/lib/steps/config.sh`
- Create: `tests/installer/test_config.bats`

- [ ] **Step 1: Write failing tests**

Create `tests/installer/test_config.bats`:
```bash
#!/usr/bin/env bats
load 'test_helper'

setup() {
    source_lib "steps/config.sh"
    MYCONEX_REPO_ROOT="$(mktemp -d)"
    mkdir -p "${MYCONEX_REPO_ROOT}/config"
    MYCONEX_NODE_NAME="test-hub"
    MYCONEX_DETECTED_TIER="T2"
    MYCONEX_ROLE="hub"
    UI_MODE="unattended"
}

teardown() { rm -rf "$MYCONEX_REPO_ROOT"; }

@test "config.sh writes node.yaml with correct role" {
    run step_config
    assert_success
    run grep "name: test-hub" "${MYCONEX_REPO_ROOT}/config/node.yaml"
    assert_success
}

@test "config.sh writes node.yaml with detected tier" {
    run step_config
    assert_success
    run grep 'tier: "T2"' "${MYCONEX_REPO_ROOT}/config/node.yaml"
    assert_success
}

@test "config.sh does not overwrite existing node.yaml on upgrade" {
    echo "existing: true" > "${MYCONEX_REPO_ROOT}/config/node.yaml"
    REINSTALL=""
    touch "${HOME}/.myconex/.installed_config" 2>/dev/null || true
    run step_config
    run grep "existing: true" "${MYCONEX_REPO_ROOT}/config/node.yaml"
    assert_success
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_config.bats
```
Expected: fail

- [ ] **Step 3: Create config.sh**

Create `install/lib/steps/config.sh`:
```bash
#!/usr/bin/env bash
# config.sh — writes config/node.yaml and .env
set -euo pipefail

SENTINEL="${HOME}/.myconex/.installed_config"
NODE_YAML="${MYCONEX_REPO_ROOT:?}/config/node.yaml"
ENV_FILE="${MYCONEX_REPO_ROOT}/.env"

step_config() {
    # node.yaml: never overwrite on upgrade (merge semantics)
    if [[ ! -f "$NODE_YAML" ]] || [[ -n "${REINSTALL:-}" ]]; then
        log_step "Writing config/node.yaml"
        cat > "$NODE_YAML" <<YAML
# MYCONEX Node Config — generated by installer
# Override any setting here; takes priority over mesh_config.yaml defaults.

node:
  name: "${MYCONEX_NODE_NAME:-}"
  tier: "${MYCONEX_DETECTED_TIER:-}"
  roles: []
  description: ""

mesh:
  static_peers: []
YAML
    else
        log_step "config/node.yaml exists — skipping (use --reinstall to overwrite)"
    fi

    # .env: append new keys as comments, never overwrite values
    if [[ ! -f "$ENV_FILE" ]]; then
        log_step "Writing .env"
        cp "${MYCONEX_REPO_ROOT}/.env.example" "$ENV_FILE" 2>/dev/null || touch "$ENV_FILE"
    fi

    # Write collected secrets as real values (idempotent — sed replaces blank lines)
    local -A secrets=(
        [DISCORD_BOT_TOKEN]="${DISCORD_BOT_TOKEN:-}"
        [OPENROUTER_API_KEY]="${OPENROUTER_API_KEY:-}"
        [NOUS_API_KEY]="${NOUS_API_KEY:-}"
        [GMAIL_CLIENT_ID]="${GMAIL_CLIENT_ID:-}"
        [GMAIL_CLIENT_SECRET]="${GMAIL_CLIENT_SECRET:-}"
    )
    for key in "${!secrets[@]}"; do
        local val="${secrets[$key]}"
        [[ -z "$val" ]] && continue
        if grep -q "^${key}=" "$ENV_FILE"; then
            sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
        else
            echo "${key}=${val}" >> "$ENV_FILE"
        fi
    done

    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "config: done"
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_config.bats
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add install/lib/steps/config.sh tests/installer/test_config.bats
git commit -m "feat(installer): steps/config.sh — writes node.yaml and .env"
```

---

## Task 12: steps/systemd.sh — Service units

**Files:**
- Create: `install/lib/steps/systemd.sh`

- [ ] **Step 1: Create systemd.sh**

Create `install/lib/steps/systemd.sh`:
```bash
#!/usr/bin/env bash
# systemd.sh — install and enable systemd units (or start.sh fallback)
set -euo pipefail

SENTINEL="${HOME}/.myconex/.installed_systemd"
UNIT_DIR="/etc/systemd/system"

step_systemd() {
    [[ -f "$SENTINEL" && -z "${REINSTALL:-}" ]] && return 0
    log_step "Installing service units"

    local role="${MYCONEX_ROLE:?}"

    if systemctl --version &>/dev/null && [[ -d "$UNIT_DIR" ]]; then
        _install_systemd_unit "$role"
    else
        log_step "systemd not available — installing start.sh fallback"
        _install_start_sh "$role"
    fi

    mkdir -p "$(dirname "$SENTINEL")"; touch "$SENTINEL"
    log_step "systemd: done"
}

_install_systemd_unit() {
    local role="$1"
    case "$role" in
        hub)
            sudo tee "${UNIT_DIR}/myconex-hub.service" > /dev/null <<UNIT
[Unit]
Description=MYCONEX Hub Services
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=${MYCONEX_REPO_ROOT}/services
ExecStart=/usr/bin/docker compose --profile full up
ExecStop=/usr/bin/docker compose --profile full down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
            sudo systemctl daemon-reload
            sudo systemctl enable --now myconex-hub.service
            ;;
        full-node)
            local py; py=$(command -v python3.11 || command -v python3)
            sudo tee "${UNIT_DIR}/myconex-node.service" > /dev/null <<UNIT
[Unit]
Description=MYCONEX Mesh Node
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${MYCONEX_REPO_ROOT}
ExecStart=${py} -m myconex --mode worker
Restart=on-failure
RestartSec=10
Environment="MYCONEX_CONFIG=/etc/myconex/mesh_config.yaml"

[Install]
WantedBy=multi-user.target
UNIT
            sudo systemctl daemon-reload
            sudo systemctl enable --now myconex-node.service
            ;;
        lightweight)
            local py; py=$(command -v python3.11 || command -v python3)
            sudo tee "${UNIT_DIR}/myconex-registration.service" > /dev/null <<UNIT
[Unit]
Description=MYCONEX Mesh Registration Agent
After=network-online.target ollama.service

[Service]
Type=simple
WorkingDirectory=${MYCONEX_REPO_ROOT}
ExecStart=${py} -m myconex --mode registration
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
            sudo systemctl daemon-reload
            sudo systemctl enable --now myconex-registration.service
            ;;
    esac
}

_install_start_sh() {
    local role="$1"
    local start="${HOME}/.myconex/start.sh"
    mkdir -p "$(dirname "$start")"
    case "$role" in
        hub)
            cat > "$start" <<SH
#!/usr/bin/env bash
cd "${MYCONEX_REPO_ROOT}/services"
exec docker compose --profile full up
SH
            ;;
        full-node)
            local py; py=$(command -v python3.11 || command -v python3)
            cat > "$start" <<SH
#!/usr/bin/env bash
cd "${MYCONEX_REPO_ROOT}"
exec ${py} -m myconex --mode worker
SH
            ;;
        lightweight)
            local py; py=$(command -v python3.11 || command -v python3)
            cat > "$start" <<SH
#!/usr/bin/env bash
cd "${MYCONEX_REPO_ROOT}"
exec ${py} -m myconex --mode registration
SH
            ;;
    esac
    chmod +x "$start"
    log_step "Start with: $start"
}
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n install/lib/steps/systemd.sh
```
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add install/lib/steps/systemd.sh
git commit -m "feat(installer): steps/systemd.sh — service units + start.sh fallback"
```

---

## Task 13: install.sh — Main orchestrator

**Files:**
- Create: `install.sh`
- Create: `tests/installer/test_install_flags.bats`

- [ ] **Step 1: Write failing flag tests**

Create `tests/installer/test_install_flags.bats`:
```bash
#!/usr/bin/env bats
load 'test_helper'

@test "install.sh --help exits 0" {
    run bash "${REPO_ROOT}/install.sh" --help
    assert_success
    assert_output --partial "Usage:"
}

@test "install.sh --role hub sets MYCONEX_ROLE=hub" {
    run bash -c "
        source '${REPO_ROOT}/install.sh' --parse-only --role hub
        echo \"\$MYCONEX_ROLE\"
    "
    assert_output "hub"
}

@test "install.sh --unattended missing file exits 1" {
    run bash "${REPO_ROOT}/install.sh" --unattended /nonexistent.yaml
    assert_failure
    assert_output --partial "not found"
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_install_flags.bats
```
Expected: fail

- [ ] **Step 3: Create install.sh**

Create `install.sh`:
```bash
#!/usr/bin/env bash
# MYCONEX Installer — canonical entry point
# Usage: ./install.sh [--role hub|full-node|lightweight] [--unattended answers.yaml]
#        [--save-answers] [--answers-out path] [--no-tui] [--reinstall]
#        [--skip-verify] [--log path]
set -euo pipefail

MYCONEX_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MYCONEX_REPO_ROOT

MYCONEX_LOG="${HOME}/.myconex/install.log"
MYCONEX_ROLE=""
MYCONEX_ANSWERS_FILE=""
MYCONEX_SAVE_ANSWERS=""
MYCONEX_ANSWERS_OUT="./myconex-answers.yaml"
REINSTALL=""
SKIP_VERIFY=""
PARSE_ONLY=""  # test hook: source + parse flags without executing
export REINSTALL

# ── Parse flags ──────────────────────────────────────────────────────────────
_parse_flags() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --role)         MYCONEX_ROLE="$2";         shift 2 ;;
            --unattended)   MYCONEX_ANSWERS_FILE="$2"; shift 2 ;;
            --save-answers) MYCONEX_SAVE_ANSWERS=1;    shift ;;
            --answers-out)  MYCONEX_ANSWERS_OUT="$2";  shift 2 ;;
            --no-tui)       UI_MODE="plain";            shift ;;
            --reinstall)    REINSTALL=1;                shift ;;
            --skip-verify)  SKIP_VERIFY=1;             shift ;;
            --log)          MYCONEX_LOG="$2";          shift 2 ;;
            --parse-only)   PARSE_ONLY=1;              shift ;;
            --help|-h)
                echo "Usage: install.sh [--role hub|full-node|lightweight]"
                echo "       [--unattended answers.yaml] [--save-answers] [--answers-out path]"
                echo "       [--no-tui] [--reinstall] [--skip-verify] [--log path]"
                exit 0 ;;
            *) echo "Unknown flag: $1" >&2; exit 1 ;;
        esac
    done
    export MYCONEX_ROLE UI_MODE MYCONEX_ANSWERS_FILE MYCONEX_ANSWERS_OUT
}

# ── Logging ───────────────────────────────────────────────────────────────────
log_step() { echo "[install] $*" | tee -a "${MYCONEX_LOG}"; }
export -f log_step

# ── Load libraries ────────────────────────────────────────────────────────────
_load_libs() {
    local lib="${MYCONEX_REPO_ROOT}/install/lib"
    source "${lib}/detect.sh"
    source "${lib}/ui.sh"
    source "${lib}/profiles.sh"
    for step in "${lib}/steps/"*.sh; do source "$step"; done
}

# ── Answer file ───────────────────────────────────────────────────────────────
_load_answers() {
    local file="$1"
    [[ -f "$file" ]] || { echo "ERROR: answer file not found: $file" >&2; exit 1; }
    # Requires yq; parse key→env var
    command -v yq &>/dev/null || { echo "ERROR: yq required for --unattended" >&2; exit 1; }
    MYCONEX_ROLE=$(yq e '.role' "$file")
    MYCONEX_NODE_NAME=$(yq e '.node_name // ""' "$file")
    MYCONEX_HUB_ADDRESS=$(yq e '.hub_address // ""' "$file")
    FEAT_DISCORD=$(yq e '.features.discord // false' "$file")
    FEAT_INTEGRATIONS=$(yq e '.features.integrations // false' "$file")
    FEAT_DASHBOARD=$(yq e '.features.dashboard // false' "$file")
    MYCONEX_OLLAMA_MODEL=$(yq e '.ollama_model // ""' "$file")
    export MYCONEX_ROLE MYCONEX_NODE_NAME MYCONEX_HUB_ADDRESS
    export FEAT_DISCORD FEAT_INTEGRATIONS FEAT_DASHBOARD MYCONEX_OLLAMA_MODEL
}

_save_answers() {
    local out="$1"
    cat > "$out" <<YAML
role: ${MYCONEX_ROLE}
node_name: "${MYCONEX_NODE_NAME:-}"
hub_address: "${MYCONEX_HUB_ADDRESS:-}"

features:
  discord: ${FEAT_DISCORD:-false}
  integrations: ${FEAT_INTEGRATIONS:-false}
  dashboard: ${FEAT_DASHBOARD:-false}

api_keys:
  discord_bot_token: ""
  openrouter_api_key: ""
  nous_api_key: ""

ollama_model: "${MYCONEX_OLLAMA_MODEL:-}"
YAML
    echo "Answers saved to: $out"
}

# ── Verification ──────────────────────────────────────────────────────────────
_verify() {
    local role="$1"
    log_step "Running verification checks"
    local ok=1
    _check_http() {
        local name="$1" url="$2"
        if curl -sf "$url" &>/dev/null; then
            printf "  \033[32m✓\033[0m  %-12s %s\n" "$name" "$url"
        else
            printf "  \033[31m✗\033[0m  %-12s %s  (FAILED)\n" "$name" "$url"
            ok=0
        fi
    }
    _check_redis() {
        if redis-cli -h localhost ping 2>/dev/null | grep -q PONG; then
            printf "  \033[32m✓\033[0m  %-12s redis://localhost:6379\n" "Redis"
        else
            printf "  \033[31m✗\033[0m  %-12s redis://localhost:6379  (FAILED)\n" "Redis"
            ok=0
        fi
    }
    if [[ "$role" == "hub" ]]; then
        _check_http "NATS"    "http://localhost:8222/healthz"
        _check_redis
        _check_http "Qdrant"  "http://localhost:6333/healthz"
        _check_http "Ollama"  "http://localhost:11434/api/tags"
        _check_http "LiteLLM" "http://localhost:4000/health/liveliness"
        _check_http "API"     "http://localhost:8765/health"
    fi
    [[ "$ok" == 1 ]] && log_step "All checks passed" || log_step "Some checks failed — see log"
}

# ── Main flow ─────────────────────────────────────────────────────────────────
main() {
    _parse_flags "$@"
    [[ -n "$PARSE_ONLY" ]] && return 0

    mkdir -p "$(dirname "$MYCONEX_LOG")"
    _load_libs

    # Detect environment
    MYCONEX_DETECTED_TIER=$(detect_tier)
    local display_mode; display_mode=$(detect_display_mode)
    UI_MODE="${UI_MODE:-$display_mode}"
    export MYCONEX_DETECTED_TIER UI_MODE

    # Load answers or run interactive flow
    if [[ -n "$MYCONEX_ANSWERS_FILE" ]]; then
        UI_MODE="unattended"
        _load_answers "$MYCONEX_ANSWERS_FILE"
    else
        # Role selection
        if [[ -z "$MYCONEX_ROLE" ]]; then
            ui_menu "What is this machine?" "hub" "full-node" "lightweight"
            MYCONEX_ROLE="$REPLY"
            export MYCONEX_ROLE
        fi

        # Feature selection
        FEAT_DISCORD=false; FEAT_INTEGRATIONS=false; FEAT_DASHBOARD=false
        if [[ "$MYCONEX_ROLE" != "lightweight" ]]; then
            ui_checklist "Optional features:" \
                "discord:off" "integrations:off" "dashboard:off"
            for sel in "${SELECTED[@]}"; do
                case "$sel" in
                    discord)      FEAT_DISCORD=true ;;
                    integrations) FEAT_INTEGRATIONS=true ;;
                    dashboard)    FEAT_DASHBOARD=true ;;
                esac
            done
        fi
        export FEAT_DISCORD FEAT_INTEGRATIONS FEAT_DASHBOARD

        # Configure
        ui_input "Node name [$(hostname)]:" MYCONEX_NODE_NAME
        MYCONEX_NODE_NAME="${MYCONEX_NODE_NAME:-$(hostname)}"
        export MYCONEX_NODE_NAME
        if [[ "$MYCONEX_ROLE" != "hub" ]]; then
            ui_input "Hub address (blank = mDNS auto-discover):" MYCONEX_HUB_ADDRESS
            export MYCONEX_HUB_ADDRESS
        fi
    fi

    # Save answers if requested (no install)
    if [[ -n "$MYCONEX_SAVE_ANSWERS" ]]; then
        _save_answers "$MYCONEX_ANSWERS_OUT"
        return 0
    fi

    # Plan preview
    log_step "Role: $MYCONEX_ROLE | Tier: $MYCONEX_DETECTED_TIER | UI: $UI_MODE"
    ui_confirm "Proceed with installation?" || { echo "Aborted."; exit 0; }

    # Execute steps
    step_core
    if [[ "$MYCONEX_ROLE" == "hub" ]]; then
        step_hub_services
        step_gpu
        step_llm
        step_hermes
    elif [[ "$MYCONEX_ROLE" == "full-node" ]]; then
        step_llm
        profile_requires_for_tier "full-node" "$MYCONEX_DETECTED_TIER" "gpu"    && step_gpu
        profile_requires_for_tier "full-node" "$MYCONEX_DETECTED_TIER" "hermes_moe" && step_hermes
    else  # lightweight
        step_registration_agent
        step_llm
    fi
    [[ "$FEAT_DISCORD"      == "true" ]] && step_discord
    [[ "$FEAT_INTEGRATIONS" == "true" ]] && step_integrations
    [[ "$FEAT_DASHBOARD"    == "true" ]] && step_dashboard
    step_config
    step_systemd

    # Verify
    [[ -z "$SKIP_VERIFY" ]] && _verify "$MYCONEX_ROLE"

    log_step "Installation complete. Node: ${MYCONEX_NODE_NAME} | Tier: ${MYCONEX_DETECTED_TIER}"
}

main "$@"
```

- [ ] **Step 4: Make executable and verify syntax**

```bash
chmod +x install.sh
bash -n install.sh
```
Expected: no output

- [ ] **Step 5: Run flag tests**

```bash
tests/lib/bats-core/bin/bats tests/installer/test_install_flags.bats
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add install.sh tests/installer/test_install_flags.bats
git commit -m "feat(installer): install.sh — main orchestrator, full install flow"
```

---

## Task 14: install/answers.yaml.example

**Files:**
- Create: `install/answers.yaml.example`

- [ ] **Step 1: Create answers.yaml.example**

Create `install/answers.yaml.example`:
```yaml
# MYCONEX Answer File — fleet/unattended install config
# Generate with: ./install.sh --save-answers --answers-out myconex-answers.yaml
# Use with:      ./install.sh --unattended myconex-answers.yaml
#
# Secrets (bot tokens, API keys) must be passed as environment variables —
# they are never written here. See README for the full env var list.

# Machine role in the mesh
role: hub                        # hub | full-node | lightweight-node

# Node identity
node_name: ""                    # default: system hostname
hub_address: ""                  # nodes only; blank = mDNS auto-discovery

# Optional feature flags
features:
  discord: false                 # install Discord gateway
  integrations: false            # install Gmail/YouTube/RSS/podcast ingesters
  dashboard: false               # install web dashboard

# API keys — LEAVE BLANK HERE, inject via env vars at install time:
#   DISCORD_BOT_TOKEN, OPENROUTER_API_KEY, NOUS_API_KEY,
#   GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET
api_keys:
  discord_bot_token: ""
  openrouter_api_key: ""
  nous_api_key: ""

# Ollama model override — blank = auto-select by detected hardware tier
# Tier defaults: T1=llama3.1:70b  T2=qwen3:8b  T3=qwen3:4b  T4=qwen3:0.6b
ollama_model: ""
```

- [ ] **Step 2: Commit**

```bash
git add install/answers.yaml.example
git commit -m "docs(installer): add answers.yaml.example with full documentation"
```

---

## Task 15: install.ps1 — Windows wrapper

**Files:**
- Create: `install.ps1`

- [ ] **Step 1: Create install.ps1**

Create `install.ps1`:
```powershell
#Requires -Version 5
<#
.SYNOPSIS
  MYCONEX Windows installer — delegates to install.sh via WSL, or runs natively.
.EXAMPLE
  .\install.ps1 --role hub
  .\install.ps1 --unattended answers.yaml
#>
param(
    [string]$Role         = "",
    [string]$Unattended   = "",
    [switch]$SaveAnswers,
    [string]$AnswersOut   = ".\myconex-answers.yaml",
    [switch]$NoTui,
    [switch]$Reinstall,
    [switch]$SkipVerify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step { param([string]$Msg); Write-Host "[install] $Msg" -ForegroundColor Cyan }

# ── Build arg list to pass through ────────────────────────────────────────────
$passArgs = @()
if ($Role)        { $passArgs += "--role", $Role }
if ($Unattended)  { $passArgs += "--unattended", $Unattended }
if ($SaveAnswers) { $passArgs += "--save-answers" }
if ($AnswersOut)  { $passArgs += "--answers-out", $AnswersOut }
if ($NoTui)       { $passArgs += "--no-tui" }
if ($Reinstall)   { $passArgs += "--reinstall" }
if ($SkipVerify)  { $passArgs += "--skip-verify" }

# ── WSL path ──────────────────────────────────────────────────────────────────
function Get-WslStatus {
    try {
        $out = wsl --status 2>&1 | Out-String
        if ($out -match "Default Version: 1") { return "wsl1" }
        if ($out -match "WSL" -or (Get-Command wsl -ErrorAction SilentlyContinue)) { return "wsl2" }
    } catch {}
    return "none"
}

$wslStatus = Get-WslStatus

if ($wslStatus -eq "wsl1") {
    Write-Step "WSL1 detected. WSL2 is required. Upgrading..."
    wsl --set-default-version 2
    Write-Step "Please restart your terminal and re-run this installer."
    exit 0
}

if ($wslStatus -eq "wsl2") {
    Write-Step "WSL2 detected — running Linux installer inside WSL"
    # Convert Windows path to WSL path
    $repoRoot = (wsl wslpath "'$PSScriptRoot'").Trim()
    $installSh = "${repoRoot}/install.sh"
    $wslArgs = @("bash", $installSh) + $passArgs
    wsl @wslArgs
    exit $LASTEXITCODE
}

# ── Native Windows path (no WSL) ──────────────────────────────────────────────
Write-Step "No WSL detected — running native Windows install"

# 1. Chocolatey
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Step "Installing Chocolatey"
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

# 2. Core deps
foreach ($pkg in @("python", "git", "docker-desktop")) {
    if (-not (Get-Command $pkg.Split('-')[0] -ErrorAction SilentlyContinue)) {
        Write-Step "Installing $pkg"
        choco install $pkg -y --no-progress
    }
}

# 3. Wait for Docker Desktop engine
Write-Step "Waiting for Docker Desktop engine..."
$retries = 30
while ($retries -gt 0) {
    if (docker info 2>$null) { break }
    Start-Sleep 5; $retries--
}
if ($retries -eq 0) { Write-Error "Docker Desktop did not start in time."; exit 1 }

# 4. Pip deps
Write-Step "Installing Python dependencies"
python -m pip install --upgrade -r "$PSScriptRoot\requirements.txt"

# 5. Task Scheduler entry for auto-start
$taskName = "MYCONEX-$($Role.ToUpper() -replace '-','')"
Write-Step "Registering Task Scheduler entry: $taskName"
$action  = New-ScheduledTaskAction -Execute "python" `
           -Argument "-m myconex --mode $(if ($Role -eq 'hub') {'api'} else {'worker'})" `
           -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -RunLevel Highest -Force | Out-Null

Write-Step "Windows install complete. Start manually: python -m myconex --mode $(if ($Role -eq 'hub') {'api'} else {'worker'})"
```

- [ ] **Step 2: Verify PowerShell syntax**

```powershell
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    "install.ps1", [ref]$null, [ref]$null)
Write-Host "Syntax OK"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add install.ps1
git commit -m "feat(installer): install.ps1 — Windows wrapper (WSL + native PS fallback)"
```

---

## Task 16: Run all tests and final check

- [ ] **Step 1: Run full test suite**

```bash
tests/lib/bats-core/bin/bats tests/installer/
```
Expected: all tests pass, 0 failures

- [ ] **Step 2: Verify all scripts have no syntax errors**

```bash
for f in install.sh install/lib/detect.sh install/lib/ui.sh install/lib/profiles.sh \
          install/lib/steps/*.sh; do
    bash -n "$f" && echo "OK: $f"
done
```
Expected: `OK:` for every file

- [ ] **Step 3: Verify hardware.py test still passes**

```bash
pytest tests/test_hardware_tiers.py -v
```
Expected: PASS

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(installer): complete cross-platform installer — hub/node/lightweight, TUI/SSH/unattended"
```
