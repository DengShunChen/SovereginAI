#!/usr/bin/env bash
# 載入叢集下載代理（~/.proxy：ssh tunnel localhost:8888 → proxy.cwa.gov.tw）
# 給 train_lora_a100.sh / sbatch 用。8888 已在聽就只 export，避免 ssh -Nf 撞 port。
# shellcheck disable=SC1090

_proxy_port_open() {
    (echo >/dev/tcp/127.0.0.1/8888) >/dev/null 2>&1
}

load_cluster_proxy() {
    export http_proxy="${http_proxy:-http://localhost:8888}"
    export https_proxy="${https_proxy:-http://localhost:8888}"
    export HTTP_PROXY="${HTTP_PROXY:-http://localhost:8888}"
    export HTTPS_PROXY="${HTTPS_PROXY:-http://localhost:8888}"
    export no_proxy="${no_proxy:-localhost,127.0.0.1}"
    export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

    if _proxy_port_open; then
        echo "proxy: reuse localhost:8888" >&2
        return 0
    fi
    if [[ ! -f "${HOME}/.proxy" ]]; then
        echo "警告：找不到 ~/.proxy 且 :8888 未開，HF 下載可能失敗" >&2
        return 0
    fi
    _had_e=0
    [[ $- == *e* ]] && _had_e=1
    set +e
    # shellcheck source=/dev/null
    source "${HOME}/.proxy"
    _rc=$?
    [[ $_had_e -eq 1 ]] && set -e
    if [[ $_rc -ne 0 ]]; then
        echo "警告：source ~/.proxy 結束碼 $_rc（若模型已在 HF cache 可忽略）" >&2
    fi
    export http_proxy="${http_proxy:-http://localhost:8888}"
    export https_proxy="${https_proxy:-http://localhost:8888}"
    export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
    export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
}
