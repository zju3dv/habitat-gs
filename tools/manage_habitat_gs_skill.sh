#!/usr/bin/env bash
set -euo pipefail

SKILL_NAME="habitat-gs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_SKILL_DIR="${REPO_ROOT}/skills/${SKILL_NAME}"

usage() {
    cat <<'EOF'
Usage:
  tools/manage_habitat_gs_skill.sh install   [--workspace PATH] [--mode copy|symlink] [--force]
  tools/manage_habitat_gs_skill.sh uninstall [--workspace PATH] [--force]
  tools/manage_habitat_gs_skill.sh status    [--workspace PATH]

Notes:
  - Only operates on skill path: <workspace>/skills/habitat-gs
  - Default install mode is "copy"
  - "symlink" mode requires target path to be accessible
EOF
}

log() {
    printf '[skill-manager] %s\n' "$*"
}

die() {
    printf '[skill-manager][error] %s\n' "$*" >&2
    exit 1
}

is_managed_target() {
    local target_dir="$1"

    if [[ -L "${target_dir}" ]]; then
        local resolved
        resolved="$(readlink -f "${target_dir}" || true)"
        [[ "${resolved}" == "${SOURCE_SKILL_DIR}" ]] && return 0
    fi

    if [[ -f "${target_dir}/SKILL.md" ]] && rg -q '^name:\s*habitat-gs' "${target_dir}/SKILL.md"; then
        return 0
    fi

    return 1
}

install_skill() {
    local workspace="$1"
    local mode="$2"
    local force="$3"
    local target_dir="${workspace}/skills/${SKILL_NAME}"

    mkdir -p "${workspace}/skills"

    if [[ -e "${target_dir}" || -L "${target_dir}" ]]; then
        if [[ "${force}" != "1" ]] && ! is_managed_target "${target_dir}"; then
            die "Refusing to overwrite unmanaged target: ${target_dir} (use --force to override)"
        fi
        rm -rf "${target_dir}"
    fi

    if [[ "${mode}" == "symlink" ]]; then
        ln -s "${SOURCE_SKILL_DIR}" "${target_dir}"
        log "Installed ${SKILL_NAME} via symlink: ${target_dir} -> ${SOURCE_SKILL_DIR}"
    else
        mkdir -p "${target_dir}"
        cp -a "${SOURCE_SKILL_DIR}/." "${target_dir}/"
        chmod +x "${target_dir}/scripts/hab" 2>/dev/null || true
        log "Installed ${SKILL_NAME} via copy: ${target_dir}"
    fi

    log "Skill env template: ${target_dir}/.env.example"
}

uninstall_skill() {
    local workspace="$1"
    local force="$2"
    local target_dir="${workspace}/skills/${SKILL_NAME}"

    if [[ ! -e "${target_dir}" && ! -L "${target_dir}" ]]; then
        log "Nothing to remove: ${target_dir}"
        return
    fi

    if [[ "${force}" != "1" ]] && ! is_managed_target "${target_dir}"; then
        die "Refusing to remove unmanaged target: ${target_dir} (use --force to override)"
    fi

    rm -rf "${target_dir}"
    log "Removed ${target_dir}"
}

status_skill() {
    local workspace="$1"
    local target_dir="${workspace}/skills/${SKILL_NAME}"
    local skill_env="${target_dir}/.env"
    local workspace_env="${workspace}/.env"

    echo "source_skill=${SOURCE_SKILL_DIR}"
    echo "workspace=${workspace}"
    echo "target=${target_dir}"

    if [[ -L "${target_dir}" ]]; then
        echo "target_type=symlink"
        echo "target_resolved=$(readlink -f "${target_dir}" || true)"
    elif [[ -d "${target_dir}" ]]; then
        echo "target_type=directory"
        echo "skill_installed=true"
    else
        echo "target_type=missing"
    fi

    echo "skill_env_exists=$([[ -f "${skill_env}" ]] && echo yes || echo no)"
    echo "workspace_env_exists=$([[ -f "${workspace_env}" ]] && echo yes || echo no)"
}

main() {
    [[ $# -ge 1 ]] || { usage; exit 1; }

    local action="$1"
    shift

    local workspace=""
    local mode="copy"
    local force="0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --workspace)
                [[ $# -ge 2 ]] || die "--workspace requires a value"
                workspace="$2"
                shift 2
                ;;
            --mode)
                [[ $# -ge 2 ]] || die "--mode requires a value"
                mode="$2"
                shift 2
                ;;
            --force)
                force="1"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done

    [[ -d "${SOURCE_SKILL_DIR}" ]] || die "Source skill not found: ${SOURCE_SKILL_DIR}"
    [[ "${mode}" == "copy" || "${mode}" == "symlink" ]] || die "--mode must be copy or symlink"

    [[ -n "${workspace}" ]] || die "Provide --workspace PATH"

    workspace="$(readlink -f "${workspace}")"
    [[ -d "${workspace}" ]] || die "Workspace path does not exist: ${workspace}"

    case "${action}" in
        install)
            install_skill "${workspace}" "${mode}" "${force}"
            ;;
        uninstall)
            uninstall_skill "${workspace}" "${force}"
            ;;
        status)
            status_skill "${workspace}"
            ;;
        *)
            die "Unknown action: ${action}"
            ;;
    esac
}

main "$@"
