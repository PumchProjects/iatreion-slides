#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/extract.sh [options]

Download selected log files for every configured download.

Options:
  -n, --dry-run  Show transfer statistics without writing files.
  -h, --help     Show this help.
EOF
}

dry_run=0

while (($#)); do
  case "$1" in
    -n|--dry-run)
      dry_run=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON:-python}"

config_rows="$($python_bin "$project_root/config.py" extract-roots)"
if [[ -z "$config_rows" ]]; then
  echo "config.toml does not contain any units" >&2
  exit 1
fi

rsync_help="$(rsync --help 2>&1)"
if grep -q -- '--info=' <<< "$rsync_help"; then
  progress_options=(--info=progress2 --no-inc-recursive)
else
  progress_options=(--progress --out-format=)
fi

sync_filters=(
  --filter='+ /processed/'
  --filter='+ /processed/process_info.toml'
  --filter='- /processed/***'
  --filter='- /*/training/**/events/'
  --filter='- /*/training/**/fold_artifacts/'
  --filter='- /*/training/**/roc_*_[0-9].png'
  --filter='+ /*/training/**/roc_*.png'
  --filter='+ /*/training/**/available_fusion.toml'
  --filter='+ /*/training/**/score_*.json'
  --filter='- /*/training/**/train_ci_*_[0-9].log'
  --filter='+ /*/training/**/train_ci_*.log'
  --filter='- /*/final/**/events/'
  --filter='- /*/final/**/manifest.toml'
  --filter='- /*/final/**/train.log'
  --filter='+ */'
  --filter='- /*/training/***'
  --filter='+ /*/final/***'
  --filter='- *'
)

remote_quote() {
  local value="$1"
  value="${value//\'/\'\\\'\'}"
  printf "'%s'" "$value"
}

remote_rsync_source() {
  local path="$1"
  printf '%s:%s' "$remote_host" "$path"
}

validate_remote_unit() {
  if ! ssh -n "$remote_host" \
    "test -d $(remote_quote "$remote_root") && test -f $(remote_quote "$remote_root/processed/process_info.toml")"; then
    echo "Missing remote root or processed/process_info.toml: $remote_root" >&2
    return 1
  fi
}

sync_unit() {
  local rsync_cmd

  validate_remote_unit
  if (( ! dry_run )); then
    mkdir -p "$local_root"
  fi

  rsync_cmd=(
    rsync
    -a
    --prune-empty-dirs
  )
  if (( dry_run )); then
    rsync_cmd+=(--dry-run --stats)
  else
    rsync_cmd+=("${progress_options[@]}")
  fi
  rsync_cmd+=(
    "${sync_filters[@]}"
    "$(remote_rsync_source "$remote_root/")"
    "$local_root/"
  )
  "${rsync_cmd[@]}" </dev/null
}

download_index=0
while IFS=$'\t' read -r remote_host remote_root local_root <&3; do
  ((download_index += 1))
  echo "===== Download $download_index: $remote_root -> $local_root ====="
  sync_unit
  echo "===== Download $download_index complete ====="
done 3<<< "$config_rows"
