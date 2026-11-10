# Sourced by run_train.sh / run_sample.sh: model, diffusion and training
# parameters come from a config file in cfgs/ instead of docker-compose.yml.
#
# Format: one KEY=VALUE per line, "#" starts a comment, the value may be quoted.
# A variable already set in the environment (e.g. `docker compose run -e
# LR_ANNEAL_STEPS=200 ...`) takes precedence over the file.

# Keys that must not change when resuming from the same log directory: they
# fix the network shape or the noise schedule (the schedule is not stored in
# the checkpoint, so a mismatch would load silently and sample wrongly).
CFG_MODEL_KEYS="IMAGE_SIZE NUM_CHANNELS NUM_RES_BLOCKS NUM_HEADS NUM_HEAD_CHANNELS \
ATTENTION_RESOLUTIONS CHANNEL_MULT LEARN_SIGMA CLASS_COND USE_SCALE_SHIFT_NORM \
RESBLOCK_UPDOWN USE_NEW_ATTENTION_ORDER ARCH MP_CLIP_ACT ATTR_COND \
DIFFUSION_STEPS NOISE_SCHEDULE"
CFG_KEYS=""

load_cfg() {
  local file="$1" line key value n=0
  [ -f "$file" ] || { echo "config not found: $file" >&2; return 1; }
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    [[ "$line" =~ ^[[:space:]]*(#.*)?$ ]] && continue
    if [[ ! "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
      echo "$file:$n: expected KEY=VALUE: $line" >&2
      return 1
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]%%[[:space:]]#*}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ "$value" =~ ^\"(.*)\"$ ]] && value="${BASH_REMATCH[1]}"
    CFG_KEYS="$CFG_KEYS $key"
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    elif [ "${!key}" != "$value" ]; then
      echo "config: $key=${!key} (environment; $file has $value)"
    fi
  done < "$file"
  echo "config: $file"
}

# Writes the effective values of the loaded keys (after environment overrides).
save_cfg() {
  local key
  for key in $CFG_KEYS; do
    printf '%s="%s"\n' "$key" "${!key}"
  done > "$1"
}

# Aborts if a config saved by an earlier run ($1) differs in CFG_MODEL_KEYS.
check_cfg_compatible() {
  local saved="$1" key diff=""
  [ -f "$saved" ] || return 0
  for key in $CFG_MODEL_KEYS; do
    local old
    old=$(grep -E "^$key=" "$saved" | tail -n 1 | sed -E 's/^[^=]*="?//; s/"$//')
    [ "$old" != "${!key}" ] && diff="$diff\n  $key: $old -> ${!key}"
  done
  if [ -n "$diff" ]; then
    echo -e "config differs from the run in $(dirname "$saved") ($saved):$diff" >&2
    echo "use another log directory for a different model/schedule" >&2
    return 1
  fi
}
