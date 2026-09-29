#!/usr/bin/env bash
# pull_image.sh — pull the LCB evaluator image from a registry, ready for evals.
#
# The pulled image is self-contained (dataset baked in), so no HF_TOKEN and no
# rebuild are needed here. By default the image is re-tagged locally as
# lcb-service:latest so the hardened run command in README.md and the scorer's
# ws://localhost:13835/evaluate expectation work unchanged.
#
# Usage (images are tagged by short SHA, so LCB_IMAGE_TAG selects the build):
#   LCB_IMAGE_REGISTRY=myregistry.com/team LCB_IMAGE_TAG=<sha> ./pull_image.sh
#   LCB_IMAGE_REGISTRY=myregistry.com/team LCB_IMAGE_TAG=<sha> ./pull_image.sh --sqsh            # also create lcb_service.sqsh
#   LCB_IMAGE_REGISTRY=myregistry.com/team LCB_IMAGE_TAG=<sha> ./pull_image.sh --sqsh out.sqsh   # enroot import to out.sqsh
#   LCB_IMAGE_REGISTRY=myregistry.com/team LCB_IMAGE_TAG=<sha> ./pull_image.sh --no-local-tag    # skip the lcb-service:latest tag
#
# Environment variables: see _image_env.sh (LCB_IMAGE_REGISTRY + LCB_IMAGE_TAG required).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -euo/d'
}

MAKE_SQSH=0
SQSH_OUT="lcb_service.sqsh"
LOCAL_TAG=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sqsh)
            MAKE_SQSH=1
            # Optional positional output file (anything that isn't another flag).
            if [[ -n "${2:-}" && "$2" != --* ]]; then
                SQSH_OUT="$2"
                shift
            fi
            ;;
        --no-local-tag) LOCAL_TAG=0 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

# shellcheck source=_image_env.sh
source "${SCRIPT_DIR}/_image_env.sh"

echo ">> Pulling ${LCB_IMAGE_REF}"
docker pull "$LCB_IMAGE_REF"

if [[ "$LOCAL_TAG" -eq 1 ]]; then
    echo ">> Tagging ${LCB_IMAGE_REF} -> ${LCB_LOCAL_TAG}"
    docker tag "$LCB_IMAGE_REF" "$LCB_LOCAL_TAG"
fi

if [[ "$MAKE_SQSH" -eq 1 ]]; then
    if ! command -v enroot >/dev/null 2>&1; then
        echo "error: --sqsh requested but 'enroot' is not installed." >&2
        exit 1
    fi
    echo ">> Importing ${LCB_IMAGE_REF} -> ${SQSH_OUT} (enroot)"
    enroot import --output "$SQSH_OUT" "dockerd://${LCB_IMAGE_REF}"
fi

echo
echo "Done. Run the hardened service with:"
echo "  docker run \\"
echo "    --name lcb-service --rm --read-only \\"
echo "    --tmpfs /tmp:rw,noexec,nosuid,size=1g \\"
echo "    -p 127.0.0.1:13835:13835 \\"
echo "    --security-opt=no-new-privileges:true \\"
echo "    --security-opt apparmor=docker-default \\"
echo "    --memory=32g --memory-swap=32g --cpus=24 \\"
echo "    -e LCB_N_WORKERS=16 --pids-limit=4096 --cap-drop=ALL \\"
echo "    ${LCB_LOCAL_TAG}"
