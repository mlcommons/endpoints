#!/usr/bin/env bash
# push_docker_image.sh — build and push the endpoints client image to a registry.
#
# Builds scripts/Dockerfile.dev at a given git ref and pushes it to GitHub
# Container Registry (default: ghcr.io/mlcommons/endpoints), tagged with both the
# short commit SHA and the ref name so consumers can pin either.
#
# Prerequisites:
#   - docker with buildx  (the script provisions a docker-container builder; the
#     default 'docker' driver can neither --push nor build multi-platform).
#   - Authenticated to the target registry. Either run `docker login ghcr.io`
#     yourself, or export GHCR_USER + GHCR_TOKEN (a GitHub PAT / GITHUB_TOKEN with
#     write:packages) and this script logs in for you.
#
# Usage:
#   ./scripts/push_docker_image.sh                        # build+push current HEAD (linux/amd64)
#   ENDPOINTS_REF=v1.2.0 ./scripts/push_docker_image.sh   # build+push a specific tag/branch/sha
#   ./scripts/push_docker_image.sh --ref main             # same, via flag
#   ./scripts/push_docker_image.sh --multi-arch           # linux/amd64,linux/arm64 manifest list
#   ./scripts/push_docker_image.sh --platform linux/arm64 # single non-native arch
#   ./scripts/push_docker_image.sh --cache                # allow layer cache (default: --no-cache)
#   ./scripts/push_docker_image.sh --multi-arch --no-push # dry build (validate both arches, publish nothing)
#   ./scripts/push_docker_image.sh --allow-dirty          # build HEAD despite uncommitted tracked changes
#
# Multi-arch note: --multi-arch (or --platform with a comma) pushes a manifest
# list usable on BOTH x86 and arm hosts; Docker pulls the matching arch. The
# non-native leg needs QEMU registered on the host once:
#     docker run --privileged --rm tonistiigi/binfmt --install all
# Dockerfile.dev's DSR1 provisioning runs under emulation and is MUCH slower.
#
# Environment variables (flags take precedence):
#   IMAGE           full image ref w/o tag   (default: ghcr.io/mlcommons/endpoints)
#   ENDPOINTS_REF   git ref to build         (default: current HEAD; checked out if it differs)
#   PLATFORM        target platform(s)       (default: linux/amd64)
#   DOCKERFILE      Dockerfile path          (default: scripts/Dockerfile.dev)
#   NO_CACHE        1=--no-cache, 0=cache    (default: 1)
#   ALLOW_DIRTY     1=build HEAD with uncommitted tracked changes (default: 0)
#   PROVISION_DSR1  passthrough build-arg    (default: Dockerfile default = 1)
#   IMAGE_DESCRIPTION  GHCR package-page description (default: "endpoints client image (ref <ref>, commit <sha>)")
#   GHCR_USER/GHCR_TOKEN  optional registry login (fallback: GITHUB_ACTOR/GITHUB_TOKEN)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILDX_BUILDER="endpoints-buildx"

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -euo/d'
}

IMAGE="${IMAGE:-ghcr.io/mlcommons/endpoints}"
ENDPOINTS_REF="${ENDPOINTS_REF:-}"
PLATFORM="${PLATFORM:-linux/amd64}"
DOCKERFILE="${DOCKERFILE:-scripts/Dockerfile.dev}"
NO_CACHE="${NO_CACHE:-1}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
PUSH=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)
            [[ -n "${2:-}" && "$2" != --* ]] || { echo "error: --ref requires a value" >&2; exit 1; }
            ENDPOINTS_REF="$2"; shift ;;
        --ref=*) ENDPOINTS_REF="${1#*=}" ;;
        --image)
            [[ -n "${2:-}" && "$2" != --* ]] || { echo "error: --image requires a value" >&2; exit 1; }
            IMAGE="$2"; shift ;;
        --image=*) IMAGE="${1#*=}" ;;
        --platform)
            [[ -n "${2:-}" && "$2" != --* ]] || { echo "error: --platform requires a value, e.g. linux/arm64" >&2; exit 1; }
            PLATFORM="$2"; shift ;;
        --platform=*) PLATFORM="${1#*=}" ;;
        --multi-arch) PLATFORM="linux/amd64,linux/arm64" ;;
        --no-cache) NO_CACHE=1 ;;
        --cache) NO_CACHE=0 ;;
        --no-push) PUSH=0 ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        --dockerfile)
            [[ -n "${2:-}" && "$2" != --* ]] || { echo "error: --dockerfile requires a value" >&2; exit 1; }
            DOCKERFILE="$2"; shift ;;
        --dockerfile=*) DOCKERFILE="${1#*=}" ;;
        -h | --help) usage; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
    esac
    shift
done

cd "$REPO_ROOT"

if ! docker buildx version >/dev/null 2>&1; then
    echo "error: 'docker buildx' is required but not available." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Optional registry login. Skipped silently if no token is provided (assumes the
# caller already ran `docker login`). Registry host is the first path segment of
# IMAGE (e.g. ghcr.io from ghcr.io/mlcommons/endpoints).
# ---------------------------------------------------------------------------
REGISTRY_HOST="${IMAGE%%/*}"
LOGIN_USER="${GHCR_USER:-${GITHUB_ACTOR:-}}"
LOGIN_TOKEN="${GHCR_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ "$PUSH" == "1" && -n "$LOGIN_TOKEN" ]]; then
    echo ">> Logging in to ${REGISTRY_HOST} as ${LOGIN_USER:-<token>}"
    echo "$LOGIN_TOKEN" | docker login "$REGISTRY_HOST" -u "${LOGIN_USER:-x-access-token}" --password-stdin
fi

# ---------------------------------------------------------------------------
# Resolve the ref to build. If ENDPOINTS_REF is set and points at a different
# commit than the current HEAD, check it out (requiring a clean tree) and restore
# the original ref on exit so the working copy is left as we found it.
# ---------------------------------------------------------------------------
ORIGINAL_REF="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
[[ "$ORIGINAL_REF" == "HEAD" ]] && ORIGINAL_REF="$(git rev-parse HEAD)"

restore_ref() {
    if [[ -n "${DID_CHECKOUT:-}" && -n "$ORIGINAL_REF" ]]; then
        echo ">> Restoring original ref ${ORIGINAL_REF}"
        git checkout --quiet "$ORIGINAL_REF"
    fi
}
trap restore_ref EXIT

# Uncommitted changes to TRACKED files (staged or unstaged) make the working tree
# differ from HEAD. Untracked files are ignored here: they are not part of any commit
# and .dockerignore governs the build context.
tree_has_tracked_changes() { ! git diff --quiet HEAD 2>/dev/null; }

if [[ -n "$ENDPOINTS_REF" ]]; then
    # Fall back to the remote-tracking ref: after actions/checkout lands a detached
    # SHA, a bare branch name like 'main' exists only as origin/main, so the plain
    # resolve would fail and abort the CI default-dispatch path.
    target_sha="$(git rev-parse --verify "${ENDPOINTS_REF}^{commit}" 2>/dev/null \
        || git rev-parse --verify "origin/${ENDPOINTS_REF}^{commit}" 2>/dev/null || true)"
    [[ -n "$target_sha" ]] || { echo "error: '$ENDPOINTS_REF' is not a valid git ref." >&2; exit 1; }
    if [[ "$target_sha" != "$(git rev-parse HEAD)" ]]; then
        if tree_has_tracked_changes; then
            echo "error: working tree has uncommitted changes to tracked files; commit/stash before building a different ref ($ENDPOINTS_REF)." >&2
            exit 1
        fi
        echo ">> Checking out ${ENDPOINTS_REF}"
        git checkout --quiet "$ENDPOINTS_REF"
        DID_CHECKOUT=1
    fi
else
    # Tag the built image with the current branch/ref name for readability.
    ENDPOINTS_REF="$ORIGINAL_REF"
fi

SHORT_SHA="$(git rev-parse --short HEAD)"
# Docker tags allow only [A-Za-z0-9_.-], must not start with '.' or '-', and cap at
# 128 chars. Map every other char (e.g. '/', '+', '~', '#') to '-' so a legitimate
# git ref name can't yield an invalid tag that fails `-t` only after the whole build.
REF_TAG="$(printf '%s' "$ENDPOINTS_REF" | tr -c 'A-Za-z0-9_.-' '-')"
[[ "$REF_TAG" == [A-Za-z0-9_]* ]] || REF_TAG="_${REF_TAG}"
REF_TAG="${REF_TAG:0:128}"

# Building the working tree as-is (no distinct ref checked out): don't publish a
# :$SHORT_SHA tag whose '.' build context doesn't match commit $SHORT_SHA.
if [[ -z "${DID_CHECKOUT:-}" && "$ALLOW_DIRTY" != "1" ]] && tree_has_tracked_changes; then
    dirty_msg="working tree has uncommitted changes to tracked files; :${SHORT_SHA} would not match commit ${SHORT_SHA}"
    if [[ "$PUSH" == "1" ]]; then
        echo "error: ${dirty_msg}." >&2
        echo "       Commit/stash the changes, or pass --allow-dirty to publish anyway." >&2
        exit 1
    fi
    echo ">> warning: ${dirty_msg} (dry run; nothing pushed)." >&2
fi

# ---------------------------------------------------------------------------
# Ensure a docker-container builder (supports --push and multi-platform).
# ---------------------------------------------------------------------------
if ! docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1; then
    echo ">> Creating buildx builder '${BUILDX_BUILDER}' (docker-container driver)"
    docker buildx create --name "$BUILDX_BUILDER" --driver docker-container >/dev/null
fi

if [[ "$PLATFORM" == *,* ]]; then
    echo ">> Multi-arch build for ${PLATFORM} — requires QEMU for non-native archs"
    echo "   (register once with: docker run --privileged --rm tonistiigi/binfmt --install all)"
fi

BUILD_ARGS=()
[[ "$NO_CACHE" == "1" ]] && BUILD_ARGS+=(--no-cache)
[[ -n "${PROVISION_DSR1:-}" ]] && BUILD_ARGS+=(--build-arg "PROVISION_DSR1=${PROVISION_DSR1}")

# Don't attach SLSA provenance attestations. buildx adds them by default on --push,
# which surfaces a spurious `unknown/unknown` entry on the GHCR package page and
# wraps even a single-arch build in an image index. Opting out yields a clean,
# single-manifest artifact; the source/revision/version annotations below preserve
# the "which commit built this" traceability the attestation would have carried.
BUILD_ARGS+=(--provenance=false)

# With --no-push we build to cache and discard (validation only). A multi-platform
# result cannot be --load-ed into the local store, so no output flag is added.
[[ "$PUSH" == "1" ]] && BUILD_ARGS+=(--push)

# OCI annotations for the GHCR package page (description + repo/commit provenance).
# --annotation requires the build to actually produce the component named by the
# level prefix, so gate on --push and pick the level by artifact shape: a multi-arch
# build pushes an image index (GHCR reads the description from the index); a
# single-arch build (provenance off, above) pushes a lone manifest, for which
# `index:` errors with "index annotations not supported for single platform export".
if [[ "$PUSH" == "1" ]]; then
    if [[ "$PLATFORM" == *,* ]]; then
        ANNOTATION_LEVEL="index"
    else
        ANNOTATION_LEVEL="manifest"
    fi
    IMAGE_DESCRIPTION="${IMAGE_DESCRIPTION:-endpoints client image (ref ${ENDPOINTS_REF}, commit ${SHORT_SHA})}"
    BUILD_ARGS+=(
        --annotation "${ANNOTATION_LEVEL}:org.opencontainers.image.source=https://github.com/mlcommons/endpoints"
        --annotation "${ANNOTATION_LEVEL}:org.opencontainers.image.revision=${SHORT_SHA}"
        --annotation "${ANNOTATION_LEVEL}:org.opencontainers.image.version=${ENDPOINTS_REF}"
        --annotation "${ANNOTATION_LEVEL}:org.opencontainers.image.description=${IMAGE_DESCRIPTION}"
    )
fi

BUILD_VERB=$([[ "$PUSH" == "1" ]] && echo "and pushing" || echo "(dry run, --no-push)")
echo ">> Building ${IMAGE}:${SHORT_SHA} (ref: ${ENDPOINTS_REF}) for ${PLATFORM} ${BUILD_VERB}"
docker buildx build \
    --builder "$BUILDX_BUILDER" \
    --platform "$PLATFORM" \
    -f "$DOCKERFILE" \
    -t "${IMAGE}:${SHORT_SHA}" \
    -t "${IMAGE}:${REF_TAG}" \
    ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} \
    .

echo
if [[ "$PUSH" == "1" ]]; then
    echo "Done. Pushed:"
    echo "  ${IMAGE}:${SHORT_SHA}"
    echo "  ${IMAGE}:${REF_TAG}"
    echo "Pull with:"
    echo "  docker pull ${IMAGE}:${SHORT_SHA}"
else
    echo "Done. Dry build succeeded for ${PLATFORM} (nothing pushed)."
fi
