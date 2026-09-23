#!/usr/bin/env bash
# push_image.sh — build (if needed) and push the LCB evaluator image to a registry.
#
# The lcb-service image is self-contained: the LiveCodeBench dataset is baked in at
# build time. Pushing the built image lets consumers pull-and-run with no HF_TOKEN
# and no rebuild (see pull_image.sh).
#
# The image is tagged <endpoints short SHA>-livecodebench — one immutable, self-identifying
# tag per build. The "-livecodebench" suffix is appended by _image_env.sh so the tag never
# collides with the client image's bare :<sha> in a shared registry package. LCB_IMAGE_TAG
# overrides the SHA part (the suffix is still appended). There is no moving channel tag.
#
# Prerequisites:
#   - docker login dhi.io            (base images are Docker Hardened Images)
#   - docker login <your registry>   (target for the push)
#
# Usage:
#   LCB_IMAGE_REGISTRY=myregistry.com/team HF_TOKEN=hf_xxx ./push_image.sh
#       Builds and pushes a MULTI-ARCH manifest (linux/amd64,linux/arm64) by default.
#   ... --platform linux/arm64                      # single arch (fast; no QEMU emulation)
#   ... --platform linux/amd64,linux/arm64          # explicit multi-arch (the default)
#   LCB_IMAGE_REGISTRY=... LCB_IMAGE_TAG=<sha> ./push_image.sh --no-build   # push existing local image
#   ... --force                                     # overwrite an existing :<sha> tag (default: refuse)
#
# All builds go through 'docker buildx' and push straight to the registry: buildx forces
# gzip layers so the image is enroot/pyxis-extractable (mlcommons/endpoints#467), and a
# multi-arch image cannot be loaded into the local docker store anyway. Building an arch
# other than the host's needs QEMU emulation registered once (needs --privileged):
#       docker run --privileged --rm tonistiigi/binfmt --install all
#   The dataset-generation step runs under emulation and is MUCH slower than native, so
#   prefer --platform <host-arch> when a single-arch image is enough.
#
# Environment variables: see _image_env.sh (LCB_IMAGE_REGISTRY required).
#   HF_TOKEN            required unless --no-build (passed as a BuildKit secret directly from the environment).
#   LCB_IMAGE_PLATFORM  target platform(s), e.g. linux/arm64 (default: linux/amd64,linux/arm64).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDX_BUILDER="lcb-buildx"

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -euo/d'
}

# Echo the (comma-separated) platforms in $2 that builder $1 cannot build, space-separated.
# The docker-container driver lists a platform only when its QEMU binfmt handler was
# registered on the host before the builder bootstrapped, so an unbuildable platform also
# flags a builder that predates QEMU registration and needs recreating.
unsupported_platforms() {
    local builder="$1" plat listing
    local -a missing=()
    listing="$(docker buildx inspect "$builder" --bootstrap 2>/dev/null | grep -i '^Platforms:')" || true
    local IFS=','
    for plat in $2; do
        plat="${plat// /}"
        { [[ -n "$listing" ]] && grep -Eq "(^|[ ,:])${plat}([ ,]|\$)" <<<"$listing"; } || missing+=("$plat")
    done
    printf '%s' "${missing[*]:-}"
}

platform_supported() { [[ -z "$(unsupported_platforms "$1" "$2")" ]]; }

NO_BUILD=0
FORCE=0
# Multi-arch by default: one publish runs on amd64 and arm64, and the buildx path forces
# gzip layers so the image is enroot/pyxis-extractable (mlcommons/endpoints#467). Override
# with --platform <spec> for a single arch (e.g. --platform linux/arm64 skips the slow QEMU
# emulation of the non-host arch).
PLATFORM="${LCB_IMAGE_PLATFORM:-linux/amd64,linux/arm64}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) NO_BUILD=1 ;;
        --force) FORCE=1 ;;
        --platform)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "error: --platform requires a value, e.g. linux/arm64" >&2
                exit 1
            fi
            PLATFORM="$2"
            shift
            ;;
        --platform=*) PLATFORM="${1#*=}" ;;
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

# ----------------------------------------------------------------------------
# Provenance: the endpoints repo commit this image is built from. It is both baked
# into the image config as an OCI label (see --build-arg below) AND used as the
# image tag — one immutable tag per build. A config LABEL (not a manifest
# --annotation) is deliberate: the buildx push sets oci-mediatypes=false, and
# Docker-media-type manifests have no annotations field, whereas config labels are
# representable in both formats and survive `docker inspect`. Scope the dirty check
# to this build context so unrelated edits elsewhere in the repo don't mark it -dirty.
# ----------------------------------------------------------------------------
ENDPOINTS_SHA="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "$ENDPOINTS_SHA" != "unknown" ]] && ! git -C "$SCRIPT_DIR" diff --quiet HEAD -- "$SCRIPT_DIR" 2>/dev/null; then
    ENDPOINTS_SHA="${ENDPOINTS_SHA}-dirty"
fi

# Tag the image by the endpoints commit SHA (LCB_IMAGE_TAG overrides). The auto SHA
# tag is only trustworthy when the tree is clean AND we build now, so when the tag is
# not given explicitly, refuse the cases where :<sha> would misrepresent the image:
#   - no git        → nothing to name the image by
#   - --no-build    → pushes a pre-built local image whose baked revision label may be
#                     a different commit than HEAD; :<sha> would lie about provenance
#   - dirty context → :<sha>-dirty is a moving, non-reproducible tag
# An explicit LCB_IMAGE_TAG is the escape hatch for all three. Set before sourcing so
# _image_env.sh builds LCB_IMAGE_REF from it.
if [[ -z "${LCB_IMAGE_TAG:-}" ]]; then
    if [[ "$ENDPOINTS_SHA" == "unknown" ]]; then
        echo "error: cannot determine a short SHA to tag the image (no git); set LCB_IMAGE_TAG." >&2
        exit 1
    fi
    if [[ "$NO_BUILD" -eq 1 ]]; then
        echo "error: --no-build pushes a pre-built local image whose baked revision may not match" >&2
        echo "       HEAD (${ENDPOINTS_SHA}); set LCB_IMAGE_TAG explicitly to name it." >&2
        exit 1
    fi
    if [[ "$ENDPOINTS_SHA" == *-dirty ]]; then
        echo "error: build context has uncommitted changes; :${ENDPOINTS_SHA} is a moving tag." >&2
        echo "       Commit the changes, or set LCB_IMAGE_TAG explicitly to push anyway." >&2
        exit 1
    fi
fi
export LCB_IMAGE_TAG="${LCB_IMAGE_TAG:-$ENDPOINTS_SHA}"

# shellcheck source=_image_env.sh
source "${SCRIPT_DIR}/_image_env.sh"
# shellcheck source=/dev/null
source "$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)/scripts/lib_registry.sh"

# Refuse to overwrite an existing remote tag unless --force: the SHA tag is meant to be
# immutable, so a second push would silently replace a published image. Delegates to the
# shared ref_exists_in_registry (fail CLOSED — anything other than a confirmed "absent"
# blocks). --force skips the check entirely.
if [[ "$FORCE" -eq 0 ]]; then
    rc=0; ref_exists_in_registry "$LCB_IMAGE_REF" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        echo "error: ${LCB_IMAGE_REF} already exists in the registry." >&2
        echo "       The SHA tag is meant to be immutable; re-run with --force to overwrite it." >&2
        exit 1
    elif [[ "$rc" -ne 1 ]]; then
        echo "error: could not verify whether ${LCB_IMAGE_REF} already exists (see above)." >&2
        echo "       Fix registry access (or install docker buildx), or re-run with --force to skip this check." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# --no-build publishes a pre-built local image (host arch only): push to a staging tag,
# verify, then promote onto the pinned tag (a plain push can't be force-compressed, so
# the immutable :<sha> is only made reachable after the gzip check passes).
# Every other invocation builds with buildx and pushes straight to the registry: buildx
# forces gzip layers (enroot/pyxis-safe, #467) and a multi-arch image cannot be
# `docker load`-ed into the local store anyway, so there is no separate build/tag step.
# ----------------------------------------------------------------------------
if [[ "$NO_BUILD" -eq 1 ]]; then
    # A pre-built local image can carry zstd base layers (containerd image store) that
    # enroot/pyxis cannot extract (#467), and this plain `docker push` applies no
    # force-compression. Verify BEFORE the pinned :<sha> tag becomes reachable: push to a
    # transient staging tag, assert its layers, then promote onto $LCB_IMAGE_REF only on
    # success — so a failed verify never poisons the immutable tag (which the guard above
    # then refuses to overwrite without --force). `imagetools create` copies layers by
    # digest (no blob re-upload); --prefer-index=false makes the pin the SAME manifest
    # digest that was verified (not a fresh 1-entry index), and it is re-asserted below.
    STAGING_REF="${LCB_IMAGE_REF%:*}:staging-${LCB_IMAGE_TAG}"
    echo ">> Tagging ${LCB_LOCAL_TAG} -> ${STAGING_REF} (staging)"
    docker tag "$LCB_LOCAL_TAG" "$STAGING_REF"

    echo ">> Pushing ${STAGING_REF} for verification"
    docker push "$STAGING_REF"
    assert_gzip_layers "$STAGING_REF" || exit 1

    echo ">> Verified; promoting ${STAGING_REF} -> ${LCB_IMAGE_REF}"
    docker buildx imagetools create --prefer-index=false --tag "$LCB_IMAGE_REF" "$STAGING_REF"
    assert_gzip_layers "$LCB_IMAGE_REF" || exit 1
    echo "   (transient ${STAGING_REF##*:} tag left in the registry; delete via your registry UI/API if desired.)"
else
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "error: HF_TOKEN is required to build the image (or pass --no-build to push an existing one)." >&2
        exit 1
    fi
    if ! docker buildx version >/dev/null 2>&1; then
        echo "error: 'docker buildx' is required but is not available." >&2
        exit 1
    fi

    # Ensure a docker-container builder exists (the default 'docker' driver can neither
    # push nor build a non-host platform). Creating it is safe and unprivileged.
    if ! docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1; then
        echo ">> Creating buildx builder '${BUILDX_BUILDER}' (docker-container driver)"
        docker buildx create --name "$BUILDX_BUILDER" --driver docker-container >/dev/null
    fi

    # An existing builder caches its emulated platforms from when it bootstrapped, so it
    # will not see QEMU handlers registered since. Recreate it once before giving up, so
    # a freshly-registered emulator is picked up instead of failing on a stale builder.
    if ! platform_supported "$BUILDX_BUILDER" "$PLATFORM"; then
        echo ">> Builder '${BUILDX_BUILDER}' lacks '${PLATFORM}'; recreating to re-detect QEMU emulators"
        docker buildx rm "$BUILDX_BUILDER" >/dev/null 2>&1 || true
        docker buildx create --name "$BUILDX_BUILDER" --driver docker-container >/dev/null
    fi

    # Still unavailable after a fresh builder ⇒ QEMU emulation is genuinely not registered.
    missing_platforms="$(unsupported_platforms "$BUILDX_BUILDER" "$PLATFORM")"
    if [[ -n "$missing_platforms" ]]; then
        echo "error: builder '${BUILDX_BUILDER}' cannot build: ${missing_platforms} — QEMU emulation for it is not registered on the host." >&2
        echo "       Register it once (needs --privileged), then re-run:" >&2
        echo "         docker run --privileged --rm tonistiigi/binfmt --install all" >&2
        echo "       Or build a single arch to skip emulation: --platform linux/$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')" >&2
        exit 1
    fi

    echo ">> Building ${LCB_IMAGE_REF} for ${PLATFORM} (endpoints ${ENDPOINTS_SHA}) and pushing (buildx) ..."
    docker buildx build \
        --builder "$BUILDX_BUILDER" \
        --platform "$PLATFORM" \
        -f "${SCRIPT_DIR}/lcb_serve.dockerfile" \
        --secret id=HF_TOKEN,env=HF_TOKEN \
        --build-arg "ENDPOINTS_SHA=${ENDPOINTS_SHA}" \
        -t "$LCB_IMAGE_REF" \
        --provenance=false \
        --output "type=image,push=true,compression=gzip,force-compression=true,oci-mediatypes=false" \
        "$SCRIPT_DIR"
    # Post-publish confirmation, not a gate: the build forces gzip layers so the pushed
    # image is enroot-safe by construction (#467). Known caveat: a transient inspect flake
    # here fails the job after the tag is live; re-run with --force to republish. (Only the
    # --no-build path needs the staging→verify→promote dance, since it can't force gzip.)
    assert_gzip_layers "$LCB_IMAGE_REF" || exit 1
fi

echo
echo "Pushed: ${LCB_IMAGE_REF}"
echo
echo "Done. Consumers can now pull with:"
echo "  LCB_IMAGE_REGISTRY=${LCB_IMAGE_REGISTRY%/} \\"
[[ "$LCB_IMAGE_NAME" != "lcb-service" ]] && echo "  LCB_IMAGE_NAME=${LCB_IMAGE_NAME} \\"
echo "  LCB_IMAGE_TAG=${LCB_IMAGE_TAG} \\"
echo "  ./pull_image.sh"
