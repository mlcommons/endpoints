#!/usr/bin/env bash
# _image_env.sh — shared image-reference resolution for the LCB push/pull scripts.
#
# Not meant to be executed directly. `source` it from push_image.sh / pull_image.sh.
# It is the single source of truth for the remote/local image reference so the two
# scripts cannot drift.
#
# Inputs (environment variables):
#   LCB_IMAGE_REGISTRY  (required)  registry + namespace, e.g. myregistry.com/team
#   LCB_IMAGE_NAME      (optional)  image repo name           (default: lcb-service)
#   LCB_IMAGE_TAG       (required)  tag — push_image.sh defaults it to the endpoints short SHA
#   LCB_LOCAL_TAG       (optional)  local tag used by run/scorer (default: lcb-service:latest)
#
# Exports:
#   LCB_IMAGE_REF  = ${LCB_IMAGE_REGISTRY}/${LCB_IMAGE_NAME}:${LCB_IMAGE_TAG}
#   LCB_LOCAL_TAG  (defaulted if unset)

if [[ -z "${LCB_IMAGE_REGISTRY:-}" ]]; then
    echo "error: LCB_IMAGE_REGISTRY is not set." >&2
    echo "       Set it to your registry + namespace, e.g.:" >&2
    echo "         export LCB_IMAGE_REGISTRY=myregistry.com/team" >&2
    return 1 2>/dev/null || exit 1
fi

LCB_IMAGE_NAME="${LCB_IMAGE_NAME:-lcb-service}"
# Images are tagged by the endpoints commit SHA (one immutable tag per build), so
# there is no channel default. push_image.sh sets this to the SHA automatically;
# a consumer pulling must name the specific build.
if [[ -z "${LCB_IMAGE_TAG:-}" ]]; then
    echo "error: LCB_IMAGE_TAG is not set." >&2
    echo "       push_image.sh defaults it to the endpoints commit SHA; for pull, set it to" >&2
    echo "       the build you want, e.g. LCB_IMAGE_TAG=\$(git rev-parse --short HEAD)" >&2
    return 1 2>/dev/null || exit 1
fi
LCB_LOCAL_TAG="${LCB_LOCAL_TAG:-lcb-service:latest}"

# Strip any trailing slash on the registry to avoid a double slash in the ref.
LCB_IMAGE_REF="${LCB_IMAGE_REGISTRY%/}/${LCB_IMAGE_NAME}:${LCB_IMAGE_TAG}"

export LCB_IMAGE_NAME LCB_IMAGE_TAG LCB_LOCAL_TAG LCB_IMAGE_REF
