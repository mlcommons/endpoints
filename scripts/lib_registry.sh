#!/usr/bin/env bash
# lib_registry.sh — shared registry helpers for the image push scripts.
#
# Not meant to be executed directly. `source` it from a push script.

# assert_gzip_layers REF — fail unless every layer of REF (walking a manifest list)
# is gzip-compressed. enroot/pyxis routes a layer to its decompressor by media type
# and does not recognise the docker-namespaced zstd type
# (application/vnd.docker.image.rootfs.diff.tar.zstd) that buildx emits for zstd base
# layers under oci-mediatypes=false; it hands the raw blob to tar, which dies with
# "tar: This does not look like a tar archive" (mlcommons/endpoints#467). Reads registry
# metadata only (no blob pull), via `docker buildx imagetools inspect`. Fails CLOSED:
# any inspect error blocks the publish rather than silently passing.
assert_gzip_layers() {
    local ref="$1"
    docker buildx imagetools inspect "$ref" --raw >/dev/null 2>&1 \
        || { echo "error: cannot inspect ${ref} to verify layer compression." >&2; return 2; }
    # --raw yields either an image index (has .manifests) or a single manifest (has
    # .layers). Parse with python for robustness across both shapes and media-type
    # namespaces (docker + oci); descend one level for a manifest list.
    python3 - "$ref" <<'PY'
import sys, json, subprocess
ref = sys.argv[1]
base = ref.split("@", 1)[0]
def raw(r):
    return json.loads(subprocess.check_output(
        ["docker", "buildx", "imagetools", "inspect", r, "--raw"]))
def layer_types(man):
    return [layer["mediaType"] for layer in man.get("layers", [])]
top = raw(ref)
types = []
if top.get("manifests"):                       # image index / manifest list
    for child in top["manifests"]:
        plat = child.get("platform", {})
        if plat.get("os") == "unknown" or plat.get("architecture") == "unknown":
            continue                            # skip attestation manifests
        types += layer_types(raw(f"{base}@{child['digest']}"))
else:                                           # single-arch manifest
    types = layer_types(top)
bad = sorted({mt for mt in types if not mt.endswith("gzip")})
if bad:
    sys.stderr.write(
        f"error: {ref} has non-gzip layer(s): {', '.join(bad)}\n"
        "       enroot/pyxis cannot extract these (mlcommons/endpoints#467).\n"
        "       Rebuild via the buildx --platform path, which forces gzip.\n")
    sys.exit(1)
print(f">> Verified: all {len(types)} layers of {ref} are gzip (enroot-safe).")
PY
}

# ref_exists_in_registry REF — probe whether REF is already published, reading registry
# metadata only (no blob pull) via `docker buildx imagetools inspect`. Return codes:
#   0  present
#   1  definitely absent (registry reported not-found)
#   2  indeterminate (auth / network / tooling error) — output echoed to stderr
# Callers enforcing an immutable tag MUST treat 2 as "block" (fail CLOSED): never
# overwrite when existence can't be verified, or the guard silently no-ops on exactly
# the hosts/creds where it can't check.
ref_exists_in_registry() {
    local ref="$1" out
    if out="$(docker buildx imagetools inspect "$ref" 2>&1)"; then
        return 0
    fi
    # A missing binary / credential-helper / permission failure also contains "not found"
    # ("docker-credential-xxx: executable file not found") but is NOT an absent manifest —
    # classifying it as absent would let a push overwrite an immutable tag. Screen these to
    # indeterminate FIRST so the not-found match below can't fire on them (fail CLOSED).
    if grep -qiE 'executable file not found|command not found|no such file or directory|permission denied|credential' <<<"$out"; then
        printf '%s\n' "$out" >&2
        return 2
    fi
    # Registry "absent" phrasings: GHCR (`<ref>: not found`), OCI/Docker distribution
    # (`manifest unknown`, `name unknown`), and ECR (`name unknown … does not exist`).
    if grep -qiE 'not found|manifest unknown|manifest_unknown|name[ _]unknown|no such manifest|does not exist' <<<"$out"; then
        return 1
    fi
    printf '%s\n' "$out" >&2
    return 2
}
