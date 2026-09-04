#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and push the benchmark's first 200 ARM64 SWE-bench images."""

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker
from datasets import load_dataset
from swebench.harness.docker_build import build_env_images, build_instance_image
from swebench.harness.test_spec.test_spec import make_test_spec

DATASET = "princeton-nlp/SWE-bench_Verified"
REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
IDS_SHA256 = "4b562a390c5bfe39a9b00d503ebce1d8b7d1c60f6b761617317a0c507534f40f"
REGISTRY = os.getenv("REGISTRY", "").rstrip("/")
TAG = "v4.1.0-arm64"
WORKERS = int(os.getenv("WORKERS", "16"))
COMPAT_IDS = {"django__django-10097", "django__django-15103"}
# This old Django runner consumes unsorted os.listdir("/testbed/tests"). The ARM
# image enumerates it differently from x86, exposing leaked test state in
# generic_inline_admin. Replay the recorded x86 order; unlisted entries still run.
DJANGO_10097_TEST_ORDER = """\
template_tests delete shortcuts fixtures admin_default_site delete_regress distinct_on_fields from_db_value
mutually_referential empty admin_autodiscover model_inheritance_regress null_fk middleware_exceptions context_processors bash_completion
files queries datatypes app_loading one_to_one view_tests responses httpwrappers
extra_regress model_regress select_related many_to_one file_uploads migrations2 auth_tests proxy_models
base transactions builtin_server update_only_fields lookup custom_columns model_formsets_regress proxy_model_inheritance
decorators contenttypes_tests prefetch_related m2m_through_regress transaction_hooks deprecation order_with_respect_to utils_tests
modeladmin redirects_tests get_object_or_404 admin_widgets basic managers_regress humanize_tests custom_managers
migration_test_data_persistence defer string_lookup many_to_one_null datetimes syndication_tests model_options m2m_and_m2o
settings_tests known_related_objects model_indexes field_deconstruction test_exceptions shell filtered_relation csrf_tests
handlers admin_filters dispatch sites_tests model_package conditional_processing null_queries test_client
admin_ordering custom_migration_operations annotations properties messages_tests absolute_url_overrides get_earliest_or_latest expressions
test_client_regress str generic_views aggregation_regress choices or_lookups m2m_signals i18n
update admin_custom_urls user_commands generic_inline_admin admin_changelist logging_tests field_subclassing urlpatterns_reverse
invalid_models_tests m2m_recursive signed_cookies_tests generic_relations_regress m2m_through m2m_intermediary db_typecasts select_related_regress
raw_query mail custom_methods fixtures_regress aggregation postgres_tests admin_views sessions_tests
custom_pk ordering nested_foreign_keys max_lengths null_fk_ordering sites_framework inspectdb expressions_window
no_models template_loader defer_regress custom_lookups many_to_many introspection swappable_models model_meta
get_or_create force_insert_update schema expressions_case generic_relations field_defaults unmanaged_models admin_docs
pagination admin_scripts m2o_recursive servers project_template template_backends dates reserved_names
timezones indexes model_fields admin_checks staticfiles_tests foreign_object check_framework forms_tests
bulk_create file_storage multiple_database urlpatterns reverse_lookup select_related_onetoone admin_inlines sitemaps_tests
serializers test_utils model_inheritance m2m_regress apps dbshell resolve_url model_forms
queryset_pickle select_for_update admin_utils flatpages_tests test_runner validators signing inline_formsets
cache fixtures_model_package signals model_formsets middleware wsgi admin_registration requests
version m2m_multiple migrate_signals validation save_delete_hooks db_functions backends migrations
""".split()


def arm_compat(spec):
    script = "\n".join(spec.env_script_list)
    script = script.replace("python=3.5 -y", "python=3.6 -y")
    script = script.replace(
        "python=3.6 setuptools==38.2.4 -y\nconda activate testbed",
        "python=3.6 -y\nconda activate testbed\npython -m pip install setuptools==38.2.4",
    )
    spec.env_script_list = script.splitlines()
    spec.repo_script_list = [
        command.replace(
            "python -m pip install -e .[test] --verbose",
            "python -m pip install jinja2==3.1.6 cython==0.29.36\n"
            "python -m pip install -e .[test] --no-build-isolation --verbose",
        )
        for command in spec.repo_script_list
    ]
    if spec.instance_id == "django__django-15103":
        spec.repo_script_list.append(
            "conda install python=3.9.20 -y && conda clean -afy"
        )
    if spec.instance_id == "django__django-10097":
        spec.repo_script_list.append(
            "cat > /opt/miniconda3/envs/testbed/lib/python3.6/site-packages/"
            "sitecustomize.py <<'PY'\n"
            "import os\n\n"
            "_listdir = os.listdir\n"
            f"_order = {DJANGO_10097_TEST_ORDER!r}\n"
            "_order_set = set(_order)\n\n"
            'def listdir(path="."):\n'
            "    entries = _listdir(path)\n"
            '    if os.path.abspath(path) != "/testbed/tests":\n'
            "        return entries\n"
            "    present = set(entries)\n"
            "    ordered = [entry for entry in _order if entry in present]\n"
            "    ordered.extend(entry for entry in entries if entry not in _order_set)\n"
            "    return ordered\n\n"
            "os.listdir = listdir\n"
            "PY"
        )
    return spec


def image_exists(name: str) -> bool:
    return (
        subprocess.run(
            ["docker", "manifest", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def main() -> None:
    if not REGISTRY:
        raise SystemExit("REGISTRY must identify the destination container registry")

    dataset = load_dataset(DATASET, split="test", revision=REVISION)
    rows = [dict(row) for row in dataset.select(range(200))]
    ids = [row["instance_id"] for row in rows]
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if len(ids) != 200 or digest != IDS_SHA256:
        raise SystemExit(
            "The pinned dataset no longer matches the benchmark's first 200 tasks"
        )

    client = docker.from_env()
    specs = [
        arm_compat(
            make_test_spec(
                row,
                arch="arm64",
                base_image_tag=TAG,
                env_image_tag=TAG,
                instance_image_tag=TAG,
            )
        )
        for row in rows
    ]
    pending = [
        s
        for s in specs
        if s.instance_id in COMPAT_IDS
        or not image_exists(f"{REGISTRY}/{s.instance_image_key}")
    ]
    if not pending:
        print("All 200 images are already in the registry")
        return

    _, failed = build_env_images(client, pending, max_workers=WORKERS)
    if failed:
        raise SystemExit(f"Failed to build {len(failed)} environment images")

    def publish(spec):
        remote = f"{REGISTRY}/{spec.instance_image_key}"
        build_instance_image(spec, client, None, spec.instance_id in COMPAT_IDS)
        image = client.images.get(spec.instance_image_key)
        if image.attrs["Architecture"] not in {"arm64", "aarch64"}:
            raise RuntimeError(f"Non-ARM image built for {spec.instance_id}")
        subprocess.run(["docker", "tag", spec.instance_image_key, remote], check=True)
        subprocess.run(["docker", "push", remote], check=True)
        return remote

    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(publish, spec): spec.instance_id for spec in pending}
        for future in as_completed(futures):
            try:
                print(f"PUSHED {future.result()}", flush=True)
            except Exception as exc:
                errors.append(futures[future])
                print(f"FAILED {futures[future]}: {exc}", flush=True)
    if errors:
        raise SystemExit(f"Failed images ({len(errors)}): {' '.join(errors)}")


if __name__ == "__main__":
    main()
