# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    EnvVar,
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
)

ARBITER_CHECK_INPUT = ActionRef(
    name="arbiter_check_input",
    target="nemoguardrails.library.arbiter.actions:arbiter_check_input",
)

ARBITER_CHECK_OUTPUT = ActionRef(
    name="arbiter_check_output",
    target="nemoguardrails.library.arbiter.actions:arbiter_check_output",
)

RAIL = RailManifest(
    name="arbiter",
    metadata=RailMetadata(
        display_name="Arbiter",
        description=(
            "Moderates input and output text with a local Arbiter sidecar wrapping the Laya "
            "typed-decision model. The escalate band intentionally surfaces as a blocked outcome "
            "with metadata.escalated=true so a following self-check rail can take over."
        ),
        categories=("input", "output"),
        capabilities=("allow", "block", "content_safety", "detect_pii", "moderate"),
        tags=("local", "sidecar", "moderation"),
        docs_url="docs/configure-rails/guardrail-catalog/community/arbiter.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="arbiter",
            spec=ConfigSpecRef(target="nemoguardrails.library.arbiter.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=("arbiter moderation on input", "arbiter moderation on output"),
        ),
        actions=RailActions(refs=(ARBITER_CHECK_INPUT, ARBITER_CHECK_OUTPUT)),
        surfaces=(
            RailSurface(
                name="arbiter moderation on input",
                direction=RailDirection.INPUT,
                action=ARBITER_CHECK_INPUT,
                bindings=(Binding.context("user_message", "user_message"),),
            ),
            RailSurface(
                name="arbiter moderation on output",
                direction=RailDirection.OUTPUT,
                action=ARBITER_CHECK_OUTPUT,
                bindings=(Binding.context("bot_message", "bot_message"),),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="ARBITER_URL", required=False, description="Base URL of the Arbiter sidecar."),
                EnvVar(name="ARBITER_API_KEY", required=False, description="Optional Bearer token for the sidecar."),
            ),
            services=(ServiceRequirement(name="Arbiter sidecar", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            remote_services=(),
            data_retention=(
                "Inference location depends on the configured base_url; when pointed at a local "
                "sidecar (the default), no data leaves the machine."
            ),
        ),
    ),
)
