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

from typing import Optional

from pydantic import model_validator

from nemoguardrails.manifests.config_schema import Field, RailConfigBaseModel, RailConfigSpec, rail_field

ARBITER_URL_ENV = "ARBITER_URL"
ARBITER_API_KEY_ENV = "ARBITER_API_KEY"


class ArbiterConfig(RailConfigBaseModel):
    """Configuration for the Arbiter moderation rail."""

    base_url: Optional[str] = Field(
        default=None,
        description=(
            f"Base URL of the Arbiter sidecar. Falls back to the {ARBITER_URL_ENV} env var, then http://localhost:8010."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description=f"Optional Bearer token for the Arbiter sidecar. Falls back to the {ARBITER_API_KEY_ENV} env var.",
    )
    block_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Probability at or above which the message is blocked. Applies to the risk nouls "
            "(jailbreak, harmful, off_topic) and to the severity score normalized to [0, 1] "
            "(severity answers come back on a 0..3 scale and are divided by 3)."
        ),
    )
    review_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability at or above which the message is escalated for review instead of allowed.",
    )
    pii_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="PII noul at or above which context['arbiter_pii_detected'] is set.",
    )
    fail_closed: bool = Field(
        default=False,
        description="When True, block content on Arbiter connection or response errors instead of failing open.",
    )
    model: str = Field(
        default="auto",
        description="Arbiter checkpoint routing hint passed to /v1/systemone.",
    )

    @model_validator(mode="after")
    def _review_threshold_not_above_block_threshold(self) -> "ArbiterConfig":
        if self.review_threshold > self.block_threshold:
            raise ValueError(
                f"review_threshold ({self.review_threshold}) must not exceed block_threshold ({self.block_threshold})."
            )
        return self


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[ArbiterConfig],
        field_info=rail_field(
            default_factory=ArbiterConfig,
            description="Configuration for the Arbiter moderation rail.",
        ),
        exports={"ArbiterConfig": ArbiterConfig},
    )
