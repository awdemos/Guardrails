# Arbiter

Moderates input and output text with a local Arbiter sidecar
wrapping the Laya typed-decision model. One batched call asks five questions
(jailbreak, harmful, PII, off-topic nouls plus a severity score) and maps them onto an
allow / escalate / block three-band decision with configurable thresholds.

Arbiter runs inference locally, so no message text leaves the machine.

See the integration docs under
`docs/configure-rails/guardrail-catalog/community/arbiter.mdx`
for setup and configuration.
