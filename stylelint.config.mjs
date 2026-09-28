/** @type {import("stylelint").Config} */
export default {
  extends: ["stylelint-config-standard"],
  referenceFiles: ["src/ui/styles/theme.css"],
  reportDescriptionlessDisables: true,
  reportInvalidScopeDisables: true,
  reportNeedlessDisables: true,
  maxWarnings: 0,
  rules: {
    "no-unknown-custom-properties": true,
    // Oxfmt owns whitespace; Stylelint owns CSS semantics and notation.
    "at-rule-empty-line-before": null,
    "comment-empty-line-before": null,
    "custom-property-empty-line-before": null,
    "declaration-empty-line-before": null,
    "rule-empty-line-before": null,
    // Independent components and mutually exclusive states need no global specificity order.
    "no-descending-specificity": null,
  },
};
