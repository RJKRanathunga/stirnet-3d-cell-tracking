# Findings

The production integration and synthetic tests establish that a geometrically
complete, unrepresented thin ellipsoid can add a supplemental marker while
effective EDT markers remain unchanged. Sphere, elongated-cell, irregular, and
arbitrary-count controls are covered by automated tests.

Four saved `missed_merge` components were replayed read-only as a bounded smoke
check. The structural counts were:

| Scene/frame | Effective EDT | Supplemental | Final |
| --- | ---: | ---: | ---: |
| 001 / 9 | 1 | 1 | 2 |
| 002 / 0 | 1 | 1 | 2 |
| 003 / 6 | 1 | 2 | 3 |
| 004 / 5 | 1 | 1 | 2 |

These counts confirm production activation on saved cases; they do not prove
biological correctness or improvement. The scenes still require visual review
of caps, body fits, watershed boundaries, and temporal identity. Thresholds
remain provisional, and case-level false-positive/false-negative findings are
required before any real-data improvement claim is made.
