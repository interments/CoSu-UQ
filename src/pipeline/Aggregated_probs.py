"""
Backward-compatible wrapper.

The CoT-UQ score aggregation implementation has been migrated to:
`baselines.cotuq_aggregate_scores`.
"""

from baselines.cotuq_aggregate_scores import main


if __name__ == "__main__":
    main()

