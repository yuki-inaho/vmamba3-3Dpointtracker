"""Demo entry point for release comparisons with an optional distilled student.

Re-render saved predictions with --stage render, or check generated media with
--stage audit. GPU inference requires --stage infer and an explicit --subset.
See README section 4.6 for prerequisites, outputs, and evaluation limitations.
Use --mode with-student --student-checkpoint to add a fourth column in a
separate Desktop output directory, preserving the original release videos.
The original compare_release_tracking.py CLI remains supported.
"""

from compare_release_tracking import main


if __name__ == "__main__":
    raise SystemExit(main())
