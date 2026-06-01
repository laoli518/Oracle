#!/usr/bin/env python3
import sys
from fine_tuning.folder_adapter_finetune import main
if "--preset" not in sys.argv:
    sys.argv.extend(["--preset", "lying_posture"])
if __name__ == "__main__":
    main()
