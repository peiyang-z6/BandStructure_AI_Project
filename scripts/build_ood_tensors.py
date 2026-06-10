"""
Command-line entry point for OOD tensor construction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ood_tensor_builder import main


if __name__ == "__main__":
    main()
