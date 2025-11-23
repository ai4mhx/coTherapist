#!/usr/bin/env python3
"""
Interactive chat interface for coTherapist.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coTherapist.pipeline import CoTherapistPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Chat with coTherapist")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable retrieval augmentation"
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable agentic reasoning"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CoTherapistPipeline(args.config)
    
    # Update config based on args
    if args.no_rag:
        pipeline.config['retrieval']['enabled'] = False
    if args.no_reasoning:
        pipeline.config['agentic']['enabled'] = False
    
    # Start chat
    pipeline.chat()


if __name__ == "__main__":
    main()
