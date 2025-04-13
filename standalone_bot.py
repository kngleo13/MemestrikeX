#!/usr/bin/env python
"""
MemeStrike Bot - Fully Autonomous Trading Bot
This script executes REAL trades with ACTUAL MONEY on Solana blockchain
Fully autonomous with transaction signing and execution
"""
import os
import sys
import time
import json
import logging
import requests
import datetime
import random
import base64
from typing import Dict, Any, Optional, List, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("trading_bot")

# Check for required Solana SDK dependencies
try:
    import base58
    
    # Try multiple import paths for Solana modules with fallbacks
    try:
        from solana.rpc.api import Client
    except ImportError:
        try:
            from solders.rpc.api import Client
        except ImportError:
            logger.error("Cannot import Client from any known location")
            raise
    
    try:
        from solana.transaction import Transaction
    except ImportError:
        try:
            from solders.transaction import Transaction
        except ImportError:
            logger.error("Cannot import Transaction from any known location")
            raise
    
    try:
        from solana.keypair import Keypair
    except ImportError:
        try:
            from solders.keypair import Keypair
        except ImportError:
            try:
                from solana.transaction.keypair import Keypair
            except ImportError:
                logger.error("Cannot import Keypair from any known location")
                raise
    
    try:
        from solana.publickey import PublicKey
    except ImportError:
        try:
            from solders.pubkey import Pubkey as PublicKey
        except ImportError:
            logger.error("Cannot import PublicKey from any known location")
            raise
    
    try:
        from solana.rpc.types import TxOpts
    except ImportError:
        # Create a compatible TxOpts class if import fails
        class TxOpts:
            def __init__(self, skip_preflight=False, preflight_commitment="confirmed"):
                self.skip_preflight = skip_preflight
                self.preflight_commitment = preflight_commitment
                
except ImportError as e:
    logger.error(f"Missing required dependency: {str(e)}")
    logger.error("Please ensure solana and base58 packages are installed: pip install solana==0.30.2 base58==2.1.1")
    sys.exit(1)

# Import the autonomous trader
from autonomous_trader import FullyAutonomousTrader

def main():
    """Main entry point for the standalone bot"""
    logger.info("Starting MemeStrike standalone trading bot")
    
    try:
        # Initialize trader
        trader = FullyAutonomousTrader()
        
        # Run the trading bot
        trader.run_trading_bot()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
