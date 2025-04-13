#!/usr/bin/env python
"""
Debug startup script for MemeStrike trading bot
This will help identify why the real_solana_trader.py script isn't starting properly
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("startup")

def main():
    """Main startup function that attempts to run the trading bot"""
    # Display current working directory and files
    cwd = os.getcwd()
    logger.info(f"Current directory: {cwd}")
    
    # List all files in current directory
    files = os.listdir(cwd)
    logger.info(f"Files in directory: {files}")
    
    # Check if our main file exists
    if "real_solana_trader.py" in files:
        logger.info("Found real_solana_trader.py, attempting to import")
        try:
            import real_solana_trader
            logger.info("Successfully imported real_solana_trader")
            
            # Check if expected functions exist
            if hasattr(real_solana_trader, "run_trading_bot"):
                logger.info("Found run_trading_bot function, attempting to run")
                try:
                    real_solana_trader.run_trading_bot()
                except Exception as e:
                    logger.error(f"Error running trading bot: {str(e)}")
                    logger.exception("Full traceback:")
            else:
                logger.error("Missing run_trading_bot function in real_solana_trader.py")
        except Exception as e:
            logger.error(f"Error importing real_solana_trader: {str(e)}")
            logger.exception("Full traceback:")
    else:
        logger.error("real_solana_trader.py not found in current directory")
        
    # Check environment variables (without revealing sensitive data)
    for env_var in ["WALLET_PRIVATE_KEY", "INITIAL_AMOUNT"]:
        if env_var in os.environ:
            masked_value = "****" if env_var == "WALLET_PRIVATE_KEY" else os.environ.get(env_var)
            logger.info(f"Environment variable {env_var} is set to: {masked_value}")
        else:
            logger.warning(f"Required environment variable {env_var} is not set")
    
    # Keep the process running
    logger.info("Startup script completed, keeping process alive")
    try:
        # Keep the service running so Render doesn't restart it
        while True:
            import time
            time.sleep(60)
            logger.info("Still running...")
    except KeyboardInterrupt:
        logger.info("Process interrupted, shutting down")

if __name__ == "__main__":
    main()
