#!/usr/bin/env python
"""
MemeStrike Bot - Production-Ready Real Money Trading Bot
This script creates REAL trade links using Jupiter Exchange
"""
import os
import sys
import time
import json
import logging
import requests
import datetime
import random
from typing import Dict, Any, Optional, List, Union, Tuple
import webbrowser
import base64

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

class JupiterLinkGenerator:
    """
    Generate Jupiter swap links for real trades
    """
    
    def __init__(self):
        """Initialize the trading bot with wallet and configuration"""
        logger.info("Initializing MemeStrike Trading Bot Link Generator")
        
        # Set default total trading amount
        self.total_amount = 0.1  # Default total amount (in SOL)
        amount_str = os.environ.get('INITIAL_AMOUNT', '0.1')
        try:
            # Remove any dollar signs and convert to float
            self.total_amount = float(amount_str.replace('$', '').strip())
            logger.info(f"Total trading amount set to: {self.total_amount} SOL")
        except ValueError:
            logger.warning(f"Could not parse amount '{amount_str}', using default: {self.total_amount}")
        
        # Set up the 6 trading lots
        self.num_lots = 6
        # Each lot gets an equal portion of the total amount
        self.lot_amount = self.total_amount / self.num_lots
        logger.info(f"Trading with {self.num_lots} lots of {self.lot_amount:.6f} SOL each")
        
        # Initialize amount for current trade
        self.amount = self.lot_amount
        
        # Initialize Solana RPC endpoint
        self.solana_rpc = "https://api.mainnet-beta.solana.com"
        
        # Initialize Jupiter API
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        
        # Get wallet address - either from env var or input
        self.wallet_address = os.environ.get('WALLET_ADDRESS')
        if not self.wallet_address:
            # Try to get wallet address from private key if provided
            private_key = os.environ.get('WALLET_PRIVATE_KEY')
            if private_key:
                try:
                    # Try to decode and get public key from private key
                    import base58
                    if not all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in private_key):
                        logger.warning("Private key contains invalid characters for base58 encoding")
                        private_key = ''.join(c for c in private_key 
                                           if c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
                    
                    private_key_bytes = base58.b58decode(private_key)
                    
                    if len(private_key_bytes) == 64:
                        # This is a full keypair (private key + public key)
                        public_key = private_key_bytes[32:]
                        self.wallet_address = base58.b58encode(public_key).decode('ascii')
                    else:
                        logger.warning("Couldn't extract wallet address from private key")
                except Exception as e:
                    logger.error(f"Error extracting wallet address from private key: {str(e)}")
            
            if not self.wallet_address:
                self.wallet_address = "AFbjwrMoKJ8LwYfaw8jLvjrfh1hYqZzdjDpWMu7LJLLH"  # Use previous address if available
        
        logger.info(f"Using wallet address: {self.wallet_address}")
        
        # Token list - only tokens verified working
        self.tokens = [
            "BONK", "WIF"  # Only using tokens that are working reliably in the logs
        ]
        
        # Token mint addresses (for actual blockchain transactions)
        self.token_addresses = {
            "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
            "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
            "SOL": "So11111111111111111111111111111111111111112"
        }
        
        # Stats
        self.trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_spent = 0.0
        self.start_time = datetime.datetime.now()
        
        # Create trades history file if it doesn't exist
        if not os.path.exists("trades_history.json"):
            with open("trades_history.json", 'w') as f:
                json.dump([], f)
            logger.info("Created empty trades history file")
            
        logger.info("Trading bot initialization complete")
    
    def check_wallet_balance(self) -> Optional[float]:
        """
        Check wallet balance in SOL
        
        Returns:
            Float balance in SOL or None if error
        """
        try:
            # Solana RPC endpoint
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [self.wallet_address]
            }
            
            response = requests.post(
                self.solana_rpc,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "value" in data["result"]:
                    # Convert lamports to SOL (1 SOL = 1,000,000,000 lamports)
                    balance = data["result"]["value"] / 1000000000.0
                    return balance
            
            logger.error(f"Failed to get wallet balance: {response.text}")
            
        except Exception as e:
            logger.error(f"Error checking wallet balance: {str(e)}")
        
        return None
    
    def generate_trade_link(self, token_symbol: str) -> Dict[str, Any]:
        """
        Generate a Jupiter prepaid swap link for a real trade
        
        Args:
            token_symbol: Token symbol to trade (e.g., "BONK")
            
        Returns:
            Dict with trade link result
        """
        logger.info(f"🔴 GENERATING TRADE LINK for {token_symbol} with {self.amount:.6f} SOL")
        
        # Check if token is supported
        if token_symbol not in self.token_addresses:
            logger.error(f"Unknown token symbol: {token_symbol}")
            return {"success": False, "error": "Unknown token symbol"}
        
        # Get token mint address
        token_address = self.token_addresses[token_symbol]
        sol_address = "So11111111111111111111111111111111111111112"
        
        # Convert amount to lamports
        amount_lamports = int(self.amount * 1000000000)
        
        try:
            # Step 1: Get quote from Jupiter
            logger.info("Getting quote from Jupiter Exchange")
            try:
                quote_response = requests.get(
                    f"{self.jupiter_api}/quote",
                    params={
                        "inputMint": sol_address,
                        "outputMint": token_address,
                        "amount": str(amount_lamports),
                        "slippageBps": "250"  # 2.5% slippage for faster entries
                    },
                    timeout=30
                )
                
                if quote_response.status_code != 200:
                    logger.error(f"Failed to get quote: {quote_response.text}")
                    return {"success": False, "error": f"Failed to get quote: {quote_response.text}"}
                
                quote_data = quote_response.json()
                
                # Validate the quote data
                if 'outAmount' not in quote_data:
                    logger.error(f"Invalid quote data: {quote_data}")
                    return {"success": False, "error": "Invalid quote data from Jupiter API"}
                
                output_amount = int(quote_data['outAmount']) / 1e9  # Convert to decimal units
                logger.info(f"Quote received: {self.amount} SOL -> {output_amount} {token_symbol}")
            except requests.RequestException as e:
                logger.error(f"Error getting quote from Jupiter: {str(e)}")
                return {"success": False, "error": f"Error getting quote: {str(e)}"}
            
            # Step 2: Generate the prepaid swap link
            try:
                # The prepaid swap link is a much simpler approach
                base_url = "https://jup.ag/swap"
                
                # Create the parameters
                params = {
                    "inputMint": sol_address,
                    "outputMint": token_address,
                    "amount": str(amount_lamports),
                    "slippage": 2.5,  # 2.5% slippage
                }
                
                # Build the URL
                query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                swap_url = f"{base_url}?{query_string}"
                
                # Generate a simpler direct URL
                direct_url = f"https://jup.ag/swap/SOL-{token_symbol}?amount={self.amount}&slippage=2.5"
                
                # Log the link
                logger.info(f"✅ Generated Jupiter swap link: {direct_url}")
                logger.info(f"Please open this link in a browser to execute the trade")
                
                # Store in the history file
                trade_data = {
                    "success": True,
                    "token": token_symbol,
                    "input_amount_sol": self.amount,
                    "output_token": token_symbol,
                    "swap_url": direct_url,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Update stats
                self.successful_trades += 1
                
                # Record the trade
                self._record_trade_to_file(trade_data)
                
                return trade_data
                
            except Exception as e:
                logger.error(f"Error generating swap link: {str(e)}")
                return {"success": False, "error": f"Error generating swap link: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Error generating trade link: {str(e)}")
            self.failed_trades += 1
            return {"success": False, "error": str(e)}
    
    def _record_trade_to_file(self, trade_data: Dict[str, Any]) -> None:
        """
        Record trades to a file for tracking
        
        Args:
            trade_data: Trade data to record
        """
        try:
            filename = "trades_history.json"
            
            # Read existing trades
            trades = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        content = f.read().strip()
                        if content:  # Check if the file is not empty
                            trades = json.loads(content)
                        else:
                            trades = []
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing trades file: {str(e)}")
                    trades = []
                except Exception as e:
                    logger.error(f"Error reading trades file: {str(e)}")
                    trades = []
            
            # Add new trade
            trades.append(trade_data)
            
            # Write back to file
            with open(filename, 'w') as f:
                json.dump(trades, f, indent=2)
            
            logger.info(f"Trade recorded to {filename}")
        except Exception as e:
            logger.error(f"Error recording trade to file: {str(e)}")
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """
        Get current trading statistics
        
        Returns:
            Dict with trading stats
        """
        runtime = datetime.datetime.now() - self.start_time
        runtime_hours = runtime.total_seconds() / 3600
        
        # Calculate trades per hour
        trades_per_hour = 0
        if runtime_hours > 0:
            trades_per_hour = self.trades_executed / runtime_hours
        
        # Get current balance
        current_balance = self.check_wallet_balance()
        
        return {
            "total_trades_executed": self.trades_executed,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": (self.successful_trades / self.trades_executed * 100) if self.trades_executed > 0 else 0,
            "runtime_hours": runtime_hours,
            "trades_per_hour": trades_per_hour,
            "trades_per_day": trades_per_hour * 24,
            "current_sol_balance": current_balance,
            "wallet_address": self.wallet_address
        }
    
    def print_trading_stats(self) -> None:
        """Print current trading statistics"""
        stats = self.get_trading_stats()
        
        logger.info("=== TRADING STATISTICS ===")
        logger.info(f"Total links generated: {stats['total_trades_executed']}")
        logger.info(f"Successful generations: {stats['successful_trades']}")
        logger.info(f"Failed generations: {stats['failed_trades']}")
        logger.info(f"Success rate: {stats['success_rate']:.2f}%")
        logger.info(f"Runtime: {stats['runtime_hours']:.2f} hours")
        logger.info(f"Links per hour: {stats['trades_per_hour']:.2f}")
        logger.info(f"Current SOL balance: {stats['current_sol_balance']:.6f}")
        logger.info(f"Wallet address: {stats['wallet_address']}")
        logger.info(f"Solscan wallet link: https://solscan.io/account/{stats['wallet_address']}")
        logger.info("=========================")
    
    def run_link_generator(self) -> None:
        """
        Main loop generating Jupiter swap links for real trading
        Runs indefinitely until interrupted
        """
        logger.info("🚀 Starting MemeStrike Trading Link Generator")
        logger.info(f"💰 Trading with wallet: {self.wallet_address}")
        logger.info(f"💼 Trading strategy: {self.num_lots} concurrent lots with {self.lot_amount:.6f} SOL each")
        
        # Verify wallet link
        logger.info(f"View your wallet on Solscan: https://solscan.io/account/{self.wallet_address}")
        
        # Main trading loop
        while True:
            try:
                # Check balance periodically
                balance = self.check_wallet_balance()
                if balance is None:
                    logger.error("Failed to get wallet balance. Will retry in 30 seconds")
                    time.sleep(30)
                    continue
                
                logger.info(f"Current wallet balance: {balance:.6f} SOL")
                
                # Generate links for each lot
                for lot_idx in range(self.num_lots):
                    # Amount for this lot
                    self.amount = self.lot_amount
                    
                    # Select a token to trade
                    token_symbol = random.choice(self.tokens)
                    
                    # Generate trade link
                    self.trades_executed += 1
                    logger.info(f"Generating trade link #{self.trades_executed} - Lot #{lot_idx+1} for {token_symbol}")
                    result = self.generate_trade_link(token_symbol)
                    
                    if result["success"]:
                        swap_url = result.get("swap_url", "")
                        logger.info(f"✅ LINK #{self.trades_executed} GENERATED: Lot #{lot_idx+1} {result['token']}")
                        logger.info(f"LINK: {swap_url}")
                        
                        # Save to file
                        output_file = f"trade_links_{datetime.datetime.now().strftime('%Y%m%d')}.txt"
                        with open(output_file, "a") as f:
                            f.write(f"{datetime.datetime.now().isoformat()} - {token_symbol} - {self.amount} SOL: {swap_url}\n")
                    else:
                        logger.error(f"❌ Link generation #{self.trades_executed} failed: Lot #{lot_idx+1} {result.get('error', 'Unknown error')}")
                    
                    # Print trading stats periodically
                    if self.trades_executed % 10 == 0:
                        self.print_trading_stats()
                    
                    # Small sleep between generations
                    time.sleep(1)
                
                # Print a summary
                logger.info("=== SUMMARY OF GENERATED TRADE LINKS ===")
                logger.info(f"Generated {self.num_lots} trade links")
                logger.info(f"Links saved to: trade_links_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
                logger.info("Please click on these links to execute trades manually")
                logger.info("====================================")
                
                # Wait a bit before next batch
                logger.info("Waiting 5 minutes before generating next batch of links...")
                time.sleep(300)  # Wait 5 minutes
            
            except Exception as e:
                logger.error(f"Error in link generation loop: {str(e)}")
                logger.exception("Exception details:")
                # Sleep a bit before retry
                time.sleep(10)


def main():
    """Main entry point"""
    logger.info("Starting MemeStrike trading link generator")
    
    try:
        # Initialize generator
        generator = JupiterLinkGenerator()
        
        # Start generation loop
        generator.run_link_generator()
    
    except KeyboardInterrupt:
        logger.info("Link generator stopped by user")
    except Exception as e:
        logger.error(f"Error starting link generator: {str(e)}")
        logger.exception("Exception details:")


if __name__ == "__main__":
    main()
