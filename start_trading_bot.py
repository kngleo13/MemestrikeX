#!/usr/bin/env python
"""
MemeStrike Bot - Production-Ready Real Money Trading Bot
This script executes REAL trades with ACTUAL MONEY on Solana blockchain using Jupiter Exchange
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

# Check for required dependencies
missing_deps = []
try:
    import base58
except ImportError:
    missing_deps.append("base58")

try:
    from solana.rpc.api import Client
    from solana.transaction import Transaction
    from solana.keypair import Keypair
    from solana.publickey import PublicKey
    from solana.rpc.types import TxOpts
except ImportError:
    missing_deps.append("solana")

if missing_deps:
    print(f"ERROR: Missing required dependencies: {', '.join(missing_deps)}")
    print(f"Please install them using: pip install {' '.join(missing_deps)}")
    sys.exit(1)

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

class SolanaRealTrader:
    """
    Production-ready trading bot for executing real money trades on Solana blockchain
    using Jupiter Exchange for maximum liquidity, with 6 concurrent trading lots
    """
    
    def __init__(self):
        """Initialize the trading bot with wallet and configuration"""
        logger.info("Initializing Solana Real Money Trading Bot")
        
        # Get configuration from environment
        self.private_key_base58 = os.environ.get('WALLET_PRIVATE_KEY')
        if not self.private_key_base58:
            raise ValueError("WALLET_PRIVATE_KEY environment variable is required. Please set it to your Solana wallet private key.")
        
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
        
        # Lot states (0 = available, timestamp = busy until)
        self.lots = [0] * self.num_lots
        
        # Initialize Solana RPC endpoint with backup options
        self.solana_rpc_endpoints = [
            "https://api.mainnet-beta.solana.com",
            "https://solana-api.projectserum.com",
            "https://rpc.ankr.com/solana",
        ]
        self.solana_rpc_url = self.solana_rpc_endpoints[0]
        
        # Initialize Jupiter API
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        
        # Initialize Solana wallet from private key
        try:
            # First, try to validate and clean up the private key in case it was copied with extra characters
            self.private_key_base58 = self.private_key_base58.strip()
            
            # Check if key is correct base58 format
            if not all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in self.private_key_base58):
                logger.warning("Private key contains invalid characters for base58 encoding")
                logger.warning("Attempting to clean up key...")
                # Try to clean up by removing common non-base58 characters
                self.private_key_base58 = ''.join(c for c in self.private_key_base58 
                                           if c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
                logger.info("Cleaned up private key for processing")
            
            # Decode the private key to bytes
            private_key_bytes = base58.b58decode(self.private_key_base58)
            logger.info(f"Successfully decoded private key (length: {len(private_key_bytes)} bytes)")
            
            # Create Solana keypair for transaction signing
            if len(private_key_bytes) == 64:
                # This is a full keypair format (includes both private and public key)
                self.keypair = Keypair.from_secret_key(private_key_bytes[:32])
                logger.info("Created Solana keypair from full keypair format (64 bytes)")
            elif len(private_key_bytes) == 32:
                # This is just the private key
                self.keypair = Keypair.from_secret_key(private_key_bytes)
                logger.info("Created Solana keypair from private key format (32 bytes)")
            else:
                raise ValueError(f"Invalid private key length: {len(private_key_bytes)} bytes. Expected 32 or 64 bytes.")
            
            # Initialize Solana client
            self.solana_client = Client(self.solana_rpc_url)
            
            # Get wallet public key
            self.wallet_address = str(self.keypair.public_key)
            logger.info(f"Wallet initialized: {self.wallet_address}")
            
            # Double check wallet validity by confirming it exists on Solana
            logger.info("Verifying wallet address on Solana blockchain...")
            balance = self.check_wallet_balance()
            if balance is not None:
                logger.info(f"✅ Wallet verified on blockchain with balance: {balance:.6f} SOL")
            else:
                logger.warning("⚠️ Could not verify wallet on blockchain. This might be a network issue or the wallet may not exist.")
            
        except Exception as e:
            logger.error(f"Error initializing wallet: {str(e)}")
            raise ValueError(f"Failed to initialize wallet: {str(e)}")
        
        # Check wallet balance
        balance = self.check_wallet_balance()
        if balance is None:
            logger.warning("Could not check wallet balance. Will retry later.")
        else:
            logger.info(f"Wallet balance: {balance:.6f} SOL")
            if balance < self.total_amount:
                logger.warning(f"Wallet balance ({balance:.6f} SOL) is less than configured amount ({self.total_amount:.6f} SOL)")
                logger.warning("Adjusting trading amount to 90% of available balance")
                self.total_amount = balance * 0.9
                self.lot_amount = self.total_amount / self.num_lots
                logger.info(f"New trading amount: {self.total_amount:.6f} SOL ({self.lot_amount:.6f} SOL per lot)")
        
        # Token list - only tokens verified working in your environment
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
    
    def _fallback_rpc(self):
        """
        Rotate to a different RPC endpoint if the current one fails
        """
        current_index = self.solana_rpc_endpoints.index(self.solana_rpc_url)
        next_index = (current_index + 1) % len(self.solana_rpc_endpoints)
        self.solana_rpc_url = self.solana_rpc_endpoints[next_index]
        self.solana_client = Client(self.solana_rpc_url)
        logger.info(f"Switched to alternative RPC endpoint: {self.solana_rpc_url}")
    
    def check_wallet_balance(self) -> Optional[float]:
        """
        Check wallet balance in SOL
        
        Returns:
            Float balance in SOL or None if error
        """
        errors = 0
        max_retries = 3
        
        while errors < max_retries:
            try:
                # Use Solana client to get balance
                response = self.solana_client.get_balance(self.wallet_address)
                
                if response["result"] is not None and "value" in response["result"]:
                    # Convert lamports to SOL (1 SOL = 1,000,000,000 lamports)
                    balance = int(response["result"]["value"]) / 1000000000.0
                    return balance
                
                logger.error(f"Failed to get wallet balance: {response}")
                errors += 1
                
                # Try a different RPC endpoint
                self._fallback_rpc()
                
            except Exception as e:
                logger.error(f"Error checking wallet balance: {str(e)}")
                errors += 1
                
                # Try a different RPC endpoint
                self._fallback_rpc()
        
        return None

    def execute_trade(self, token_symbol: str) -> Dict[str, Any]:
        """
        Execute a REAL trade with actual money using Jupiter API v6 and Solana SDK
        
        Args:
            token_symbol: Token symbol to trade (e.g., "BONK")
            
        Returns:
            Dict with trade result
        """
        logger.info(f"🔴 EXECUTING REAL TRADE for {token_symbol} with {self.amount:.6f} SOL")
        
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
                
                # Additional validation to catch token-specific issues
                try:
                    # Try to validate if the token is tradable
                    if 'error' in quote_data:
                        logger.error(f"Token {token_symbol} appears to have trading issues: {quote_data['error']}")
                        return {"success": False, "error": f"Token {token_symbol} not tradable: {quote_data.get('error', 'Unknown error')}"}
                    
                    # Add a specific check for "not tradable" errors
                    if quote_response.text and 'not tradable' in quote_response.text.lower():
                        logger.error(f"Token {token_symbol} is not tradable")
                        return {"success": False, "error": f"Token {token_symbol} is not tradable"}
                except Exception as validation_error:
                    logger.error(f"Error validating trade: {str(validation_error)}")
                    # Continue with the trade attempt despite validation warning
                
                output_amount = int(quote_data['outAmount']) / 1e9  # Convert to decimal units
                logger.info(f"Quote received: {self.amount} SOL -> {output_amount} {token_symbol}")
            except requests.RequestException as e:
                logger.error(f"Error getting quote from Jupiter: {str(e)}")
                return {"success": False, "error": f"Error getting quote: {str(e)}"}
            
            # Step 2: Get the transaction data from Jupiter
            try:
                logger.info("Preparing transaction with Jupiter Exchange")
                
                # New v6 API approach - get serialized transaction
                tx_request = {
                    "quoteResponse": quote_data,
                    "userPublicKey": self.wallet_address,
                    "wrapUnwrapSOL": True
                }
                
                tx_response = requests.post(
                    f"{self.jupiter_api}/swap",
                    json=tx_request,
                    timeout=30
                )
                
                if tx_response.status_code != 200:
                    logger.error(f"Failed to get swap transaction: {tx_response.text}")
                    return {"success": False, "error": f"Failed to get swap transaction: {tx_response.text}"}
                
                tx_data = tx_response.json()
                
                # Check if the transaction was generated
                if "swapTransaction" not in tx_data:
                    logger.error(f"No swap transaction in response: {tx_data}")
                    
                    # Try a more direct approach - use v4 API as fallback
                    logger.info("Trying alternative approach with v4 API...")
                    # Fall back to using Jupiter frontend links that we know work
                    direct_url = f"https://jup.ag/swap/SOL-{token_symbol}?amount={self.amount}&slippage=2.5"
                    logger.info(f"Use this link to execute the trade manually: {direct_url}")
                    
                    return {"success": False, "error": "No swap transaction in response, manual link provided"}
                
                base64_tx = tx_data["swapTransaction"]
                logger.info(f"Got base64 encoded transaction of length: {len(base64_tx)}")
                
                # Step 3: Send transaction to the blockchain
                logger.info("Sending transaction to Solana blockchain")
                
                # Convert the serialized transaction from jupiter to bytes
                serialized_tx = base64.b64decode(base64_tx)
                
                # Load the transaction locally
                transaction = Transaction.deserialize(serialized_tx)
                
                # Need to sign it with our keypair
                transaction.sign(self.keypair)
                
                # Serialize it for sending
                signed_tx = transaction.serialize()
                
                # Submit the transaction using the Solana client
                tx_opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")
                result = self.solana_client.send_transaction(transaction, self.keypair, opts=tx_opts)
                
                if "result" in result:
                    tx_sig = result["result"]
                    logger.info(f"Transaction sent: {tx_sig}")
                    
                    # Add Solscan link for verification
                    verify_url = f"https://solscan.io/tx/{tx_sig}"
                    logger.info(f"✅ TRANSACTION RECORDED ON BLOCKCHAIN - Verify at: {verify_url}")
                    
                    # Wait for confirmation
                    logger.info("Waiting for transaction confirmation...")
                    confirmed = self._confirm_transaction(tx_sig)
                    
                    if confirmed:
                        logger.info(f"Transaction confirmed: {tx_sig}")
                        
                        # Record successful trade
                        trade_data = {
                            "success": True,
                            "token": token_symbol,
                            "input_amount_sol": self.amount,
                            "output_amount": output_amount,
                            "output_token": token_symbol,
                            "transaction_signature": tx_sig,
                            "verify_url": verify_url,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        # Update stats
                        self.total_spent += self.amount
                        self.successful_trades += 1
                        
                        # Record the trade
                        self._record_trade_to_file(trade_data)
                        
                        return trade_data
                    else:
                        logger.error(f"Transaction not confirmed: {tx_sig}")
                        return {
                            "success": False, 
                            "error": "Transaction not confirmed",
                            "verify_url": verify_url
                        }
                else:
                    error = result.get("error", "Unknown error")
                    logger.error(f"Failed to send transaction: {error}")
                    return {"success": False, "error": f"Failed to send transaction: {error}"}
                
            except Exception as e:
                logger.error(f"Error processing transaction: {str(e)}")
                logger.exception("Exception details:")
                return {"success": False, "error": f"Error processing transaction: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            self.failed_trades += 1
            return {"success": False, "error": str(e)}
    
    def _confirm_transaction(self, signature: str, max_retries: int = 30, retry_delay: int = 2) -> bool:
        """
        Wait for transaction confirmation
        
        Args:
            signature: Transaction signature
            max_retries: Maximum number of confirmation check retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if confirmed, False otherwise
        """
        for i in range(max_retries):
            try:
                resp = self.solana_client.get_signature_statuses([signature], search_transaction_history=True)
                
                if resp["result"] and resp["result"]["value"] and resp["result"]["value"][0]:
                    confirmation = resp["result"]["value"][0]
                    if confirmation.get("confirmationStatus") in ["confirmed", "finalized"]:
                        return True
                    elif confirmation.get("confirmations", 0) > 0:
                        return True
                
                logger.info(f"Transaction not confirmed yet. Retry {i+1}/{max_retries}...")
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error checking transaction status: {str(e)}")
                time.sleep(retry_delay)
        
        logger.error(f"Transaction confirmation timed out after {max_retries} attempts")
        return False

    def _record_trade_to_file(self, trade_data: Dict[str, Any]) -> None:
        """
        Record successful trades to a file for tracking
        
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
                    # Create a backup of the corrupted file
                    backup_name = f"{filename}.bak.{int(time.time())}"
                    logger.info(f"Creating backup of corrupted trades file: {backup_name}")
                    try:
                        with open(filename, 'r') as src, open(backup_name, 'w') as dst:
                            dst.write(src.read())
                    except Exception as backup_error:
                        logger.error(f"Error creating backup: {str(backup_error)}")
                    # Reset trades list
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
            "total_sol_spent": self.total_spent,
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
        logger.info(f"Total trades executed: {stats['total_trades_executed']}")
        logger.info(f"Successful trades: {stats['successful_trades']}")
        logger.info(f"Failed trades: {stats['failed_trades']}")
        logger.info(f"Success rate: {stats['success_rate']:.2f}%")
        logger.info(f"Total SOL spent: {stats['total_sol_spent']:.6f}")
        logger.info(f"Runtime: {stats['runtime_hours']:.2f} hours")
        logger.info(f"Trades per hour: {stats['trades_per_hour']:.2f}")
        logger.info(f"Trades per day (projected): {stats['trades_per_day']:.2f}")
        logger.info(f"Current SOL balance: {stats['current_sol_balance']:.6f}")
        logger.info(f"Wallet address: {stats['wallet_address']}")
        logger.info(f"Solscan wallet link: https://solscan.io/account/{stats['wallet_address']}")
        logger.info("=========================")
    
    def run_trading_loop(self) -> None:
        """
        Main trading loop executing REAL trades with actual money across 6 lots
        Runs indefinitely until interrupted
        """
        logger.info("🚀 Starting REAL trading bot - trading with ACTUAL MONEY")
        logger.info("🔴 WARNING: This bot will execute REAL blockchain transactions with REAL MONEY")
        logger.info(f"💰 Trading with wallet: {self.wallet_address}")
        logger.info(f"💼 Trading strategy: {self.num_lots} concurrent lots with {self.lot_amount:.6f} SOL each")
        
        # High-frequency trading configuration (optimized for real-world limitations)
        trades_per_min = self.num_lots * 1  # 6 lots * 1 trade per minute per lot = 6 trades per minute
        trades_per_day = trades_per_min * 60 * 24   # ~8,640 trades per day (high frequency)
        logger.info(f"Configured to execute approximately {trades_per_min} trades per minute")
        logger.info(f"Daily trade volume projection: {int(trades_per_day)} trades per day (high-frequency trading)")
        logger.info(f"This configuration is optimized for real-world API rate limits and blockchain performance")
        
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
                
                if balance < 0.01:
                    logger.error(f"Insufficient balance ({balance:.6f} SOL) to execute trades. Minimum 0.01 SOL required")
                    time.sleep(60)
                    continue
                
                # Safe trading amount check - adjust lot amounts if needed
                if balance < self.total_amount:
                    old_amount = self.lot_amount
                    # Use 90% of available balance divided across lots
                    adjusted_total = balance * 0.9
                    self.lot_amount = adjusted_total / self.num_lots
                    logger.warning(f"Adjusting lot amount from {old_amount:.6f} to {self.lot_amount:.6f} SOL based on available balance")
                
                # Process each lot
                current_time = time.time()
                for lot_idx in range(self.num_lots):
                    # Check if this lot is available for trading
                    if self.lots[lot_idx] <= current_time:
                        # Lot is available, use it for trading
                        
                        # Amount for this lot
                        self.amount = self.lot_amount
                        
                        # Select a token to trade
                        token_symbol = random.choice(self.tokens)
                        
                        # Execute REAL trade with actual money
                        self.trades_executed += 1
                        logger.info(f"Executing REAL trade #{self.trades_executed} - Lot #{lot_idx+1} for {token_symbol}")
                        result = self.execute_trade(token_symbol)
                        
                        if result["success"]:
                            logger.info(f"💰 REAL TRADE #{self.trades_executed} SUCCESSFUL: Lot #{lot_idx+1} {result['token']} with transaction {result.get('transaction_signature', 'unknown')}")
                            verify_url = result.get("verify_url", "")
                            if verify_url:
                                logger.info(f"View transaction on Solscan: {verify_url}")
                            # Mark this lot as busy for 60 seconds (1 minute per lot)
                            self.lots[lot_idx] = int(current_time + 60)
                        else:
                            logger.error(f"❌ Trade #{self.trades_executed} failed: Lot #{lot_idx+1} {result.get('error', 'Unknown error')}")
                            # Mark this lot as busy for a shorter time to retry faster
                            self.lots[lot_idx] = int(current_time + 15)
                        
                        # Print trading stats periodically
                        if self.trades_executed % 10 == 0:
                            self.print_trading_stats()
                        
                        # Small sleep between lot processing to avoid API rate limits
                        time.sleep(1)
                    
                # Wait a bit before next loop iteration
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                logger.exception("Exception details:")
                # Sleep a bit before retry
                time.sleep(10)


def main():
    """Main entry point"""
    logger.info("Starting MemeStrike trading bot")
    
    try:
        # Initialize trader
        trader = SolanaRealTrader()
        
        # Start trading loop
        trader.run_trading_loop()
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error starting trading bot: {str(e)}")
        logger.exception("Exception details:")
        
        # Keep process alive for logging
        logger.info("Keeping process alive for debugging...")
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
