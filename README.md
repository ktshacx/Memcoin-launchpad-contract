# SeiPAD - A Decentralized Token Creation and Trading Platform

SeiPAD is a decentralized platform built to facilitate the creation, funding, and trading of tokens using a **bonding curve** mechanism. This platform allows users to create new tokens, fund them through a decentralized model, and trade them once they reach their funding goal. It also incorporates a fee model for token creation and transactions, and integrates with **Uniswap V2** for liquidity management.

## Features

- **Token Creation**: Users can create new tokens by paying a creation fee.
- **Funding Phase**: Tokens are funded through user contributions. Once the funding goal is reached, liquidity is added to the Uniswap V2 pool.
- **Trading Phase**: After the funding phase, tokens enter the trading phase, and users can buy and sell them on the platform.
- **Bonding Curve**: The price of tokens follows a bonding curve, adjusting based on the supply and demand.
- **Fees**: A fee is charged on transactions, including token buys and sells, and is distributed to the contract owner.

## Smart Contract Functions

### Token Creation

- `createToken`: Allows users to create a new token by paying the required creation fees.
  
### Buying Tokens

- `buy`: Users can contribute Ether (ETH) to purchase tokens during the funding phase. The price of tokens increases based on the bonding curve mechanism.
- Tokens are minted for the buyer, and any excess ETH (after funding the token and paying fees) is returned.

### Selling Tokens

- `sell`: Users can sell their tokens during the funding phase. The contract calculates how much ETH the user will receive, based on the amount of tokens sold, and deducts the corresponding amount from the token's collateral.

### Admin Functions

- `setBondingCurve`: Allows the owner to set the bonding curve contract.
- `setFeePercent`: Allows the owner to set the fee percentage for transactions.
- `claimFee`, `claimETH`, `claimToken`: Functions to allow the contract owner to claim fees, ETH, or tokens from the contract.

### Liquidity Management

- `createLiquidityPool`: Creates a liquidity pool on Uniswap V2 for the token.
- `addLiquidity`: Adds liquidity to the Uniswap pool using ETH and the token.
- `burnLiquidityToken`: Burns liquidity tokens after adding them to Uniswap to maintain a deflationary effect.

---

## Bonding Curve

### What is a Bonding Curve?

A **bonding curve** is a mathematical function used to determine the price of a token based on its supply. It ensures that as more tokens are bought and the supply increases, the price of each subsequent token increases as well. The bonding curve helps create a price discovery mechanism where early investors can buy tokens at a lower price, and later buyers pay a higher price, incentivizing early participation.

In the SeiPAD contract, the bonding curve is implemented using an exponential formula. As users buy tokens during the **funding phase**, the price increases according to the bonding curve formula, which is:

`Price = A * exp(B * x) / (exp(B * x) - 1)`


Where:
- `A` is a constant derived from the funding goal and funding supply.
- `B` is a constant derived from the funding goal and the total funding supply.
- `x` is the total supply of tokens.

The bonding curve works in two main phases:

1. **Funding Phase**: 
   - Tokens are in the funding phase where their price is dynamically set according to the bonding curve. The more tokens are purchased, the higher the price goes.
   - The price increases exponentially, making early contributions cheaper and later contributions more expensive.

2. **Trading Phase**:
   - Once the funding goal is reached, the tokens are minted and liquidity is added to a Uniswap pool. From this point onwards, the tokens can be traded freely, and the bonding curve no longer affects the price.

---

## Contract Structure

### Variables

- `MAX_SUPPLY`: Maximum number of tokens that can ever be minted (1 billion tokens).
- `INITIAL_SUPPLY`: Initial supply of tokens allocated for the creation phase (20% of `MAX_SUPPLY`).
- `FUNDING_SUPPLY`: The total supply of tokens available for funding (80% of `MAX_SUPPLY`).
- `FUNDING_GOAL`: The target funding goal for the token (100 Ether).
- `feePercent`: The percentage fee taken on every transaction (buy/sell).
- `collateral`: Tracks the amount of Ether raised for each token during the funding phase.
- `tokens`: A mapping to track the state of each token (whether it’s in the `NOT_CREATED`, `FUNDING`, or `TRADING` phase).

### Events

- `TokenCreated`: Emitted when a new token is created.
- `TokenLiqudityAdded`: Emitted when liquidity is added to Uniswap for a token.
- `Buy`: Emitted when a user buys tokens.
- `Sell`: Emitted when a user sells tokens.

---

## Dependencies

- **Uniswap V2**: The contract interacts with Uniswap V2 for creating liquidity pools and adding liquidity.
- **Bonding Curve**: The contract relies on a custom `BondingCurve` contract to determine the token price during the funding phase.

---

## How to Use

### Creating a Token

1. Pay the creation fee to create a new token.
2. Provide the name and symbol for the new token.

### Funding a Token

1. Contribute ETH to a token during the **funding phase**.
2. The price of the token increases as more ETH is contributed due to the bonding curve.
3. Once the funding goal is met, the token enters the **trading phase**, and liquidity is added to Uniswap.

### Buying and Selling Tokens

- **Buy** tokens during the funding phase by sending ETH to the contract.
- **Sell** tokens during the funding phase and receive ETH in return.

---

## Conclusion

SeiPAD provides an innovative way to fund and trade tokens using bonding curves and decentralized liquidity management. The integration with Uniswap ensures that tokens can be easily traded once they reach their funding goal, and the fee structure provides incentives for the platform’s maintenance and growth.
