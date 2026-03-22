// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TradeLogger {
    struct Trade {
        string ticker;
        string action;
        uint256 amountScaled; // Scaled by 1e18
        uint256 priceScaled;  // Scaled by 1e18
        uint256 timestamp;
    }

    Trade[] public tradeHistory;

    event TradeLogged(
        string indexed ticker,
        string action,
        uint256 amountScaled,
        uint256 priceScaled,
        uint256 timestamp
    );

    function logTrade(
        string memory _ticker,
        string memory _action,
        uint256 _amountScaled,
        uint256 _priceScaled
    ) public {
        Trade memory newTrade = Trade({
            ticker: _ticker,
            action: _action,
            amountScaled: _amountScaled,
            priceScaled: _priceScaled,
            timestamp: block.timestamp
        });
        
        tradeHistory.push(newTrade);
        emit TradeLogged(_ticker, _action, _amountScaled, _priceScaled, block.timestamp);
    }

    function getTradeCount() public view returns (uint256) {
        return tradeHistory.length;
    }
}
