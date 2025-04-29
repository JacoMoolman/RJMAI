//--- START OF FILE api_trading_ea.mq5 ---

//--- Include Trade library
#include <Trade\Trade.mqh>
#include <stdlib.mqh> // Required for StringToDouble, StringToInteger

//--- Input parameters for bar counts per timeframe
input group "Data Fetching Settings"
input int InpM1Bars  = 500; // Number of M1 bars to fetch
input int InpM5Bars  = 300; // Number of M5 bars to fetch
input int InpM30Bars = 200; // Number of M30 bars to fetch
input int InpH1Bars  = 100;  // Number of H1 bars to fetch
input int InpH4Bars  = 50;  // Number of H4 bars to fetch
input int InpD1Bars  = 20;  // Number of D1 bars to fetch

//--- Input parameters for Trading
input group "Trading Settings"
// input double InpLots          = 0.01;     // Lot Size - REMOVED, will come from API
input ulong  InpMagicNumber   = 12345;    // Magic Number for trades
input uint   InpSlippage      = 10;       // Slippage in points
input string InpApiUrl        = "http://localhost:5000/test"; // API Endpoint URL

//--- Technical Indicator Parameters ---
input group "Technical Indicators"
input int    InpMA20Period   = 20;     // MA 20 Period
input int    InpMA50Period   = 50;     // MA 50 Period
input int    InpMA100Period  = 100;    // MA 100 Period
input int    InpRSIPeriod    = 14;     // RSI Period
input int    InpBBPeriod     = 20;     // Bollinger Bands Period
input int    InpBBDeviation  = 2;      // Bollinger Bands Deviation
input int    InpMACDFast     = 12;     // MACD Fast Period
input int    InpMACDSlow     = 26;     // MACD Slow Period
input int    InpMACDSignal   = 9;      // MACD Signal Period
input int    InpStochKPeriod = 5;      // Stochastic K Period
input int    InpStochDPeriod = 3;      // Stochastic D Period
input int    InpStochSlowing = 3;      // Stochastic Slowing
input int    InpADXPeriod    = 14;     // ADX Period

//+------------------------------------------------------------------+
//| DLL Imports                                                      |
//+------------------------------------------------------------------+
#import "restmql_x64.dll"
   int CPing(string &str);
   string CPing2();
   string Get(string url);
   string Post(string url, string data);
#import

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade trade; // Trade execution object
static datetime lastBarTime = 0; // Tracks the last bar time for new bar detection
static string lastError = ""; // Stores the last error for sending in the next data cycle

// Structure to hold parsed results
struct ApiResult
{
   string instruction;
   double sl_pips;
   double tp_pips;
   double lots;
   string message;
   bool   success; // Was parsing successful?
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("API Trading EA initializing...");
   PrintFormat("Data Bar Counts: M1=%d, M5=%d, M30=%d, H1=%d, H4=%d, D1=%d",
               InpM1Bars, InpM5Bars, InpM30Bars, InpH1Bars, InpH4Bars, InpD1Bars);
   PrintFormat("Trading Settings: Magic=%d, Slippage=%d points", // Removed Lots from here
               InpMagicNumber, InpSlippage);
   Print("API URL: ", InpApiUrl);

   //--- Setup Trade object
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippage);
   trade.SetTypeFillingBySymbol(Symbol()); // Important for execution type (FOK/IOC)

   //--- Initialize lastBarTime
   lastBarTime = iTime(Symbol(), Period(), 0);
   Print("Initial bar time set to: ", TimeToString(lastBarTime));

   Print("API Trading EA initialized!");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("API Trading EA deinitialized! Reason: ", reason);
   lastBarTime = 0; // Reset static variable
   lastError = ""; // Reset static variable
}

//+------------------------------------------------------------------+
//| Expert tick function - Triggers on new bar                       |
//+------------------------------------------------------------------+
void OnTick()
{
   // Get the time of the current bar (index 0) on the chart's timeframe
   datetime currentBarTime = iTime(Symbol(), Period(), 0);

   // Check if a new bar has started
   if (currentBarTime != lastBarTime)
   {
      // --- Check if connected before proceeding ---
      if(TerminalInfoInteger(TERMINAL_CONNECTED) && !MQLInfoInteger(MQL_DEBUG)) // Don't trade if not connected or debugging
      {
          // PrintFormat("New bar detected on %s. Processing...", TimeframeToString(Period())); // Reduced verbosity

          // --- Send data and get instructions ---
          ApiResult result = SendDataAndGetInstruction(); // Get the parsed result struct

          // --- Process the received instruction ---
          if(result.success) // Check if parsing was successful
          {
              ProcessInstruction(result.instruction, result.sl_pips, result.tp_pips, result.lots);
              // Print API message if any
              if(result.message != "") Print("API Message: ", result.message);
          }
          else
          {
              Print("Error retrieving or parsing instruction from API. Message: ", result.message);
          }
      }
      else
      {
          if(!TerminalInfoInteger(TERMINAL_CONNECTED)) Print("Terminal not connected. Skipping cycle.");
          if(MQLInfoInteger(MQL_DEBUG)) Print("Debugger attached. Skipping cycle.");
      }
      // ---------------------------------------------

      // Update the time of the last known bar
      lastBarTime = currentBarTime;
   }
}


//+------------------------------------------------------------------+
//| Helper Function to Extract JSON Value (Basic)                    |
//+------------------------------------------------------------------+
string GetJsonValue(const string& json, const string& key)
{
   string search_key = "\"" + key + "\":";
   int start_pos = StringFind(json, search_key);
   if(start_pos < 0) return ""; // Key not found

   start_pos += StringLen(search_key); // Move past the key and colon

   // Find the start of the value (skip whitespace)
   while(start_pos < StringLen(json) && (StringGetCharacter(json, start_pos) == ' ' || StringGetCharacter(json, start_pos) == '\t'))
   {
      start_pos++;
   }
   if(start_pos >= StringLen(json)) return ""; // Reached end unexpectedly

   char first_char = StringGetCharacter(json, start_pos);
   int end_pos = -1;

   if(first_char == '"') // String value
   {
      start_pos++; // Move past the opening quote
      end_pos = StringFind(json, "\"", start_pos);
      if(end_pos < 0) return ""; // Closing quote not found
   }
   else // Numeric or boolean value (potentially)
   {
      // Find the next comma or closing brace
      int comma_pos = StringFind(json, ",", start_pos);
      int brace_pos = StringFind(json, "}", start_pos);

      if(comma_pos >= 0 && brace_pos >= 0)
         end_pos = MathMin(comma_pos, brace_pos);
      else if(comma_pos >= 0)
         end_pos = comma_pos;
      else if(brace_pos >= 0)
         end_pos = brace_pos;
      else
         end_pos = StringLen(json); // If it's the last element

      if(end_pos <= start_pos) return ""; // Error case
      // Trim trailing whitespace before returning substring
      int temp_end = end_pos -1;
       while(temp_end >= start_pos && (StringGetCharacter(json, temp_end) == ' ' || StringGetCharacter(json, temp_end) == '\t' || StringGetCharacter(json, temp_end) == '\n' || StringGetCharacter(json, temp_end) == '\r'))
       {
           temp_end--;
       }
       end_pos = temp_end + 1;
   }

    if (end_pos <= start_pos) return ""; // Ensure valid range

   return StringSubstr(json, start_pos, end_pos - start_pos);
}


//+------------------------------------------------------------------+
//| Function to fetch data, send to API, and return parsed result  |
//+------------------------------------------------------------------+
ApiResult SendDataAndGetInstruction()
{
   ApiResult result; // Initialize result struct
   result.instruction = "HOLD"; // Default values
   result.sl_pips = 0.0;
   result.tp_pips = 0.0;
   result.lots = 0.0;
   result.message = "Initialization error";
   result.success = false; // Assume failure initially

   string url = InpApiUrl;
   string symbol_name = Symbol();

   // --- Check for existing position for THIS symbol and THIS magic number ---
   bool position_exists = false;
   double open_trade_pnl = 0.0; // PNL is not sent anymore, but check existence

   int total_positions = PositionsTotal();
   for(int i = total_positions - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == symbol_name && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         {
            position_exists = true;
            // open_trade_pnl = PositionGetDouble(POSITION_PROFIT); // We don't need to send PnL now
            break;
         }
      }
   }

   // Only send data if no position exists
   if(position_exists)
   {
      // Print("Position still open - skipping API call"); // Reduce noise
      result.instruction = "HOLD"; // No action needed
      result.message = "Position open, no API call made.";
      result.success = true; // Technically successful from MQL perspective (no error)
      return result; // Return immediately
   }
   else
   {
      Print("No position exists - preparing and sending market data...");

      // --- Get Account Information ---
      double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);

      // Prepare timeframes and bar counts
      ENUM_TIMEFRAMES timeframes_to_send[] = { PERIOD_M1, PERIOD_M5, PERIOD_M30, PERIOD_H1, PERIOD_H4, PERIOD_D1 };
      int bars_per_timeframe[] = { InpM1Bars, InpM5Bars, InpM30Bars, InpH1Bars, InpH4Bars, InpD1Bars };

      // Start building the JSON payload
      string json_payload = "{";
      json_payload += "\"symbol\": \"" + symbol_name + "\",";
      json_payload += "\"account_balance\": " + DoubleToString(account_balance, 2) + ",";
      // json_payload += "\"open_trade_pnl\": 0.0,"; // Not relevant when position_exists is false
      json_payload += "\"position_exists\": false"; // Explicitly state no position

      // Include last error message if there was one
      if(lastError != "")
      {
         json_payload += ", \"error\": \"" + JsonEscape(lastError) + "\""; // Escape potential JSON special chars
         Print("Including previous error in payload: ", lastError);
         lastError = ""; // Clear the error after adding it
      }

      json_payload += ", \"data\": {"; // Start data block

      bool first_tf = true;
      int total_bars_sent = 0;

      for (int i = 0; i < ArraySize(timeframes_to_send); i++)
      {
         ENUM_TIMEFRAMES current_tf = timeframes_to_send[i];
         string tf_string = TimeframeToString(current_tf);
         int bars_to_fetch_for_this_tf = bars_per_timeframe[i];

         if (bars_to_fetch_for_this_tf <= 0) continue;

         MqlRates rates_array[];
         ArraySetAsSeries(rates_array, true);
         int copied_count = CopyRates(symbol_name, current_tf, 0, bars_to_fetch_for_this_tf, rates_array);

         if (copied_count > 0)
         {
             total_bars_sent += copied_count;
            if (!first_tf) { json_payload += ","; }
            first_tf = false;
            json_payload += "\"" + tf_string + "\": [";

            // --- Get Indicator Handles ---
             int ma20_handle = iMA(symbol_name, current_tf, InpMA20Period, 0, MODE_SMA, PRICE_CLOSE);
             int ma50_handle = iMA(symbol_name, current_tf, InpMA50Period, 0, MODE_SMA, PRICE_CLOSE);
             int ma100_handle = iMA(symbol_name, current_tf, InpMA100Period, 0, MODE_SMA, PRICE_CLOSE);
             int bb_handle = iBands(symbol_name, current_tf, InpBBPeriod, 0, InpBBDeviation, PRICE_CLOSE);
             int rsi_handle = iRSI(symbol_name, current_tf, InpRSIPeriod, PRICE_CLOSE);
             int stoch_handle = iStochastic(symbol_name, current_tf, InpStochKPeriod, InpStochDPeriod, InpStochSlowing, MODE_SMA, STO_LOWHIGH);
             int macd_handle = iMACD(symbol_name, current_tf, InpMACDFast, InpMACDSlow, InpMACDSignal, PRICE_CLOSE);
             int adx_handle = iADX(symbol_name, current_tf, InpADXPeriod);
             int ichimoku_handle = iIchimoku(symbol_name, current_tf, 9, 26, 52); // Standard Ichimoku periods

            // --- Prepare Indicator Buffers ---
            double ma20[], ma50[], ma100[];
            double upper_band[], middle_band[], lower_band[];
            double rsi[];
            double stoch_k[], stoch_d[];
            double macd_main[], macd_signal[];
            double adx[], plus_di[], minus_di[];
            double ichimoku_tenkan[], ichimoku_kijun[], ichimoku_senkou_a[], ichimoku_senkou_b[];

            // Set arrays as series
            ArraySetAsSeries(ma20, true); ArraySetAsSeries(ma50, true); ArraySetAsSeries(ma100, true);
            ArraySetAsSeries(upper_band, true); ArraySetAsSeries(middle_band, true); ArraySetAsSeries(lower_band, true);
            ArraySetAsSeries(rsi, true); ArraySetAsSeries(stoch_k, true); ArraySetAsSeries(stoch_d, true);
            ArraySetAsSeries(macd_main, true); ArraySetAsSeries(macd_signal, true);
            ArraySetAsSeries(adx, true); ArraySetAsSeries(plus_di, true); ArraySetAsSeries(minus_di, true);
            ArraySetAsSeries(ichimoku_tenkan, true); ArraySetAsSeries(ichimoku_kijun, true);
            ArraySetAsSeries(ichimoku_senkou_a, true); ArraySetAsSeries(ichimoku_senkou_b, true);

            // Check for valid handles before copying
            bool handles_ok = ma20_handle != INVALID_HANDLE && ma50_handle != INVALID_HANDLE && ma100_handle != INVALID_HANDLE &&
                              bb_handle != INVALID_HANDLE && rsi_handle != INVALID_HANDLE && stoch_handle != INVALID_HANDLE &&
                              macd_handle != INVALID_HANDLE && adx_handle != INVALID_HANDLE && ichimoku_handle != INVALID_HANDLE;

             // --- Copy Indicator Data ---
             if (handles_ok) {
                 CopyBuffer(ma20_handle, 0, 0, copied_count, ma20); CopyBuffer(ma50_handle, 0, 0, copied_count, ma50); CopyBuffer(ma100_handle, 0, 0, copied_count, ma100);
                 CopyBuffer(bb_handle, 0, 0, copied_count, upper_band); CopyBuffer(bb_handle, 1, 0, copied_count, middle_band); CopyBuffer(bb_handle, 2, 0, copied_count, lower_band);
                 CopyBuffer(rsi_handle, 0, 0, copied_count, rsi);
                 CopyBuffer(stoch_handle, 0, 0, copied_count, stoch_k); CopyBuffer(stoch_handle, 1, 0, copied_count, stoch_d);
                 CopyBuffer(macd_handle, 0, 0, copied_count, macd_main); CopyBuffer(macd_handle, 1, 0, copied_count, macd_signal);
                 CopyBuffer(adx_handle, 0, 0, copied_count, adx); CopyBuffer(adx_handle, 1, 0, copied_count, plus_di); CopyBuffer(adx_handle, 2, 0, copied_count, minus_di);
                 CopyBuffer(ichimoku_handle, 0, 0, copied_count, ichimoku_tenkan); CopyBuffer(ichimoku_handle, 1, 0, copied_count, ichimoku_kijun);
                 CopyBuffer(ichimoku_handle, 2, 0, copied_count, ichimoku_senkou_a); CopyBuffer(ichimoku_handle, 3, 0, copied_count, ichimoku_senkou_b);
             } else {
                 PrintFormat("Warning: Invalid indicator handle(s) for %s. Indicators might be missing.", tf_string);
             }

            // Release handles
            if (ma20_handle != INVALID_HANDLE) IndicatorRelease(ma20_handle); if (ma50_handle != INVALID_HANDLE) IndicatorRelease(ma50_handle); if (ma100_handle != INVALID_HANDLE) IndicatorRelease(ma100_handle);
            if (bb_handle != INVALID_HANDLE) IndicatorRelease(bb_handle); if (rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle); if (stoch_handle != INVALID_HANDLE) IndicatorRelease(stoch_handle);
            if (macd_handle != INVALID_HANDLE) IndicatorRelease(macd_handle); if (adx_handle != INVALID_HANDLE) IndicatorRelease(adx_handle); if (ichimoku_handle != INVALID_HANDLE) IndicatorRelease(ichimoku_handle);


            // Build JSON for bars
            for (int j = 0; j < copied_count; j++)
            {
                // Use TimeToString for consistent formatting suitable for JSON parsing
                string time_str = TimeToString(rates_array[j].time, "%Y.%m.%d %H:%M:%S");

                json_payload += "{";
                // Ensure time is quoted correctly for JSON standard
                json_payload += "\"time\": \"" + time_str + "\",";
                json_payload += "\"open\": "   + DoubleToString(rates_array[j].open, _Digits) + ",";
                json_payload += "\"high\": "   + DoubleToString(rates_array[j].high, _Digits) + ",";
                json_payload += "\"low\": "    + DoubleToString(rates_array[j].low, _Digits) + ",";
                json_payload += "\"close\": "  + DoubleToString(rates_array[j].close, _Digits) + ",";
                json_payload += "\"volume\": " + (string)rates_array[j].tick_volume + ",";
                json_payload += "\"spread\": " + (string)rates_array[j].spread;

                // Add technical indicators if data is valid and handles were okay
                bool valid_inds = handles_ok && j < ArraySize(ma20) && j < ArraySize(ma50) && j < ArraySize(ma100) && // Add other arrays
                                  j < ArraySize(rsi) && j < ArraySize(stoch_k) && j < ArraySize(stoch_d) &&
                                  j < ArraySize(upper_band) && j < ArraySize(middle_band) && j < ArraySize(lower_band) &&
                                  j < ArraySize(macd_main) && j < ArraySize(macd_signal) &&
                                  j < ArraySize(adx) && j < ArraySize(plus_di) && j < ArraySize(minus_di) &&
                                  j < ArraySize(ichimoku_tenkan) && j < ArraySize(ichimoku_kijun) && j < ArraySize(ichimoku_senkou_a) && j < ArraySize(ichimoku_senkou_b);

                 if(valid_inds) {
                    json_payload += ",\"ma20\": " + DoubleToString(ma20[j], _Digits) + ",\"ma50\": " + DoubleToString(ma50[j], _Digits) + ",\"ma100\": " + DoubleToString(ma100[j], _Digits);
                    json_payload += ",\"bb_upper\": " + DoubleToString(upper_band[j], _Digits) + ",\"bb_middle\": " + DoubleToString(middle_band[j], _Digits) + ",\"bb_lower\": " + DoubleToString(lower_band[j], _Digits);
                    json_payload += ",\"rsi\": " + DoubleToString(rsi[j], 2);
                    json_payload += ",\"stoch_k\": " + DoubleToString(stoch_k[j], 2) + ",\"stoch_d\": " + DoubleToString(stoch_d[j], 2);
                    json_payload += ",\"macd_main\": " + DoubleToString(macd_main[j], _Digits) + ",\"macd_signal\": " + DoubleToString(macd_signal[j], _Digits) + ",\"macd_hist\": " + DoubleToString(macd_main[j] - macd_signal[j], _Digits);
                    json_payload += ",\"adx\": " + DoubleToString(adx[j], 2) + ",\"plus_di\": " + DoubleToString(plus_di[j], 2) + ",\"minus_di\": " + DoubleToString(minus_di[j], 2);
                    json_payload += ",\"ichimoku_tenkan\": " + DoubleToString(ichimoku_tenkan[j], _Digits) + ",\"ichimoku_kijun\": " + DoubleToString(ichimoku_kijun[j], _Digits) + ",\"ichimoku_senkou_a\": " + DoubleToString(ichimoku_senkou_a[j], _Digits) + ",\"ichimoku_senkou_b\": " + DoubleToString(ichimoku_senkou_b[j], _Digits);
                 } else {
                    // Optionally add null values if needed by Python, or just omit them
                    // json_payload += ",\"ma20\": null"; // etc.
                 }

                json_payload += "}";
                if (j < copied_count - 1) { json_payload += ","; }
            }
            json_payload += "]";
         }
         else
         {
            PrintFormat("Warning: Could not fetch bars for %s. Copied: %d (req: %d), Err: %d. Skipping.", tf_string, copied_count, bars_to_fetch_for_this_tf, GetLastError());
         }

         Sleep(10); // Small sleep between timeframes
      }

      json_payload += "}}"; // Close "data" and main object

      // Check if any data was actually added
      if(total_bars_sent == 0)
      {
          Print("Error: No bar data could be fetched for any timeframe. Aborting API call.");
          result.message = "Failed to fetch any market data.";
          result.success = false;
          return result;
      }

      // --- Send Data and Get Response ---
      Print("Sending data to API (" + IntegerToString(StringLen(json_payload)) + " bytes)...");
      string response = "";
      // Add retry mechanism for Post request? For now, simple call.
      response = Post(url, json_payload);

      // --- Parse the Response ---
      if(response != "")
      {
         Print("Response received: ", response);

         // Extract values using helper function
         string instruction_str = GetJsonValue(response, "instruction");
         string sl_pips_str     = GetJsonValue(response, "sl_pips");
         string tp_pips_str     = GetJsonValue(response, "tp_pips");
         string lots_str        = GetJsonValue(response, "lots");
         string message_str     = GetJsonValue(response, "message");

         // Validate extracted values
         if(instruction_str != "" && sl_pips_str != "" && tp_pips_str != "" && lots_str != "")
         {
             result.instruction = instruction_str;
             // Convert strings to numbers, handle potential errors
             result.sl_pips = StringToDouble(sl_pips_str);
             result.tp_pips = StringToDouble(tp_pips_str);
             result.lots = StringToDouble(lots_str);
             result.message = message_str; // Optional message from API

             // Basic validation
             if (result.instruction == "BUY" || result.instruction == "SELL") {
                 if (result.sl_pips <= 0 || result.tp_pips <= 0 || result.lots <= 0) {
                     Print("Error: Invalid SL/TP pips or Lots received from API. SL=", DoubleToString(result.sl_pips), ", TP=", DoubleToString(result.tp_pips), ", Lots=", DoubleToString(result.lots));
                     lastError = "Invalid SL/TP/Lots received"; // Store error
                     result.success = false;
                     result.message = "Received invalid SL/TP/Lots values.";
                 } else {
                     result.success = true; // Valid trade parameters received
                 }
             } else if (result.instruction == "HOLD") {
                 result.success = true; // HOLD is a valid scenario
             } else {
                 Print("Error: Unknown instruction received from API: ", result.instruction);
                 lastError = "Unknown instruction: " + result.instruction; // Store error
                 result.success = false;
                 result.message = "Received unknown instruction.";
             }
         }
         else
         {
            Print("Error: Failed to parse required fields from API response.");
            lastError = "Failed to parse API response"; // Store error
            result.message = "Could not parse JSON response fields.";
            result.success = false;
         }
      }
      else
      {
         Print("Error: No response received from API");
         lastError = "No response from API"; // Store error
         result.message = "No response received from API.";
         result.success = false;
      }
   }

   return result;
}

//+------------------------------------------------------------------+
//| Function to process the trading instruction from the API         |
//+------------------------------------------------------------------+
void ProcessInstruction(string instruction, double sl_pips, double tp_pips, double lots)
{
   if(instruction == "BUY")
   {
      ExecuteBuyOrder(sl_pips, tp_pips, lots);
   }
   else if(instruction == "SELL")
   {
      ExecuteSellOrder(sl_pips, tp_pips, lots);
   }
   else if(instruction == "HOLD")
   {
      Print("Instruction: HOLD - No trade action taken");
   }
   else
   {
      Print("Unknown instruction received in ProcessInstruction: ", instruction);
      // This case should ideally be caught during parsing, but good to have a fallback log
   }
}

//+------------------------------------------------------------------+
//| Function to execute a BUY order using API parameters           |
//+------------------------------------------------------------------+
void ExecuteBuyOrder(double sl_pips, double tp_pips, double lots)
{
   // Basic validation (should have been done during parsing, but double-check)
   if (sl_pips <= 0 || tp_pips <= 0 || lots <= 0) {
       Print("BUY Order Canceled: Invalid SL/TP/Lots. SL Pips=", sl_pips, ", TP Pips=", tp_pips, ", Lots=", lots);
       return;
   }

   double price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);

   // Ensure price and point are valid
   if(price <= 0 || point <= 0) {
       Print("BUY Order Canceled: Invalid symbol price or point value.");
       lastError = "Invalid symbol info for trade execution";
       return;
   }

   // Calculate SL and TP prices
   double sl_price = price - (sl_pips * point);
   double tp_price = price + (tp_pips * point);

   // Normalize prices (adjust to correct decimal places)
   sl_price = NormalizeDouble(sl_price, digits);
   tp_price = NormalizeDouble(tp_price, digits);

   // Execute the trade
   PrintFormat("Executing BUY order: Lots=%.2f, SL=%.5f (%.1f pips), TP=%.5f (%.1f pips) at Ask=%.5f",
               lots, sl_price, sl_pips, tp_price, tp_pips, price);

   if(!trade.Buy(lots, Symbol(), price, sl_price, tp_price, "API Signal")) // Use current price in Buy call
   {
      Print("Error executing BUY order: ", trade.ResultComment(), " (Code: ", trade.ResultRetcode(), ")");
      lastError = "Error executing BUY (" + IntegerToString(trade.ResultRetcode()) + "): " + trade.ResultComment();
   }
   else
   {
      Print("BUY order executed successfully. Result: ", trade.ResultComment());
   }
}

//+------------------------------------------------------------------+
//| Function to execute a SELL order using API parameters          |
//+------------------------------------------------------------------+
void ExecuteSellOrder(double sl_pips, double tp_pips, double lots)
{
    // Basic validation
   if (sl_pips <= 0 || tp_pips <= 0 || lots <= 0) {
       Print("SELL Order Canceled: Invalid SL/TP/Lots. SL Pips=", sl_pips, ", TP Pips=", tp_pips, ", Lots=", lots);
       return;
   }

   double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);

    // Ensure price and point are valid
   if(price <= 0 || point <= 0) {
       Print("SELL Order Canceled: Invalid symbol price or point value.");
       lastError = "Invalid symbol info for trade execution";
       return;
   }

   // Calculate SL and TP prices
   double sl_price = price + (sl_pips * point);
   double tp_price = price - (tp_pips * point);

   // Normalize prices
   sl_price = NormalizeDouble(sl_price, digits);
   tp_price = NormalizeDouble(tp_price, digits);

   // Execute the trade
   PrintFormat("Executing SELL order: Lots=%.2f, SL=%.5f (%.1f pips), TP=%.5f (%.1f pips) at Bid=%.5f",
               lots, sl_price, sl_pips, tp_price, tp_pips, price);

   if(!trade.Sell(lots, Symbol(), price, sl_price, tp_price, "API Signal")) // Use current price in Sell call
   {
      Print("Error executing SELL order: ", trade.ResultComment(), " (Code: ", trade.ResultRetcode(), ")");
      lastError = "Error executing SELL (" + IntegerToString(trade.ResultRetcode()) + "): " + trade.ResultComment();
   }
   else
   {
      Print("SELL order executed successfully. Result: ", trade.ResultComment());
   }
}

//+------------------------------------------------------------------+
//| Helper function to convert ENUM_TIMEFRAMES to string             |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES period)
{
   switch (period)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      default:         return EnumToString(period);
   }
}

//+------------------------------------------------------------------+
//| Helper function to escape JSON special characters                |
//+------------------------------------------------------------------+
string JsonEscape(string str)
{
   StringReplace(str, "\\", "\\\\"); // Escape backslashes first
   StringReplace(str, "\"", "\\\""); // Escape double quotes
   StringReplace(str, "\r", "\\r");  // Escape carriage return
   StringReplace(str, "\n", "\\n");  // Escape newline
   StringReplace(str, "\t", "\\t");  // Escape tab
   // Add other escapes if necessary (e.g., forward slash, control characters)
   return str;
}
//+------------------------------------------------------------------+

// --- END OF FILE api_trading_ea.mq5 ---