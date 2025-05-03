//--- Include Trade library
#include <Trade\Trade.mqh>

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
input double InpLots          = 0.01;     // Default Trade Lot Size (if not specified by API)
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
input int    InpATR14Period  = 14;     // ATR 14 Period
input int    InpATR50Period  = 50;     // ATR 50 Period

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

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("API Trading EA initializing...");
   PrintFormat("Data Bar Counts: M1=%d, M5=%d, M30=%d, H1=%d, H4=%d, D1=%d",
               InpM1Bars, InpM5Bars, InpM30Bars, InpH1Bars, InpH4Bars, InpD1Bars);
   PrintFormat("Trading Settings: Default Lots=%.2f, Magic=%d, Slippage=%d points",
               InpLots, InpMagicNumber, InpSlippage);
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
          //PrintFormat("New bar detected on %s. Previous: %s, Current: %s. Processing...", TimeframeToString(Period()), TimeToString(lastBarTime), TimeToString(currentBarTime));

          // --- Send data and get instructions ---
          string instruction = SendDataAndGetInstruction(); // Changed function name for clarity

          // --- Process the received instruction ---
          if(instruction != "" && instruction != "Error") // Ensure we got a valid instruction string
          {
              ProcessInstruction(instruction);
          }
          else if (instruction == "Error")
          {
              Print("Error retrieving or parsing instruction from API.");
          }
          // If instruction is empty, it means no data was sent (e.g., first_tf remained true)

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
//| Function to fetch data, send to API, and return instruction    |
//+------------------------------------------------------------------+
string SendDataAndGetInstruction()
{
   string url = InpApiUrl; // Use input parameter for URL
   string symbol_name = Symbol();
   string instruction = ""; // Initialize instruction string

   // --- Check for existing position for THIS symbol and THIS magic number ---
   bool position_exists = false;
   double open_trade_pnl = 0.0;
   
   // --- Get Account Information ---
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   
   int total_positions = PositionsTotal();
   for(int i = total_positions - 1; i >= 0; i--) // Loop backwards is safer if closing positions
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)) // Select position to get details
      {
         if(PositionGetString(POSITION_SYMBOL) == symbol_name && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         {
            position_exists = true;
            open_trade_pnl = PositionGetDouble(POSITION_PROFIT);
            break; // Found the relevant position for this symbol/magic
         }
      }
   }

   // Only send data if no position exists (trade closed)
   if(position_exists)
   {
      // If position exists, do nothing - skip API call completely
      //Print("Position still open - not sending any data to API");
      return ""; // Return empty string to indicate no action needed
   }
   else
   {
      // No position exists (trade closed or no trade yet) - send full data
      Print("No position exists - sending full market data");
      
      // Prepare timeframes to send
      ENUM_TIMEFRAMES timeframes_to_send[] = {
         PERIOD_M1,
         PERIOD_M5,
         PERIOD_M30,
         PERIOD_H1, 
         PERIOD_H4,
         PERIOD_D1
      };
      
      // Prepare corresponding bar counts
      int bars_per_timeframe[] = {
         InpM1Bars,
         InpM5Bars,
         InpM30Bars,
         InpH1Bars,
         InpH4Bars,
         InpD1Bars
      };

      // Start building the JSON payload
      string json_payload = "{";
      json_payload += "\"symbol\": \"" + symbol_name + "\","; 
      json_payload += "\"account_balance\": " + DoubleToString(account_balance, 2) + ",";
      json_payload += "\"open_trade_pnl\": " + DoubleToString(open_trade_pnl, 2) + ",";
      json_payload += "\"position_exists\": false,";
      json_payload += "\"data\": {";

      // Keep track of first timeframe to handle commas properly
      bool first_tf = true;
      
      // Process each timeframe
      for (int i = 0; i < ArraySize(timeframes_to_send); i++)
      {
         ENUM_TIMEFRAMES current_tf = timeframes_to_send[i];
         string tf_string = TimeframeToString(current_tf);
         int bars_to_fetch_for_this_tf = bars_per_timeframe[i];

         // --- Fetch bar data ---
         MqlRates rates[];
         ArraySetAsSeries(rates, true); // Most recent data first
         int copied_count = CopyRates(symbol_name, current_tf, 0, bars_to_fetch_for_this_tf, rates);
         
         if (copied_count > 0)
         {
            //PrintFormat("Fetched %d bars for %s (requested %d)", copied_count, tf_string, bars_to_fetch_for_this_tf);
            if (!first_tf) { json_payload += ","; }
            first_tf = false;
            json_payload += "\"" + tf_string + "\": [";
            
            // --- Fetch technical indicator data ---
            // Initialize indicator buffers
            double ma20_buffer[];
            double ma50_buffer[];
            double ma100_buffer[];
            double bb_upper_buffer[];
            double bb_middle_buffer[];
            double bb_lower_buffer[];
            double rsi_buffer[];
            double stoch_k_buffer[];
            double stoch_d_buffer[];
            double macd_main_buffer[];
            double macd_signal_buffer[];
            double macd_hist_buffer[];
            double adx_buffer[];
            double plus_di_buffer[];
            double minus_di_buffer[];
            double atr14_buffer[];
            double atr50_buffer[];
            
            // Get indicator handles
            int ma20_handle = iMA(symbol_name, current_tf, InpMA20Period, 0, MODE_SMA, PRICE_CLOSE);
            int ma50_handle = iMA(symbol_name, current_tf, InpMA50Period, 0, MODE_SMA, PRICE_CLOSE);
            int ma100_handle = iMA(symbol_name, current_tf, InpMA100Period, 0, MODE_SMA, PRICE_CLOSE);
            int bb_handle = iBands(symbol_name, current_tf, InpBBPeriod, 0, InpBBDeviation, PRICE_CLOSE);
            int rsi_handle = iRSI(symbol_name, current_tf, InpRSIPeriod, PRICE_CLOSE);
            int stoch_handle = iStochastic(symbol_name, current_tf, InpStochKPeriod, InpStochDPeriod, InpStochSlowing, MODE_SMA, STO_LOWHIGH);
            int macd_handle = iMACD(symbol_name, current_tf, InpMACDFast, InpMACDSlow, InpMACDSignal, PRICE_CLOSE);
            int adx_handle = iADX(symbol_name, current_tf, InpADXPeriod);
            int atr14_handle = iATR(symbol_name, current_tf, InpATR14Period);
            int atr50_handle = iATR(symbol_name, current_tf, InpATR50Period);
            
            // Set up arrays properly
            ArraySetAsSeries(ma20_buffer, true);
            ArraySetAsSeries(ma50_buffer, true);
            ArraySetAsSeries(ma100_buffer, true);
            ArraySetAsSeries(bb_upper_buffer, true);
            ArraySetAsSeries(bb_middle_buffer, true);
            ArraySetAsSeries(bb_lower_buffer, true);
            ArraySetAsSeries(rsi_buffer, true);
            ArraySetAsSeries(stoch_k_buffer, true);
            ArraySetAsSeries(stoch_d_buffer, true);
            ArraySetAsSeries(macd_main_buffer, true);
            ArraySetAsSeries(macd_signal_buffer, true);
            ArraySetAsSeries(macd_hist_buffer, true);
            ArraySetAsSeries(adx_buffer, true);
            ArraySetAsSeries(plus_di_buffer, true);
            ArraySetAsSeries(minus_di_buffer, true);
            ArraySetAsSeries(atr14_buffer, true);
            ArraySetAsSeries(atr50_buffer, true);
            
            // Fetch indicator values
            bool ma20_valid = CopyBuffer(ma20_handle, 0, 0, copied_count, ma20_buffer) == copied_count;
            bool ma50_valid = CopyBuffer(ma50_handle, 0, 0, copied_count, ma50_buffer) == copied_count;
            bool ma100_valid = CopyBuffer(ma100_handle, 0, 0, copied_count, ma100_buffer) == copied_count;
            bool bb_upper_valid = CopyBuffer(bb_handle, 1, 0, copied_count, bb_upper_buffer) == copied_count;
            bool bb_middle_valid = CopyBuffer(bb_handle, 0, 0, copied_count, bb_middle_buffer) == copied_count;
            bool bb_lower_valid = CopyBuffer(bb_handle, 2, 0, copied_count, bb_lower_buffer) == copied_count;
            bool rsi_valid = CopyBuffer(rsi_handle, 0, 0, copied_count, rsi_buffer) == copied_count;
            bool stoch_k_valid = CopyBuffer(stoch_handle, 0, 0, copied_count, stoch_k_buffer) == copied_count;
            bool stoch_d_valid = CopyBuffer(stoch_handle, 1, 0, copied_count, stoch_d_buffer) == copied_count;
            bool macd_main_valid = CopyBuffer(macd_handle, 0, 0, copied_count, macd_main_buffer) == copied_count;
            bool macd_signal_valid = CopyBuffer(macd_handle, 1, 0, copied_count, macd_signal_buffer) == copied_count;
            bool macd_hist_valid = CopyBuffer(macd_handle, 2, 0, copied_count, macd_hist_buffer) == copied_count;
            bool adx_valid = CopyBuffer(adx_handle, 0, 0, copied_count, adx_buffer) == copied_count;
            bool plus_di_valid = CopyBuffer(adx_handle, 1, 0, copied_count, plus_di_buffer) == copied_count;
            bool minus_di_valid = CopyBuffer(adx_handle, 2, 0, copied_count, minus_di_buffer) == copied_count;
            bool atr14_valid = CopyBuffer(atr14_handle, 0, 0, copied_count, atr14_buffer) == copied_count;
            bool atr50_valid = CopyBuffer(atr50_handle, 0, 0, copied_count, atr50_buffer) == copied_count;
            
            // Build JSON array for this timeframe
            for (int j = 0; j < copied_count; j++)
            {
               if (j > 0) { json_payload += ","; }
               
               // Format basic bar data
               json_payload += "{";
               json_payload += "\"time\": " + TimeToString(rates[j].time, TIME_DATE|TIME_MINUTES|TIME_SECONDS) + ",";
               json_payload += "\"open\": " + DoubleToString(rates[j].open, 6) + ",";
               json_payload += "\"high\": " + DoubleToString(rates[j].high, 6) + ",";
               json_payload += "\"low\": " + DoubleToString(rates[j].low, 6) + ",";
               json_payload += "\"close\": " + DoubleToString(rates[j].close, 6) + ",";
               json_payload += "\"volume\": " + IntegerToString(rates[j].tick_volume) + ",";
               json_payload += "\"spread\": " + IntegerToString(rates[j].spread) + ",";
               // Add current bid and ask prices
               json_payload += "\"bid\": " + DoubleToString(SymbolInfoDouble(Symbol(), SYMBOL_BID), 6) + ",";
               json_payload += "\"ask\": " + DoubleToString(SymbolInfoDouble(Symbol(), SYMBOL_ASK), 6);
               
               // Add technical indicators if available
               if (ma20_valid) { json_payload += ",\"ma20\": " + DoubleToString(ma20_buffer[j], 6); }
               if (ma50_valid) { json_payload += ",\"ma50\": " + DoubleToString(ma50_buffer[j], 6); }
               if (ma100_valid) { json_payload += ",\"ma100\": " + DoubleToString(ma100_buffer[j], 6); }
               
               if (bb_upper_valid) { json_payload += ",\"bb_upper\": " + DoubleToString(bb_upper_buffer[j], 6); }
               if (bb_middle_valid) { json_payload += ",\"bb_middle\": " + DoubleToString(bb_middle_buffer[j], 6); }
               if (bb_lower_valid) { json_payload += ",\"bb_lower\": " + DoubleToString(bb_lower_buffer[j], 6); }
               
               if (rsi_valid) { json_payload += ",\"rsi\": " + DoubleToString(rsi_buffer[j], 2); }
               
               if (stoch_k_valid) { json_payload += ",\"stoch_k\": " + DoubleToString(stoch_k_buffer[j], 2); }
               if (stoch_d_valid) { json_payload += ",\"stoch_d\": " + DoubleToString(stoch_d_buffer[j], 2); }
               
               if (macd_main_valid) { json_payload += ",\"macd_main\": " + DoubleToString(macd_main_buffer[j], 6); }
               if (macd_signal_valid) { json_payload += ",\"macd_signal\": " + DoubleToString(macd_signal_buffer[j], 6); }
               if (macd_hist_valid) { json_payload += ",\"macd_hist\": " + DoubleToString(macd_hist_buffer[j], 6); }
               
               if (adx_valid) { json_payload += ",\"adx\": " + DoubleToString(adx_buffer[j], 2); }
               if (plus_di_valid) { json_payload += ",\"plus_di\": " + DoubleToString(plus_di_buffer[j], 2); }
               if (minus_di_valid) { json_payload += ",\"minus_di\": " + DoubleToString(minus_di_buffer[j], 2); }

               // Add ATR values
               if (atr14_valid) { 
                  json_payload += ",\"atr14\": " + DoubleToString(atr14_buffer[j], 6);
                  // Calculate ATR as percentage of price
                  double atr14_percent = 100.0 * atr14_buffer[j] / rates[j].close;
                  json_payload += ",\"atr14_pct\": " + DoubleToString(atr14_percent, 2);
               }
               if (atr50_valid) { 
                  json_payload += ",\"atr50\": " + DoubleToString(atr50_buffer[j], 6);
                  // Calculate ATR as percentage of price
                  double atr50_percent = 100.0 * atr50_buffer[j] / rates[j].close;
                  json_payload += ",\"atr50_pct\": " + DoubleToString(atr50_percent, 2);
               }
               
               json_payload += "}";
            }
            
            json_payload += "]";
            
            // Release indicator handles
            if (ma20_handle != INVALID_HANDLE) IndicatorRelease(ma20_handle);
            if (ma50_handle != INVALID_HANDLE) IndicatorRelease(ma50_handle);
            if (ma100_handle != INVALID_HANDLE) IndicatorRelease(ma100_handle);
            if (bb_handle != INVALID_HANDLE) IndicatorRelease(bb_handle);
            if (rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
            if (stoch_handle != INVALID_HANDLE) IndicatorRelease(stoch_handle);
            if (macd_handle != INVALID_HANDLE) IndicatorRelease(macd_handle);
            if (adx_handle != INVALID_HANDLE) IndicatorRelease(adx_handle);
            if (atr14_handle != INVALID_HANDLE) IndicatorRelease(atr14_handle);
            if (atr50_handle != INVALID_HANDLE) IndicatorRelease(atr50_handle);
         }
         else
         {
            PrintFormat("Warning: Could not fetch bars for %s. Copied: %d (req: %d), Err: %d. Skipping.", tf_string, copied_count, bars_to_fetch_for_this_tf, GetLastError());
         }
      }

      json_payload += "}}"; // Close "data" and main object

      // --- Include last error message if there was one ---
      if (lastError != "")
      {
         json_payload = StringSubstr(json_payload, 0, StringLen(json_payload)-1); // Remove closing brace
         json_payload += ",\"error\": \"" + lastError + "\"}"; // Add error and close
         lastError = ""; // Clear the error after sending
      }
      
      // --- Send the data to the API ---
      string response = Post(url, json_payload);
      //PrintFormat("Response: %s", response);
      
      if (response != "")
      {
         // Extract instruction from API response using standard string functions
         string search_text = "INSTRUCTION: ";
         int position = StringFind(response, search_text);
         
         if (position != -1)
         {
            // Found the instruction text
            position += StringLen(search_text); // Move position to start of instruction
            
            // Extract instruction up to the next quote or end of string
            int end_position = StringFind(response, "\"", position);
            if (end_position == -1) end_position = StringLen(response); // If no quote found, use end of string
            
            instruction = StringSubstr(response, position, end_position - position);
            //Print("Extracted instruction from API: ", instruction);
            return instruction;
         }
         else
         {
            Print("Response does not contain an INSTRUCTION");
            return "";
         }
      }
      else
      {
         Print("No response received from API");
         // Store error for next cycle
         lastError = "No response from API";
         return "Error";
      }
   }

   return instruction;
}

//+------------------------------------------------------------------+
//| Function to process the trading instruction from the API         |
//+------------------------------------------------------------------+
void ProcessInstruction(string instruction)
{
   if(instruction == "")
   {
      Print("No instruction to process");
      return;
   }
   
   // Check for our new format with lots, SL, TP (FORMAT: "BUY:0.01:1.12345:1.12545")
   string components[];
   int num_components = StringSplit(instruction, ':', components);
   
   if(num_components > 1) // Format includes SL and/or TP
   {
      string action = components[0];
      double lots = (num_components > 1) ? StringToDouble(components[1]) : InpLots;
      double sl_price = (num_components > 2) ? StringToDouble(components[2]) : 0.0;
      double tp_price = (num_components > 3) ? StringToDouble(components[3]) : 0.0;
      
      // Execute the appropriate action
      if(action == "BUY")
      {
         ExecuteBuyOrderWithParams(lots, sl_price, tp_price);
      }
      else if(action == "SELL")
      {
         ExecuteSellOrderWithParams(lots, sl_price, tp_price);
      }
      else
      {
         Print("Unknown action in instruction: ", action);
      }
   }
   else // Single-word format (for backward compatibility)
   {
      // Process the instruction based on first character (B=Buy, S=Sell, H=Hold)
      if(instruction == "BUY")
      {
         ExecuteBuyOrder();
      }
      else if(instruction == "SELL") 
      {
         ExecuteSellOrder();
      }
      else if(instruction == "HOLD")
      {
         Print("Instruction: HOLD - No trade action taken");
      }
      // For backward compatibility, also check for single letters
      else if(StringGetCharacter(instruction, 0) == 'B')
      {
         ExecuteBuyOrder();
      }
      else if(StringGetCharacter(instruction, 0) == 'S')
      {
         ExecuteSellOrder();
      }
      else if(StringGetCharacter(instruction, 0) == 'H')
      {
         Print("Instruction: Hold - No trade action taken");
      }
      else
      {
         Print("Unknown instruction: ", instruction);
      }
   }
}

//+------------------------------------------------------------------+
//| Function to execute a BUY order with specific parameters         |
//+------------------------------------------------------------------+
void ExecuteBuyOrderWithParams(double lot_size, double sl_price, double tp_price)
{
   double price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   
   // Execute the trade
   Print("Executing BUY order: Lots=", lot_size, ", SL=", sl_price, ", TP=", tp_price);
   if(!trade.Buy(lot_size, Symbol(), 0, sl_price, tp_price, "API Signal"))
   {
      Print("Error executing BUY order: ", GetLastError());
      lastError = "Error executing BUY: " + IntegerToString(GetLastError());
   }
   else
   {
      Print("BUY order executed successfully");
   }
}

//+------------------------------------------------------------------+
//| Function to execute a SELL order with specific parameters        |
//+------------------------------------------------------------------+
void ExecuteSellOrderWithParams(double lot_size, double sl_price, double tp_price)
{
   double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   
   // Execute the trade
   Print("Executing SELL order: Lots=", lot_size, ", SL=", sl_price, ", TP=", tp_price);
   if(!trade.Sell(lot_size, Symbol(), 0, sl_price, tp_price, "API Signal"))
   {
      Print("Error executing SELL order: ", GetLastError());
      lastError = "Error executing SELL: " + IntegerToString(GetLastError());
   }
   else
   {
      Print("SELL order executed successfully");
   }
}

//+------------------------------------------------------------------+
//| Function to execute a BUY order with default parameters          |
//+------------------------------------------------------------------+
void ExecuteBuyOrder()
{
   double price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   double lot_size = InpLots; // Default lot size from settings
   
   // Calculate default SL and TP (10 and 20 points)
   double sl_points = 100;
   double tp_points = 200;
   
   // Convert to price values
   double points_value = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double sl_price = price - (sl_points * points_value);
   double tp_price = price + (tp_points * points_value);
   
   // Execute the trade
   Print("Executing BUY order: Lots=", lot_size, ", SL=", sl_points, " points, TP=", tp_points, " points");
   if(!trade.Buy(lot_size, Symbol(), 0, sl_price, tp_price, "API Signal"))
   {
      Print("Error executing BUY order: ", GetLastError());
      lastError = "Error executing BUY: " + IntegerToString(GetLastError());
   }
   else
   {
      Print("BUY order executed successfully");
   }
}

//+------------------------------------------------------------------+
//| Function to execute a SELL order with default parameters         |
//+------------------------------------------------------------------+
void ExecuteSellOrder()
{
   double price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double lot_size = InpLots; // Default lot size from settings
   
   // Calculate default SL and TP (10 and 20 points)
   double sl_points = 100;
   double tp_points = 200;
   
   // Convert to price values
   double points_value = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double sl_price = price + (sl_points * points_value);
   double tp_price = price - (tp_points * points_value);
   
   // Execute the trade
   Print("Executing SELL order: Lots=", lot_size, ", SL=", sl_points, " points, TP=", tp_points, " points");
   if(!trade.Sell(lot_size, Symbol(), 0, sl_price, tp_price, "API Signal"))
   {
      Print("Error executing SELL order: ", GetLastError());
      lastError = "Error executing SELL: " + IntegerToString(GetLastError());
   }
   else
   {
      Print("SELL order executed successfully");
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
      // Add other timeframes if needed
      default:         return EnumToString(period); // Fallback for unhandled enums
   }
}
//+-------------------------------------------------------------------+
