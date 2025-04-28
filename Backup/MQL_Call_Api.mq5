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
input double InpLots          = 0.01;     // Trade Lot Size
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

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("API Trading EA initializing...");
   PrintFormat("Data Bar Counts: M1=%d, M5=%d, M30=%d, H1=%d, H4=%d, D1=%d",
               InpM1Bars, InpM5Bars, InpM30Bars, InpH1Bars, InpH4Bars, InpD1Bars);
   PrintFormat("Trading Settings: Lots=%.2f, Magic=%d, Slippage=%d points",
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
      
      for (int i = 0; i < ArraySize(timeframes_to_send); i++)
      {
         ENUM_TIMEFRAMES current_tf = timeframes_to_send[i];
         string tf_string = TimeframeToString(current_tf);
         int bars_to_fetch_for_this_tf = bars_per_timeframe[i];

         if (bars_to_fetch_for_this_tf <= 0) { PrintFormat("Skipping timeframe %s (bar count <= 0).", tf_string); continue; }

         MqlRates rates_array[];
         ArraySetAsSeries(rates_array, true);
         int copied_count = CopyRates(symbol_name, current_tf, 0, bars_to_fetch_for_this_tf, rates_array);

         if (copied_count > 0)
         {
            //PrintFormat("Fetched %d bars for %s (requested %d)", copied_count, tf_string, bars_to_fetch_for_this_tf);
            if (!first_tf) { json_payload += ","; }
            first_tf = false;
            json_payload += "\"" + tf_string + "\": [";
            
            // Calculate indicators for this timeframe using standard MT5 indicator functions
            double ma20[], ma50[], ma100[];
            double upper_band[], middle_band[], lower_band[];
            double rsi[];
            double stoch_k[], stoch_d[];
            double macd_main[], macd_signal[];
            double adx[], plus_di[], minus_di[];
            double ichimoku_tenkan[], ichimoku_kijun[], ichimoku_senkou_a[], ichimoku_senkou_b[];
            
            // Initialize arrays
            ArraySetAsSeries(ma20, true);
            ArraySetAsSeries(ma50, true);
            ArraySetAsSeries(ma100, true);
            ArraySetAsSeries(upper_band, true);
            ArraySetAsSeries(middle_band, true);
            ArraySetAsSeries(lower_band, true);
            ArraySetAsSeries(rsi, true);
            ArraySetAsSeries(stoch_k, true);
            ArraySetAsSeries(stoch_d, true);
            ArraySetAsSeries(macd_main, true);
            ArraySetAsSeries(macd_signal, true);
            ArraySetAsSeries(adx, true);
            ArraySetAsSeries(plus_di, true);
            ArraySetAsSeries(minus_di, true);
            ArraySetAsSeries(ichimoku_tenkan, true);
            ArraySetAsSeries(ichimoku_kijun, true);
            ArraySetAsSeries(ichimoku_senkou_a, true);
            ArraySetAsSeries(ichimoku_senkou_b, true);
            
            // Calculate the indicators
            int ma20_handle = iMA(symbol_name, current_tf, InpMA20Period, 0, MODE_SMA, PRICE_CLOSE);
            int ma50_handle = iMA(symbol_name, current_tf, InpMA50Period, 0, MODE_SMA, PRICE_CLOSE);
            int ma100_handle = iMA(symbol_name, current_tf, InpMA100Period, 0, MODE_SMA, PRICE_CLOSE);
            int bb_handle = iBands(symbol_name, current_tf, InpBBPeriod, 0, InpBBDeviation, PRICE_CLOSE);
            int rsi_handle = iRSI(symbol_name, current_tf, InpRSIPeriod, PRICE_CLOSE);
            int stoch_handle = iStochastic(symbol_name, current_tf, InpStochKPeriod, InpStochDPeriod, InpStochSlowing, MODE_SMA, STO_LOWHIGH);
            int macd_handle = iMACD(symbol_name, current_tf, InpMACDFast, InpMACDSlow, InpMACDSignal, PRICE_CLOSE);
            int adx_handle = iADX(symbol_name, current_tf, InpADXPeriod);
            int ichimoku_handle = iIchimoku(symbol_name, current_tf, 9, 26, 52); // Standard Ichimoku periods
            
            // Check for valid handles
            bool valid_handles = ma20_handle != INVALID_HANDLE && 
                                ma50_handle != INVALID_HANDLE && 
                                ma100_handle != INVALID_HANDLE && 
                                bb_handle != INVALID_HANDLE && 
                                rsi_handle != INVALID_HANDLE && 
                                stoch_handle != INVALID_HANDLE && 
                                macd_handle != INVALID_HANDLE && 
                                adx_handle != INVALID_HANDLE && 
                                ichimoku_handle != INVALID_HANDLE;
                                
            if (valid_handles) {
               // Copy indicator values
               CopyBuffer(ma20_handle, 0, 0, copied_count, ma20);
               CopyBuffer(ma50_handle, 0, 0, copied_count, ma50);
               CopyBuffer(ma100_handle, 0, 0, copied_count, ma100);
               CopyBuffer(bb_handle, 0, 0, copied_count, upper_band);
               CopyBuffer(bb_handle, 1, 0, copied_count, middle_band);
               CopyBuffer(bb_handle, 2, 0, copied_count, lower_band);
               CopyBuffer(rsi_handle, 0, 0, copied_count, rsi);
               CopyBuffer(stoch_handle, 0, 0, copied_count, stoch_k);
               CopyBuffer(stoch_handle, 1, 0, copied_count, stoch_d);
               CopyBuffer(macd_handle, 0, 0, copied_count, macd_main);
               CopyBuffer(macd_handle, 1, 0, copied_count, macd_signal);
               CopyBuffer(adx_handle, 0, 0, copied_count, adx);
               CopyBuffer(adx_handle, 1, 0, copied_count, plus_di);
               CopyBuffer(adx_handle, 2, 0, copied_count, minus_di);
               CopyBuffer(ichimoku_handle, 0, 0, copied_count, ichimoku_tenkan);
               CopyBuffer(ichimoku_handle, 1, 0, copied_count, ichimoku_kijun);
               CopyBuffer(ichimoku_handle, 2, 0, copied_count, ichimoku_senkou_a);
               CopyBuffer(ichimoku_handle, 3, 0, copied_count, ichimoku_senkou_b);
            }
            
            // Release handles to avoid resource leaks
            if (ma20_handle != INVALID_HANDLE) IndicatorRelease(ma20_handle);
            if (ma50_handle != INVALID_HANDLE) IndicatorRelease(ma50_handle);
            if (ma100_handle != INVALID_HANDLE) IndicatorRelease(ma100_handle);
            if (bb_handle != INVALID_HANDLE) IndicatorRelease(bb_handle);
            if (rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
            if (stoch_handle != INVALID_HANDLE) IndicatorRelease(stoch_handle);
            if (macd_handle != INVALID_HANDLE) IndicatorRelease(macd_handle);
            if (adx_handle != INVALID_HANDLE) IndicatorRelease(adx_handle);
            if (ichimoku_handle != INVALID_HANDLE) IndicatorRelease(ichimoku_handle);
            
            for (int j = 0; j < copied_count; j++)
            {
               // Check if indicator data is valid for this bar
               bool valid_data = valid_handles && 
                              j < ArraySize(ma20) && j < ArraySize(ma50) && j < ArraySize(ma100) &&
                              j < ArraySize(rsi) && j < ArraySize(stoch_k) && j < ArraySize(stoch_d) &&
                              j < ArraySize(upper_band) && j < ArraySize(middle_band) && j < ArraySize(lower_band);
               
               json_payload += "{";
               json_payload += "\"time\": "   + (string)rates_array[j].time + ",";
               json_payload += "\"open\": "   + DoubleToString(rates_array[j].open, _Digits) + ",";
               json_payload += "\"high\": "   + DoubleToString(rates_array[j].high, _Digits) + ",";
               json_payload += "\"low\": "    + DoubleToString(rates_array[j].low, _Digits) + ",";
               json_payload += "\"close\": "  + DoubleToString(rates_array[j].close, _Digits) + ",";
               json_payload += "\"volume\": " + (string)rates_array[j].tick_volume + ",";
               json_payload += "\"spread\": " + (string)rates_array[j].spread;
               
               // Add technical indicators if data is valid
               if(valid_data) {
                  // Moving Averages
                  json_payload += ",\"ma20\": "  + DoubleToString(ma20[j], _Digits);
                  json_payload += ",\"ma50\": "  + DoubleToString(ma50[j], _Digits);
                  json_payload += ",\"ma100\": " + DoubleToString(ma100[j], _Digits);
                  
                  // Bollinger Bands
                  json_payload += ",\"bb_upper\": " + DoubleToString(upper_band[j], _Digits);
                  json_payload += ",\"bb_middle\": " + DoubleToString(middle_band[j], _Digits);
                  json_payload += ",\"bb_lower\": " + DoubleToString(lower_band[j], _Digits);
                  
                  // RSI
                  json_payload += ",\"rsi\": " + DoubleToString(rsi[j], 2);
                  
                  // Stochastic
                  json_payload += ",\"stoch_k\": " + DoubleToString(stoch_k[j], 2);
                  json_payload += ",\"stoch_d\": " + DoubleToString(stoch_d[j], 2);
                  
                  // MACD
                  json_payload += ",\"macd_main\": " + DoubleToString(macd_main[j], _Digits);
                  json_payload += ",\"macd_signal\": " + DoubleToString(macd_signal[j], _Digits);
                  json_payload += ",\"macd_hist\": " + DoubleToString(macd_main[j] - macd_signal[j], _Digits);
                  
                  // ADX
                  json_payload += ",\"adx\": " + DoubleToString(adx[j], 2);
                  json_payload += ",\"plus_di\": " + DoubleToString(plus_di[j], 2);
                  json_payload += ",\"minus_di\": " + DoubleToString(minus_di[j], 2);
                  
                  // Ichimoku
                  json_payload += ",\"ichimoku_tenkan\": " + DoubleToString(ichimoku_tenkan[j], _Digits);
                  json_payload += ",\"ichimoku_kijun\": " + DoubleToString(ichimoku_kijun[j], _Digits);
                  json_payload += ",\"ichimoku_senkou_a\": " + DoubleToString(ichimoku_senkou_a[j], _Digits);
                  json_payload += ",\"ichimoku_senkou_b\": " + DoubleToString(ichimoku_senkou_b[j], _Digits);
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

         Sleep(20); // Reduce sleep slightly, still good practice
      }

      json_payload += "}}"; // Close "data" and main object

      // --- Include last error message if there was one ---
      if(lastError != "")
      {
         // Remove closing braces to add error field
         json_payload = StringSubstr(json_payload, 0, StringLen(json_payload) - 1); 
         json_payload += ", \"error\": \"" + lastError + "\"}";
         Print("Including error in payload: ", lastError);
         lastError = ""; // Clear the error after sending
      }
      
      // Send the data and get response
      Print("Sending data to API...");
      string response = Post(url, json_payload);
      Print("Response received: ", response);
      
      // Parse the response to extract instruction
      if(response != "")
      {
         if(StringFind(response, "INSTRUCTION:") >= 0)
         {
            // Extract the instruction part after "INSTRUCTION: "
            int pos = StringFind(response, "INSTRUCTION:") + 13; // 13 is the length of "INSTRUCTION: "
            instruction = StringSubstr(response, pos, StringLen(response) - pos - 2); // -2 to remove trailing quotes and brace
            Print("Parsed Instruction: '", instruction, "'");
            
            // Return the instruction (BUY, SELL, or HOLD)
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
   
   // Process the instruction based on first character (B=Buy, S=Sell, H=Hold)
   ushort first_char = StringGetCharacter(instruction, 0);
   
   // Check if instruction is one of our new full words
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
   else if(first_char == 'B')
   {
      ExecuteBuyOrder();
   }
   else if(first_char == 'S')
   {
      ExecuteSellOrder();
   }
   else if(first_char == 'H')
   {
      Print("Instruction: Hold - No trade action taken");
   }
   else
   {
      Print("Unknown instruction: ", instruction);
   }
}

//+------------------------------------------------------------------+
//| Function to execute a BUY order                                  |
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
//| Function to execute a SELL order                                 |
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
//+-------------------------------------------------------------------++