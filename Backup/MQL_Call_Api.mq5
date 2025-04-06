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
          PrintFormat("New bar detected on %s. Previous: %s, Current: %s. Processing...",
                      TimeframeToString(Period()), TimeToString(lastBarTime), TimeToString(currentBarTime));

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

   // --- Get Account Information ---
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double open_trade_pnl = 0.0;
   int total_positions = PositionsTotal();
   for(int i = total_positions - 1; i >= 0; i--) // Loop backwards is safer if closing positions
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)) // Select position to get details
      {
         if(PositionGetString(POSITION_SYMBOL) == symbol_name && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         {
            open_trade_pnl = PositionGetDouble(POSITION_PROFIT);
            // PrintFormat("Found open position #%d for %s, PnL: %.2f", ticket, symbol_name, open_trade_pnl); // Debugging
            break; // Found the relevant position for this symbol/magic
         }
      }
   }
   // PrintFormat("Account Balance: %.2f, Open Trade PnL for %s: %.2f", account_balance, symbol_name, open_trade_pnl); // Debugging

   // Define the timeframes to fetch using the ENUM values
   ENUM_TIMEFRAMES timeframes_to_send[] =
   {
      PERIOD_D1,
      PERIOD_H4,
      PERIOD_H1,
      PERIOD_M30,
      PERIOD_M5,
      PERIOD_M1
   };

   // --- Build the JSON Payload ---
   string json_payload = "{";
   json_payload += "\"symbol\": \"" + symbol_name + "\","; // Symbol first
   json_payload += "\"account_balance\": " + DoubleToString(account_balance, 2) + ","; // Add balance
   json_payload += "\"open_trade_pnl\": " + DoubleToString(open_trade_pnl, 2) + ",";   // Add PnL
   json_payload += "\"data\": {"; // Now the data block

   int tf_count = ArraySize(timeframes_to_send);
   bool first_tf = true; // Flag to handle commas between timeframes

   // Loop through each timeframe defined in timeframes_to_send
   for (int i = 0; i < tf_count; i++)
   {
      ENUM_TIMEFRAMES current_tf = timeframes_to_send[i];
      string tf_string = TimeframeToString(current_tf);
      int bars_to_fetch_for_this_tf = 0;

      // --- Determine how many bars to fetch ---
      switch(current_tf)
      {
         case PERIOD_M1:  bars_to_fetch_for_this_tf = 500;  break;
         case PERIOD_M5:  bars_to_fetch_for_this_tf = 300;  break;
         case PERIOD_M30: bars_to_fetch_for_this_tf = 200; break;
         case PERIOD_H1:  bars_to_fetch_for_this_tf = 100;  break;
         case PERIOD_H4:  bars_to_fetch_for_this_tf = 50;  break;
         case PERIOD_D1:  bars_to_fetch_for_this_tf = 20;  break;
         default: PrintFormat("Warning: No input bar count defined for timeframe %s. Skipping.", tf_string); continue;
      }

      if (bars_to_fetch_for_this_tf <= 0) { PrintFormat("Skipping timeframe %s (bar count <= 0).", tf_string); continue; }

      MqlRates rates_array[];
      ArraySetAsSeries(rates_array, true);
      int copied_count = CopyRates(symbol_name, current_tf, 0, bars_to_fetch_for_this_tf, rates_array);

      if (copied_count > 0)
      {
         PrintFormat("Fetched %d bars for %s (requested %d)", copied_count, tf_string, bars_to_fetch_for_this_tf);
         if (!first_tf) { json_payload += ","; }
         first_tf = false;
         json_payload += "\"" + tf_string + "\": [";
         for (int j = 0; j < copied_count; j++)
         {
            json_payload += "{";
            json_payload += "\"time\": "   + (string)rates_array[j].time + ",";
            json_payload += "\"open\": "   + DoubleToString(rates_array[j].open, _Digits) + ",";
            json_payload += "\"high\": "   + DoubleToString(rates_array[j].high, _Digits) + ",";
            json_payload += "\"low\": "    + DoubleToString(rates_array[j].low, _Digits) + ",";
            json_payload += "\"close\": "  + DoubleToString(rates_array[j].close, _Digits) + ",";
            json_payload += "\"volume\": " + (string)rates_array[j].tick_volume + ",";
            json_payload += "\"spread\": " + (string)rates_array[j].spread;
            json_payload += "}";
            if (j < copied_count - 1) { json_payload += ","; }
         }
         json_payload += "]";
      }
      else { PrintFormat("Warning: Could not fetch bars for %s. Copied: %d (req: %d), Err: %d. Skipping.", tf_string, copied_count, bars_to_fetch_for_this_tf, GetLastError()); }

      Sleep(20); // Reduce sleep slightly, still good practice
   }

   json_payload += "}}"; // Close "data" and main object

   // --- Send Payload and Parse Response ---
   if (!first_tf) // Only send if data was added
   {
        Print("Sending API request to: ", url);
        // Print("Payload snippet: ", StringSubstr(json_payload, 0, 200)); // Optional: uncomment for debugging
        Print("Payload length: ", StringLen(json_payload));

        string response = Post(url, json_payload);
        Print("API Response Raw: ", response); // Log the raw response

        // --- Parse the instruction ---
        string search_key = "\"INSTRUCTION: "; // Includes the space after the colon
        int instruction_pos = StringFind(response, search_key);

        if(instruction_pos >= 0)
        {
           // Find the position after "INSTRUCTION: "
           int start_pos = instruction_pos + StringLen(search_key);
           // Find the closing quote " after the instruction character
           int end_pos = StringFind(response, "\"", start_pos);

           if(end_pos > start_pos)
           {
               // Extract the single character instruction
               instruction = StringSubstr(response, start_pos, end_pos - start_pos);
               // Trim potential whitespace just in case (though unlikely with this format)
               StringTrimLeft(instruction);
               StringTrimRight(instruction);
               Print("Parsed Instruction: '", instruction, "'"); // Log parsed instruction clearly
           }
           else
           {
               Print("Error: Could not find closing quote for instruction in response: ", response);
               instruction = "Error";
           }
        }
        else
        {
            Print("Error: 'INSTRUCTION: ' key not found in response: ", response);
            instruction = "Error";
        }
   }
   else
   {
       Print("No data fetched for any timeframe. API request skipped.");
       instruction = ""; // No instruction if nothing was sent
   }

   return instruction;
}

//+------------------------------------------------------------------+
//| Function to process the trading instruction from the API         |
//+------------------------------------------------------------------+
void ProcessInstruction(string instruction)
{
   string symbol = Symbol();
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);

   if(ask == 0 || bid == 0 || point == 0) // Cannot trade if prices or point size are invalid
   {
       Print("Error: Invalid market info (Ask/Bid/Point). Cannot process instruction.");
       return;
   }

   // --- Check for existing position for THIS symbol and THIS magic number ---
   bool position_exists = false;
   if(PositionSelect(symbol)) // Selects position based on symbol
   {
      // Check if the selected position's magic number matches our EA's magic number
      if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
      {
         position_exists = true;
      }
   }

   // --- Act on Instruction ---
   if(instruction == "B") // Buy Instruction
   {
      if(position_exists)
      {
         Print("Instruction 'B' received, but a position (Magic: ", InpMagicNumber, ") already exists for ", symbol, ". Holding.");
      }
      else
      {
         Print("Instruction 'B' received. Opening BUY order for ", symbol);
         double sl = 0.0;
         double tp = 0.0;

         // Use trade object to open position without SL/TP
         if(!trade.Buy(InpLots, symbol, ask, 0.0, 0.0, "Buy signal from API")) // Pass 0.0 for sl and tp
         {
            Print("Error Opening Buy Order: ", trade.ResultRetcode(), " - ", trade.ResultComment());
         }
         else
         {
            Print("Buy Order Opened Successfully. Ticket: ", trade.ResultOrder());
         }
      }
   }
   else if(instruction == "S") // Sell Instruction
   {
      if(position_exists)
      {
         Print("Instruction 'S' received, but a position (Magic: ", InpMagicNumber, ") already exists for ", symbol, ". Holding.");
      }
      else
      {
         Print("Instruction 'S' received. Opening SELL order for ", symbol);
         double sl = 0.0;
         double tp = 0.0;

         // Use trade object to open position without SL/TP
         if(!trade.Sell(InpLots, symbol, bid, 0.0, 0.0, "Sell signal from API")) // Pass 0.0 for sl and tp
         {
            Print("Error Opening Sell Order: ", trade.ResultRetcode(), " - ", trade.ResultComment());
         }
         else
         {
            Print("Sell Order Opened Successfully. Ticket: ", trade.ResultOrder());
         }
      }
   }
   else if(instruction == "C") // Close Instruction
   {
      if(position_exists)
      {
         Print("Instruction 'C' received. Closing position for ", symbol, " (Magic: ", InpMagicNumber, ")");
         // CTrade::PositionClose will automatically close the position selected by PositionSelect if magic matches
         if(!trade.PositionClose(symbol)) // Closes the position for the specified symbol (uses magic number set in OnInit)
         {
            Print("Error Closing Position: ", trade.ResultRetcode(), " - ", trade.ResultComment());
         }
         else
         {
            Print("Position Closed Successfully. Result: ", trade.ResultComment()); // ResultComment often has details on close
         }
      }
      else
      {
         Print("Instruction 'C' received, but no open position found for ", symbol, " with Magic ", InpMagicNumber, ".");
      }
   }
   else if(instruction == "H") // Hold Instruction
   {
      Print("Instruction 'H' received. No action taken.");
   }
   else // Invalid Instruction
   {
      Print("Warning: Received unknown or invalid instruction '", instruction, "'. No action taken.");
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