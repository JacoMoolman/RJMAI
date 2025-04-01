//+------------------------------------------------------------------+
//|                                        JMAI-AllTimeFrames.mq5 |
//|                                            Copyright 2025, User |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, User"
#property link      "https://www.mql5.com"
#property version   "1.04" // Incremented version

//+------------------------------------------------------------------+
//| DLL Imports                                                      |
//+------------------------------------------------------------------+
#import "restmql_x64.dll"
   int CPing(string &str);
   string CPing2();
   string Get(string url);
   string Post(string url, string data);
#import

// Static variable to track the time of the last bar on the chart's timeframe
static datetime lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("API Test EA initializing...");
   // Initialize lastBarTime to prevent sending data immediately on the very first tick
   // We'll let the first new bar trigger the send in OnTick
   lastBarTime = iTime(Symbol(), Period(), 0);
   Print("Initial bar time set to: ", TimeToString(lastBarTime));
   Print("API Test EA initialized!");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("API Test EA deinitialized! Reason: ", reason);
   lastBarTime = 0; // Reset static variable
}

//+------------------------------------------------------------------+
//| Expert tick function - Now triggers on new bar                   |
//+------------------------------------------------------------------+
void OnTick()
{
   // Get the time of the current bar (index 0) on the chart's timeframe
   datetime currentBarTime = iTime(Symbol(), Period(), 0);

   // Check if a new bar has started
   if (currentBarTime != lastBarTime)
   {
      PrintFormat("New bar detected on %s. Previous: %s, Current: %s. Sending data...",
                  TimeframeToString(Period()), TimeToString(lastBarTime), TimeToString(currentBarTime));

      // --- Send the multi-timeframe historical data ---
      SendHistoricalData();
      // ---------------------------------------------

      // Update the time of the last known bar
      lastBarTime = currentBarTime;
   }
   // --- Optional: Add a heartbeat print to show OnTick is running between bars ---
   /*
   static datetime lastTickPrint = 0;
   if(TimeCurrent() - lastTickPrint >= 60) // Print every 60 seconds
   {
      Print("OnTick running at ", TimeToString(TimeCurrent()), ". Waiting for new bar on ", TimeframeToString(Period()));
      lastTickPrint = TimeCurrent();
   }
   */
   // -----------------------------------------------------------------------------
}

//+------------------------------------------------------------------+
//| Function to fetch and send historical data (unchanged logic)     |
//+------------------------------------------------------------------+
void SendHistoricalData()
{
   string url = "http://localhost:5000/test"; // Your API endpoint
   string symbol_name = Symbol();
   int bars_to_fetch = 5;

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
   json_payload += "\"symbol\": \"" + symbol_name + "\",";
   json_payload += "\"data\": {";

   int tf_count = ArraySize(timeframes_to_send);
   bool first_tf = true; // Flag to handle commas between timeframes

   // Loop through each timeframe defined in timeframes_to_send
   for (int i = 0; i < tf_count; i++)
   {
      ENUM_TIMEFRAMES current_tf = timeframes_to_send[i];
      string tf_string = TimeframeToString(current_tf); // Helper function

      MqlRates rates_array[]; // Array to store bar data
      ArraySetAsSeries(rates_array, true); // Set array as series (index 0 is the current bar)

      // Copy historical rates for the specific timeframe
      int copied_count = CopyRates(symbol_name, current_tf, 0, bars_to_fetch, rates_array);

      // Check if we got any data for this specific timeframe
      if (copied_count > 0)
      {
         PrintFormat("Fetched %d bars for %s (requested %d)", copied_count, tf_string, bars_to_fetch);

         // Add comma before this timeframe's data if it's not the first one
         if (!first_tf)
         {
            json_payload += ",";
         }
         first_tf = false; // Reset flag after the first successful timeframe

         // Start the JSON array for this timeframe's bars
         json_payload += "\"" + tf_string + "\": [";

         // Loop through the copied bars (newest [0] to oldest [copied_count-1])
         for (int j = 0; j < copied_count; j++)
         {
            // Start bar object
            json_payload += "{";
            json_payload += "\"time\": " + (string)rates_array[j].time + ",";
            json_payload += "\"open\": " + DoubleToString(rates_array[j].open, _Digits) + ",";
            json_payload += "\"high\": " + DoubleToString(rates_array[j].high, _Digits) + ",";
            json_payload += "\"low\": " + DoubleToString(rates_array[j].low, _Digits) + ",";
            json_payload += "\"close\": " + DoubleToString(rates_array[j].close, _Digits) + ",";
            json_payload += "\"volume\": " + (string)rates_array[j].tick_volume + ",";
            json_payload += "\"spread\": " + (string)rates_array[j].spread;
            // Close bar object
            json_payload += "}";

            // Add comma between bar objects if it's not the last bar
            if (j < copied_count - 1)
            {
               json_payload += ",";
            }
         }
         // Close the JSON array for this timeframe
         json_payload += "]";
      }
      else // Handle cases where no bars were copied for this timeframe
      {
         int error_code = GetLastError();
         PrintFormat("Warning: Could not fetch bars for %s. Copied: %d, Error code: %d. Skipping timeframe.",
                     tf_string, copied_count, error_code);
      }
      Sleep(50); // Small delay between fetching different timeframes

   } // End loop through timeframes

   // Close the data object and the main JSON object
   json_payload += "}"; // Close "data" object
   json_payload += "}"; // Close main JSON object

   // --- Send the JSON Payload ---
   Print("Sending API request to: ", url);
   Print("Payload length: ", StringLen(json_payload)); // Print length as a check

   string response = Post(url, json_payload);

   // Print the response from the API server
   Print("API Response: ", response);
}
//+------------------------------------------------------------------+
//| Helper function to convert ENUM_TIMEFRAMES to string (unchanged) |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES period)
{
   switch (period)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return EnumToString(period);
   }
}
//+------------------------------------------------------------------+