Minimize
 obj: + 233.3 x_user1_load0_hour0 + 222.3 x_user1_load0_hour1 
 + 207.4 x_user1_load0_hour2 + 204.3 x_user1_load0_hour3 
 + 193.3 x_user1_load0_hour4 + 180.4 x_user1_load0_hour5 
 + 163.3 x_user1_load0_hour6 + 156.3 x_user1_load0_hour7 
 + 151.4 x_user1_load0_hour8 + 140.3 x_user1_load0_hour9 
 + 237.2 x_user1_load1_hour0 + 217.4 x_user1_load1_hour1 
 + 209.4 x_user1_load1_hour2 + 203.2 x_user1_load1_hour3 
 + 197.4 x_user1_load1_hour4 + 177.4 x_user1_load1_hour5 
 + 161.2 x_user1_load1_hour6 + 159.4 x_user1_load1_hour7 
 + 149.4 x_user1_load1_hour8 + 141.2 x_user1_load1_hour9 
 + 133.4 x_user1_load1_hour10 + 228.7 x_user1_load2_hour0 
 + 222.6 x_user1_load2_hour1 + 205.7 x_user1_load2_hour2 
 + 203.7 x_user1_load2_hour3 + 188.6 x_user1_load2_hour4 
 + 178.7 x_user1_load2_hour5 + 163.7 x_user1_load2_hour6 
 + 153.6 x_user1_load2_hour7 + 150.7 x_user1_load2_hour8 
 + 138.7 x_user1_load2_hour9 + 154 x_user1_load3_hour0 
 + 148.3 x_user1_load3_hour1 + 137.7 x_user1_load3_hour2 
 + 136 x_user1_load3_hour3 + 127.3 x_user1_load3_hour4 
 + 119.7 x_user1_load3_hour5 + 109 x_user1_load3_hour6 
 + 103.3 x_user1_load3_hour7 + 100.7 x_user1_load3_hour8 
 + 93 x_user1_load3_hour9 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7 
 + 1 x_user1_load0_hour8 + 1 x_user1_load0_hour9  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7 
 + 1 x_user1_load1_hour8 + 1 x_user1_load1_hour9 + 1 x_user1_load1_hour10 
  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7 
 + 1 x_user1_load2_hour8 + 1 x_user1_load2_hour9  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour8 + 1 x_user1_load3_hour9  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 3 x_user1_load2_hour0 + 2 x_user1_load3_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 4 x_user1_load1_hour0 + 3 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 2 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 4 x_user1_load1_hour1 
 + 3 x_user1_load2_hour2 + 1 x_user1_load2_hour1 + 2 x_user1_load2_hour0 
 + 2 x_user1_load3_hour2 + 1 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 1 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 4 x_user1_load1_hour2 
 + 3 x_user1_load2_hour3 + 1 x_user1_load2_hour2 + 2 x_user1_load2_hour1 
 + 2 x_user1_load3_hour3 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour1 
  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 2 x_user1_load0_hour3 
 + 1 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 4 x_user1_load1_hour3 
 + 3 x_user1_load2_hour4 + 1 x_user1_load2_hour3 + 2 x_user1_load2_hour2 
 + 2 x_user1_load3_hour4 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour2 
  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 2 x_user1_load0_hour4 
 + 1 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 4 x_user1_load1_hour4 
 + 3 x_user1_load2_hour5 + 1 x_user1_load2_hour4 + 2 x_user1_load2_hour3 
 + 2 x_user1_load3_hour5 + 1 x_user1_load3_hour4 + 1 x_user1_load3_hour3 
  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 2 x_user1_load0_hour5 
 + 1 x_user1_load0_hour4 + 2 x_user1_load1_hour6 + 4 x_user1_load1_hour5 
 + 3 x_user1_load2_hour6 + 1 x_user1_load2_hour5 + 2 x_user1_load2_hour4 
 + 2 x_user1_load3_hour6 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour4 
  <= 4.0
 MaxLoadConnection_user1_7: + 3 x_user1_load0_hour7 + 2 x_user1_load0_hour6 
 + 1 x_user1_load0_hour5 + 2 x_user1_load1_hour7 + 4 x_user1_load1_hour6 
 + 3 x_user1_load2_hour7 + 1 x_user1_load2_hour6 + 2 x_user1_load2_hour5 
 + 2 x_user1_load3_hour7 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour5 
  <= 4.0
 MaxLoadConnection_user1_8: + 3 x_user1_load0_hour8 + 2 x_user1_load0_hour7 
 + 1 x_user1_load0_hour6 + 2 x_user1_load1_hour8 + 4 x_user1_load1_hour7 
 + 3 x_user1_load2_hour8 + 1 x_user1_load2_hour7 + 2 x_user1_load2_hour6 
 + 2 x_user1_load3_hour8 + 1 x_user1_load3_hour7 + 1 x_user1_load3_hour6 
  <= 4.0
 MaxLoadConnection_user1_9: + 3 x_user1_load0_hour9 + 2 x_user1_load0_hour8 
 + 1 x_user1_load0_hour7 + 2 x_user1_load1_hour9 + 4 x_user1_load1_hour8 
 + 3 x_user1_load2_hour9 + 1 x_user1_load2_hour8 + 2 x_user1_load2_hour7 
 + 2 x_user1_load3_hour9 + 1 x_user1_load3_hour8 + 1 x_user1_load3_hour7 
  <= 4.0
 MaxLoadConnection_user1_10: + 2 x_user1_load0_hour9 + 1 x_user1_load0_hour8 
 + 2 x_user1_load1_hour10 + 4 x_user1_load1_hour9 + 1 x_user1_load2_hour9 
 + 2 x_user1_load2_hour8 + 1 x_user1_load3_hour9 + 1 x_user1_load3_hour8 
  <= 4.0
 MaxLoadConnection_user1_11: + 1 x_user1_load0_hour9 + 4 x_user1_load1_hour10 
 + 2 x_user1_load2_hour9 + 1 x_user1_load3_hour9  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load0_hour8
  x_user1_load0_hour9 x_user1_load1_hour0 x_user1_load1_hour1
  x_user1_load1_hour2 x_user1_load1_hour3 x_user1_load1_hour4
  x_user1_load1_hour5 x_user1_load1_hour6 x_user1_load1_hour7
  x_user1_load1_hour8 x_user1_load1_hour9 x_user1_load1_hour10
  x_user1_load2_hour0 x_user1_load2_hour1 x_user1_load2_hour2
  x_user1_load2_hour3 x_user1_load2_hour4 x_user1_load2_hour5
  x_user1_load2_hour6 x_user1_load2_hour7 x_user1_load2_hour8
  x_user1_load2_hour9 x_user1_load3_hour0 x_user1_load3_hour1
  x_user1_load3_hour2 x_user1_load3_hour3 x_user1_load3_hour4
  x_user1_load3_hour5 x_user1_load3_hour6 x_user1_load3_hour7
  x_user1_load3_hour8 x_user1_load3_hour9
General

End