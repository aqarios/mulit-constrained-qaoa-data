Minimize
 obj: + 216.7 x_user1_load0_hour0 + 183.5 x_user1_load0_hour1 
 + 152.9 x_user1_load0_hour2 + 141 x_user1_load0_hour3 
 + 127.8 x_user1_load0_hour4 + 119.4 x_user1_load0_hour5 
 + 113.4 x_user1_load0_hour6 + 124.2 x_user1_load0_hour7 
 + 143.5 x_user1_load0_hour8 + 163.7 x_user1_load0_hour9 
 + 219.6 x_user1_load1_hour0 + 177.4 x_user1_load1_hour1 
 + 153.8 x_user1_load1_hour2 + 138.8 x_user1_load1_hour3 
 + 130.8 x_user1_load1_hour4 + 115.2 x_user1_load1_hour5 
 + 110.4 x_user1_load1_hour6 + 126 x_user1_load1_hour7 
 + 140.4 x_user1_load1_hour8 + 163.6 x_user1_load1_hour9 
 + 193.4 x_user1_load1_hour10 + 208.7 x_user1_load2_hour0 
 + 181.6 x_user1_load2_hour1 + 150.1 x_user1_load2_hour2 
 + 140.4 x_user1_load2_hour3 + 124.2 x_user1_load2_hour4 
 + 120 x_user1_load2_hour5 + 117 x_user1_load2_hour6 + 126 x_user1_load2_hour7 
 + 148.4 x_user1_load2_hour8 + 168.7 x_user1_load2_hour9 
 + 141.8 x_user1_load3_hour0 + 121.7 x_user1_load3_hour1 
 + 101 x_user1_load3_hour2 + 93.8 x_user1_load3_hour3 + 84 x_user1_load3_hour4 
 + 79.8 x_user1_load3_hour5 + 76.8 x_user1_load3_hour6 
 + 83.4 x_user1_load3_hour7 + 97.3 x_user1_load3_hour8 
 + 110.8 x_user1_load3_hour9 

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