Minimize
 obj: + 145.3 x_user1_load0_hour0 + 180.4 x_user1_load0_hour1 
 + 197.9 x_user1_load0_hour2 + 223.6 x_user1_load0_hour3 
 + 229.8 x_user1_load0_hour4 + 234 x_user1_load0_hour5 
 + 227 x_user1_load0_hour6 + 218 x_user1_load0_hour7 
 + 209.6 x_user1_load0_hour8 + 185.3 x_user1_load0_hour9 
 + 181.5 x_user1_load1_hour0 + 217.4 x_user1_load1_hour1 
 + 241.6 x_user1_load1_hour2 + 266.8 x_user1_load1_hour3 
 + 270.4 x_user1_load1_hour4 + 269 x_user1_load1_hour5 
 + 264.2 x_user1_load1_hour6 + 252.4 x_user1_load1_hour7 
 + 236.9 x_user1_load1_hour8 + 208.5 x_user1_load1_hour9 
 + 174.4 x_user1_load2_hour0 + 215.4 x_user1_load2_hour1 
 + 235.7 x_user1_load2_hour2 + 264 x_user1_load2_hour3 
 + 267.8 x_user1_load2_hour4 + 271.4 x_user1_load2_hour5 
 + 264.8 x_user1_load2_hour6 + 252 x_user1_load2_hour7 
 + 240.7 x_user1_load2_hour8 + 211.4 x_user1_load2_hour9 
 + 116.2 x_user1_load3_hour0 + 145.4 x_user1_load3_hour1 
 + 160.1 x_user1_load3_hour2 + 183.2 x_user1_load3_hour3 
 + 191.8 x_user1_load3_hour4 + 196.6 x_user1_load3_hour5 
 + 189.2 x_user1_load3_hour6 + 184 x_user1_load3_hour7 
 + 178.5 x_user1_load3_hour8 + 159.2 x_user1_load3_hour9 
 + 103.3 x_user1_load4_hour0 + 120.3 x_user1_load4_hour1 
 + 136.9 x_user1_load4_hour2 + 151 x_user1_load4_hour3 
 + 156.6 x_user1_load4_hour4 + 153.8 x_user1_load4_hour5 
 + 150.6 x_user1_load4_hour6 + 147 x_user1_load4_hour7 
 + 136.9 x_user1_load4_hour8 + 122.3 x_user1_load4_hour9 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7 
 + 1 x_user1_load0_hour8 + 1 x_user1_load0_hour9  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7 
 + 1 x_user1_load1_hour8 + 1 x_user1_load1_hour9  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7 
 + 1 x_user1_load2_hour8 + 1 x_user1_load2_hour9  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour8 + 1 x_user1_load3_hour9  = 1.0
 load_load4_scheduled: + 1 x_user1_load4_hour0 + 1 x_user1_load4_hour1 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour4 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour7 
 + 1 x_user1_load4_hour8 + 1 x_user1_load4_hour9  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 3 x_user1_load2_hour0 + 3 x_user1_load3_hour0 + 1 x_user1_load4_hour0 
  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 1 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 2 x_user1_load1_hour0 + 3 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 3 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 1 x_user1_load4_hour1 + 2 x_user1_load4_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 2 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 2 x_user1_load1_hour1 
 + 3 x_user1_load1_hour0 + 3 x_user1_load2_hour2 + 1 x_user1_load2_hour1 
 + 3 x_user1_load2_hour0 + 3 x_user1_load3_hour2 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour0 + 1 x_user1_load4_hour2 + 2 x_user1_load4_hour1 
 + 1 x_user1_load4_hour0  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 1 x_user1_load0_hour2 
 + 2 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 2 x_user1_load1_hour2 
 + 3 x_user1_load1_hour1 + 3 x_user1_load2_hour3 + 1 x_user1_load2_hour2 
 + 3 x_user1_load2_hour1 + 3 x_user1_load3_hour3 + 1 x_user1_load3_hour2 
 + 1 x_user1_load3_hour1 + 1 x_user1_load4_hour3 + 2 x_user1_load4_hour2 
 + 1 x_user1_load4_hour1  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 1 x_user1_load0_hour3 
 + 2 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 2 x_user1_load1_hour3 
 + 3 x_user1_load1_hour2 + 3 x_user1_load2_hour4 + 1 x_user1_load2_hour3 
 + 3 x_user1_load2_hour2 + 3 x_user1_load3_hour4 + 1 x_user1_load3_hour3 
 + 1 x_user1_load3_hour2 + 1 x_user1_load4_hour4 + 2 x_user1_load4_hour3 
 + 1 x_user1_load4_hour2  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 1 x_user1_load0_hour4 
 + 2 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 2 x_user1_load1_hour4 
 + 3 x_user1_load1_hour3 + 3 x_user1_load2_hour5 + 1 x_user1_load2_hour4 
 + 3 x_user1_load2_hour3 + 3 x_user1_load3_hour5 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour3 + 1 x_user1_load4_hour5 + 2 x_user1_load4_hour4 
 + 1 x_user1_load4_hour3  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 1 x_user1_load0_hour5 
 + 2 x_user1_load0_hour4 + 2 x_user1_load1_hour6 + 2 x_user1_load1_hour5 
 + 3 x_user1_load1_hour4 + 3 x_user1_load2_hour6 + 1 x_user1_load2_hour5 
 + 3 x_user1_load2_hour4 + 3 x_user1_load3_hour6 + 1 x_user1_load3_hour5 
 + 1 x_user1_load3_hour4 + 1 x_user1_load4_hour6 + 2 x_user1_load4_hour5 
 + 1 x_user1_load4_hour4  <= 4.0
 MaxLoadConnection_user1_7: + 3 x_user1_load0_hour7 + 1 x_user1_load0_hour6 
 + 2 x_user1_load0_hour5 + 2 x_user1_load1_hour7 + 2 x_user1_load1_hour6 
 + 3 x_user1_load1_hour5 + 3 x_user1_load2_hour7 + 1 x_user1_load2_hour6 
 + 3 x_user1_load2_hour5 + 3 x_user1_load3_hour7 + 1 x_user1_load3_hour6 
 + 1 x_user1_load3_hour5 + 1 x_user1_load4_hour7 + 2 x_user1_load4_hour6 
 + 1 x_user1_load4_hour5  <= 4.0
 MaxLoadConnection_user1_8: + 3 x_user1_load0_hour8 + 1 x_user1_load0_hour7 
 + 2 x_user1_load0_hour6 + 2 x_user1_load1_hour8 + 2 x_user1_load1_hour7 
 + 3 x_user1_load1_hour6 + 3 x_user1_load2_hour8 + 1 x_user1_load2_hour7 
 + 3 x_user1_load2_hour6 + 3 x_user1_load3_hour8 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour6 + 1 x_user1_load4_hour8 + 2 x_user1_load4_hour7 
 + 1 x_user1_load4_hour6  <= 4.0
 MaxLoadConnection_user1_9: + 3 x_user1_load0_hour9 + 1 x_user1_load0_hour8 
 + 2 x_user1_load0_hour7 + 2 x_user1_load1_hour9 + 2 x_user1_load1_hour8 
 + 3 x_user1_load1_hour7 + 3 x_user1_load2_hour9 + 1 x_user1_load2_hour8 
 + 3 x_user1_load2_hour7 + 3 x_user1_load3_hour9 + 1 x_user1_load3_hour8 
 + 1 x_user1_load3_hour7 + 1 x_user1_load4_hour9 + 2 x_user1_load4_hour8 
 + 1 x_user1_load4_hour7  <= 4.0
 MaxLoadConnection_user1_10: + 1 x_user1_load0_hour9 + 2 x_user1_load0_hour8 
 + 2 x_user1_load1_hour9 + 3 x_user1_load1_hour8 + 1 x_user1_load2_hour9 
 + 3 x_user1_load2_hour8 + 1 x_user1_load3_hour9 + 1 x_user1_load3_hour8 
 + 2 x_user1_load4_hour9 + 1 x_user1_load4_hour8  <= 4.0
 MaxLoadConnection_user1_11: + 2 x_user1_load0_hour9 + 3 x_user1_load1_hour9 
 + 3 x_user1_load2_hour9 + 1 x_user1_load3_hour9 + 1 x_user1_load4_hour9 
  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load0_hour8
  x_user1_load0_hour9 x_user1_load1_hour0 x_user1_load1_hour1
  x_user1_load1_hour2 x_user1_load1_hour3 x_user1_load1_hour4
  x_user1_load1_hour5 x_user1_load1_hour6 x_user1_load1_hour7
  x_user1_load1_hour8 x_user1_load1_hour9 x_user1_load2_hour0
  x_user1_load2_hour1 x_user1_load2_hour2 x_user1_load2_hour3
  x_user1_load2_hour4 x_user1_load2_hour5 x_user1_load2_hour6
  x_user1_load2_hour7 x_user1_load2_hour8 x_user1_load2_hour9
  x_user1_load3_hour0 x_user1_load3_hour1 x_user1_load3_hour2
  x_user1_load3_hour3 x_user1_load3_hour4 x_user1_load3_hour5
  x_user1_load3_hour6 x_user1_load3_hour7 x_user1_load3_hour8
  x_user1_load3_hour9 x_user1_load4_hour0 x_user1_load4_hour1
  x_user1_load4_hour2 x_user1_load4_hour3 x_user1_load4_hour4
  x_user1_load4_hour5 x_user1_load4_hour6 x_user1_load4_hour7
  x_user1_load4_hour8 x_user1_load4_hour9
General

End