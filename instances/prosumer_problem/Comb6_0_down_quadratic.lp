Minimize
 obj: + 155.8 x_user1_load0_hour0 + 200 x_user1_load0_hour1 
 + 217.8 x_user1_load0_hour2 + 236 x_user1_load0_hour3 
 + 225.8 x_user1_load0_hour4 + 206 x_user1_load0_hour5 
 + 198.6 x_user1_load1_hour0 + 242 x_user1_load1_hour1 
 + 263.6 x_user1_load1_hour2 + 277 x_user1_load1_hour3 
 + 258.6 x_user1_load1_hour4 + 226 x_user1_load1_hour5 
 + 188.8 x_user1_load2_hour0 + 238.8 x_user1_load2_hour1 
 + 257.8 x_user1_load2_hour2 + 275.8 x_user1_load2_hour3 
 + 258.8 x_user1_load2_hour4 + 232.8 x_user1_load2_hour5 
 + 60 x_user1_load3_hour0 + 89.4 x_user1_load3_hour1 + 99 x_user1_load3_hour2 
 + 116.4 x_user1_load3_hour3 + 120 x_user1_load3_hour4 
 + 119.4 x_user1_load3_hour5 + 99 x_user1_load3_hour6 
 + 80.4 x_user1_load3_hour7 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 3 x_user1_load2_hour0 + 3 x_user1_load3_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 1 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 2 x_user1_load1_hour0 + 3 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 3 x_user1_load3_hour1  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 2 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 2 x_user1_load1_hour1 
 + 3 x_user1_load1_hour0 + 3 x_user1_load2_hour2 + 1 x_user1_load2_hour1 
 + 3 x_user1_load2_hour0 + 3 x_user1_load3_hour2  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 1 x_user1_load0_hour2 
 + 2 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 2 x_user1_load1_hour2 
 + 3 x_user1_load1_hour1 + 3 x_user1_load2_hour3 + 1 x_user1_load2_hour2 
 + 3 x_user1_load2_hour1 + 3 x_user1_load3_hour3  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 1 x_user1_load0_hour3 
 + 2 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 2 x_user1_load1_hour3 
 + 3 x_user1_load1_hour2 + 3 x_user1_load2_hour4 + 1 x_user1_load2_hour3 
 + 3 x_user1_load2_hour2 + 3 x_user1_load3_hour4  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 1 x_user1_load0_hour4 
 + 2 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 2 x_user1_load1_hour4 
 + 3 x_user1_load1_hour3 + 3 x_user1_load2_hour5 + 1 x_user1_load2_hour4 
 + 3 x_user1_load2_hour3 + 3 x_user1_load3_hour5  <= 4.0
 MaxLoadConnection_user1_6: + 1 x_user1_load0_hour5 + 2 x_user1_load0_hour4 
 + 2 x_user1_load1_hour5 + 3 x_user1_load1_hour4 + 1 x_user1_load2_hour5 
 + 3 x_user1_load2_hour4 + 3 x_user1_load3_hour6  <= 4.0
 MaxLoadConnection_user1_7: + 2 x_user1_load0_hour5 + 3 x_user1_load1_hour5 
 + 3 x_user1_load2_hour5 + 3 x_user1_load3_hour7  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load1_hour0 x_user1_load1_hour1 x_user1_load1_hour2
  x_user1_load1_hour3 x_user1_load1_hour4 x_user1_load1_hour5
  x_user1_load2_hour0 x_user1_load2_hour1 x_user1_load2_hour2
  x_user1_load2_hour3 x_user1_load2_hour4 x_user1_load2_hour5
  x_user1_load3_hour0 x_user1_load3_hour1 x_user1_load3_hour2
  x_user1_load3_hour3 x_user1_load3_hour4 x_user1_load3_hour5
  x_user1_load3_hour6 x_user1_load3_hour7
General

End