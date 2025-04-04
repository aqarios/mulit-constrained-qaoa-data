Minimize
 obj: + 335.7 x_user1_load0_hour0 + 304.7 x_user1_load0_hour1 
 + 268.6 x_user1_load0_hour2 + 248.7 x_user1_load0_hour3 
 + 109 x_user1_load1_hour0 + 99 x_user1_load1_hour1 + 88 x_user1_load1_hour2 
 + 81 x_user1_load1_hour3 + 115.4 x_user1_load2_hour0 
 + 100.3 x_user1_load2_hour1 + 91.3 x_user1_load2_hour2 
 + 83.4 x_user1_load2_hour3 + 75.3 x_user1_load2_hour4 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4  = 1.0
 MaxLoadConnection_user1_0: + 4 x_user1_load0_hour0 + 1 x_user1_load1_hour0 
 + 1 x_user1_load2_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 4 x_user1_load0_hour1 + 3 x_user1_load0_hour0 
 + 1 x_user1_load1_hour1 + 1 x_user1_load1_hour0 + 1 x_user1_load2_hour1 
 + 2 x_user1_load2_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 4 x_user1_load0_hour2 + 3 x_user1_load0_hour1 
 + 2 x_user1_load0_hour0 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour0 + 1 x_user1_load2_hour2 + 2 x_user1_load2_hour1 
  <= 4.0
 MaxLoadConnection_user1_3: + 4 x_user1_load0_hour3 + 3 x_user1_load0_hour2 
 + 2 x_user1_load0_hour1 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour2 
 + 1 x_user1_load1_hour1 + 1 x_user1_load2_hour3 + 2 x_user1_load2_hour2 
  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour2 + 1 x_user1_load2_hour4 
 + 2 x_user1_load2_hour3  <= 4.0
 MaxLoadConnection_user1_5: + 2 x_user1_load0_hour3 + 1 x_user1_load1_hour3 
 + 2 x_user1_load2_hour4  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load1_hour0 x_user1_load1_hour1
  x_user1_load1_hour2 x_user1_load1_hour3 x_user1_load2_hour0
  x_user1_load2_hour1 x_user1_load2_hour2 x_user1_load2_hour3
  x_user1_load2_hour4
General

End