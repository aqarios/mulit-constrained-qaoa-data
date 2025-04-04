Minimize
 obj: + 247.9 x_user1_load0_hour0 + 315.8 x_user1_load0_hour1 
 + 338.8 x_user1_load0_hour2 + 337.6 x_user1_load0_hour3 
 + 87.9 x_user1_load1_hour0 + 107.9 x_user1_load1_hour1 
 + 113.6 x_user1_load1_hour2 + 109.9 x_user1_load1_hour3 
 + 84.2 x_user1_load2_hour0 + 103.7 x_user1_load2_hour1 
 + 115.8 x_user1_load2_hour2 + 115.6 x_user1_load2_hour3 
 + 102 x_user1_load2_hour4 

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