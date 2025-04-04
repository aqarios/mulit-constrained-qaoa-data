Minimize
 obj: + 290.1 x_user1_load0_hour0 + 220.2 x_user1_load0_hour1 
 + 185.2 x_user1_load0_hour2 + 206.4 x_user1_load0_hour3 
 + 90.1 x_user1_load1_hour0 + 70.1 x_user1_load1_hour1 
 + 62.4 x_user1_load1_hour2 + 72.1 x_user1_load1_hour3 
 + 99.8 x_user1_load2_hour0 + 70.3 x_user1_load2_hour1 
 + 60.2 x_user1_load2_hour2 + 64.4 x_user1_load2_hour3 
 + 82 x_user1_load2_hour4 

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