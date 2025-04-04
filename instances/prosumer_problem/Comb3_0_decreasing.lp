Minimize
 obj: + 120 x_user1_load0_hour0 + 111 x_user1_load0_hour1 
 + 90 x_user1_load0_hour2 + 84 x_user1_load0_hour3 + 72 x_user1_load0_hour4 
 + 117 x_user1_load1_hour0 + 104 x_user1_load1_hour1 + 88 x_user1_load1_hour2 
 + 80 x_user1_load1_hour3 + 157 x_user1_load2_hour0 + 141 x_user1_load2_hour1 
 + 118 x_user1_load2_hour2 + 108 x_user1_load2_hour3 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 3 x_user1_load2_hour0  <= 3.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 2 x_user1_load1_hour1 
 + 1 x_user1_load1_hour0 + 3 x_user1_load2_hour1 + 1 x_user1_load2_hour0 
  <= 3.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 2 x_user1_load1_hour2 
 + 1 x_user1_load1_hour1 + 3 x_user1_load2_hour2 + 1 x_user1_load2_hour1 
  <= 3.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 2 x_user1_load1_hour3 
 + 1 x_user1_load1_hour2 + 3 x_user1_load2_hour3 + 1 x_user1_load2_hour2 
  <= 3.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 1 x_user1_load1_hour3 
 + 1 x_user1_load2_hour3  <= 3.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load1_hour0
  x_user1_load1_hour1 x_user1_load1_hour2 x_user1_load1_hour3
  x_user1_load2_hour0 x_user1_load2_hour1 x_user1_load2_hour2
  x_user1_load2_hour3
General

End