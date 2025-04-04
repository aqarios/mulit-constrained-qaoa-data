Minimize
 obj: + 172 x_user1_load0_hour0 + 114 x_user1_load0_hour1 
 + 104 x_user1_load0_hour2 + 155 x_user1_load0_hour3 + 176 x_user1_load1_hour0 
 + 131 x_user1_load1_hour1 + 151 x_user1_load1_hour2 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 3 x_user1_load1_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 2 x_user1_load1_hour2 + 3 x_user1_load1_hour1 + 1 x_user1_load1_hour0 
  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 3 x_user1_load1_hour2 + 1 x_user1_load1_hour1  <= 4.0
 MaxLoadConnection_user1_4: + 2 x_user1_load0_hour3 + 1 x_user1_load1_hour2 
  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load1_hour0 x_user1_load1_hour1
  x_user1_load1_hour2
General

End