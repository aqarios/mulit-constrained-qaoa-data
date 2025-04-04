Minimize
 obj: + 146 x_user1_load0_hour0 + 137 x_user1_load0_hour1 
 + 188 x_user1_load0_hour2 + 172 x_user1_load1_hour0 + 114 x_user1_load1_hour1 
 + 104 x_user1_load1_hour2 + 155 x_user1_load1_hour3 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3  = 1.0
 MaxLoadConnection_user1_0: + 1 x_user1_load0_hour0 + 3 x_user1_load1_hour0 
  <= 3.0
 MaxLoadConnection_user1_1: + 1 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 3 x_user1_load1_hour1 + 2 x_user1_load1_hour0  <= 3.0
 MaxLoadConnection_user1_2: + 1 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 3 x_user1_load0_hour0 + 3 x_user1_load1_hour2 + 2 x_user1_load1_hour1 
  <= 3.0
 MaxLoadConnection_user1_3: + 2 x_user1_load0_hour2 + 3 x_user1_load0_hour1 
 + 3 x_user1_load1_hour3 + 2 x_user1_load1_hour2  <= 3.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour2 + 2 x_user1_load1_hour3 
  <= 3.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load1_hour0 x_user1_load1_hour1 x_user1_load1_hour2
  x_user1_load1_hour3
General

End