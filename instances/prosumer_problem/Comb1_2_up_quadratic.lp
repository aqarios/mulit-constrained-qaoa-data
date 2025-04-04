Minimize
 obj: + 190.3 x_user1_load0_hour0 + 149.9 x_user1_load0_hour1 
 + 125 x_user1_load0_hour2 + 142 x_user1_load0_hour3 
 + 172.5 x_user1_load0_hour4 + 130.1 x_user1_load1_hour0 
 + 100 x_user1_load1_hour1 + 82.6 x_user1_load1_hour2 
 + 92.1 x_user1_load1_hour3 + 112.3 x_user1_load1_hour4 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
  <= 3.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 1 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 1 x_user1_load1_hour0  <= 3.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 2 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour0  <= 3.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 1 x_user1_load0_hour2 
 + 2 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 1 x_user1_load1_hour2 
 + 1 x_user1_load1_hour1  <= 3.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 1 x_user1_load0_hour3 
 + 2 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 1 x_user1_load1_hour3 
 + 1 x_user1_load1_hour2  <= 3.0
 MaxLoadConnection_user1_5: + 1 x_user1_load0_hour4 + 2 x_user1_load0_hour3 
 + 1 x_user1_load1_hour4 + 1 x_user1_load1_hour3  <= 3.0
 MaxLoadConnection_user1_6: + 2 x_user1_load0_hour4 + 1 x_user1_load1_hour4 
  <= 3.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load1_hour0
  x_user1_load1_hour1 x_user1_load1_hour2 x_user1_load1_hour3
  x_user1_load1_hour4
General

End