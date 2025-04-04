Minimize
 obj: + 191.6 x_user1_load0_hour0 + 223.7 x_user1_load0_hour1 
 + 229.2 x_user1_load0_hour2 + 211.9 x_user1_load0_hour3 
 + 156 x_user1_load0_hour4 + 124.2 x_user1_load1_hour0 
 + 167.9 x_user1_load1_hour1 + 187.4 x_user1_load1_hour2 
 + 195.6 x_user1_load1_hour3 + 177.6 x_user1_load1_hour4 
 + 132.3 x_user1_load1_hour5 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5  = 1.0
 MaxLoadConnection_user1_0: + 1 x_user1_load0_hour0 + 3 x_user1_load1_hour0 
  <= 3.0
 MaxLoadConnection_user1_1: + 1 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 3 x_user1_load1_hour1 + 2 x_user1_load1_hour0  <= 3.0
 MaxLoadConnection_user1_2: + 1 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 3 x_user1_load0_hour0 + 3 x_user1_load1_hour2 + 2 x_user1_load1_hour1 
  <= 3.0
 MaxLoadConnection_user1_3: + 1 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 3 x_user1_load0_hour1 + 3 x_user1_load1_hour3 + 2 x_user1_load1_hour2 
  <= 3.0
 MaxLoadConnection_user1_4: + 1 x_user1_load0_hour4 + 2 x_user1_load0_hour3 
 + 3 x_user1_load0_hour2 + 3 x_user1_load1_hour4 + 2 x_user1_load1_hour3 
  <= 3.0
 MaxLoadConnection_user1_5: + 2 x_user1_load0_hour4 + 3 x_user1_load0_hour3 
 + 3 x_user1_load1_hour5 + 2 x_user1_load1_hour4  <= 3.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour4 + 2 x_user1_load1_hour5 
  <= 3.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load1_hour0
  x_user1_load1_hour1 x_user1_load1_hour2 x_user1_load1_hour3
  x_user1_load1_hour4 x_user1_load1_hour5
General

End