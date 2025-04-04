Minimize
 obj: + 259 x_user1_load0_hour0 + 238.8 x_user1_load0_hour1 
 + 206.2 x_user1_load0_hour2 + 120 x_user1_load1_hour0 
 + 113.1 x_user1_load1_hour1 + 93.9 x_user1_load1_hour2 
 + 90 x_user1_load1_hour3 + 80.1 x_user1_load1_hour4 
 + 72.9 x_user1_load1_hour5 + 193.1 x_user1_load2_hour0 
 + 169.3 x_user1_load2_hour1 + 152.6 x_user1_load2_hour2 
 + 140.1 x_user1_load2_hour3 + 126.3 x_user1_load2_hour4 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4  = 1.0
 MaxLoadConnection_user1_0: + 4 x_user1_load0_hour0 + 3 x_user1_load1_hour0 
 + 2 x_user1_load2_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 4 x_user1_load0_hour1 + 1 x_user1_load0_hour0 
 + 3 x_user1_load1_hour1 + 2 x_user1_load2_hour1 + 3 x_user1_load2_hour0 
  <= 4.0
 MaxLoadConnection_user1_2: + 4 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 3 x_user1_load1_hour2 + 2 x_user1_load2_hour2 
 + 3 x_user1_load2_hour1  <= 4.0
 MaxLoadConnection_user1_3: + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 3 x_user1_load1_hour3 + 2 x_user1_load2_hour3 
 + 3 x_user1_load2_hour2  <= 4.0
 MaxLoadConnection_user1_4: + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 3 x_user1_load1_hour4 + 2 x_user1_load2_hour4 + 3 x_user1_load2_hour3 
  <= 4.0
 MaxLoadConnection_user1_5: + 1 x_user1_load0_hour2 + 3 x_user1_load1_hour5 
 + 3 x_user1_load2_hour4  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load1_hour0 x_user1_load1_hour1 x_user1_load1_hour2
  x_user1_load1_hour3 x_user1_load1_hour4 x_user1_load1_hour5
  x_user1_load2_hour0 x_user1_load2_hour1 x_user1_load2_hour2
  x_user1_load2_hour3 x_user1_load2_hour4
General

End