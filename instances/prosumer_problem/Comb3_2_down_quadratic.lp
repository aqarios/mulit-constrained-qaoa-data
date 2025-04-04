Minimize
 obj: + 60 x_user1_load0_hour0 + 96.3 x_user1_load0_hour1 
 + 107.4 x_user1_load0_hour2 + 120 x_user1_load0_hour3 
 + 113.4 x_user1_load0_hour4 + 96.3 x_user1_load0_hour5 
 + 54 x_user1_load0_hour6 + 72.1 x_user1_load1_hour0 + 100 x_user1_load1_hour1 
 + 111.6 x_user1_load1_hour2 + 117.8 x_user1_load1_hour3 
 + 107.7 x_user1_load1_hour4 + 82.2 x_user1_load1_hour5 
 + 92.1 x_user1_load2_hour0 + 132.1 x_user1_load2_hour1 
 + 147.4 x_user1_load2_hour2 + 157.8 x_user1_load2_hour3 
 + 145.5 x_user1_load2_hour4 + 114.3 x_user1_load2_hour5 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5  = 1.0
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
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 2 x_user1_load1_hour4 
 + 1 x_user1_load1_hour3 + 3 x_user1_load2_hour4 + 1 x_user1_load2_hour3 
  <= 3.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 2 x_user1_load1_hour5 
 + 1 x_user1_load1_hour4 + 3 x_user1_load2_hour5 + 1 x_user1_load2_hour4 
  <= 3.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 1 x_user1_load1_hour5 
 + 1 x_user1_load2_hour5  <= 3.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load1_hour0 x_user1_load1_hour1
  x_user1_load1_hour2 x_user1_load1_hour3 x_user1_load1_hour4
  x_user1_load1_hour5 x_user1_load2_hour0 x_user1_load2_hour1
  x_user1_load2_hour2 x_user1_load2_hour3 x_user1_load2_hour4
  x_user1_load2_hour5
General

End