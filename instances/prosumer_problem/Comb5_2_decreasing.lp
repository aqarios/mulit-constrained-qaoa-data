Minimize
 obj: + 341.5 x_user1_load0_hour0 + 318 x_user1_load0_hour1 
 + 289.5 x_user1_load0_hour2 + 277 x_user1_load0_hour3 
 + 251.5 x_user1_load0_hour4 + 224 x_user1_load0_hour5 
 + 111.5 x_user1_load1_hour0 + 104 x_user1_load1_hour1 
 + 95.5 x_user1_load1_hour2 + 91 x_user1_load1_hour3 
 + 81.5 x_user1_load1_hour4 + 72 x_user1_load1_hour5 + 117 x_user1_load2_hour0 
 + 104.5 x_user1_load2_hour1 + 98 x_user1_load2_hour2 
 + 92.5 x_user1_load2_hour3 + 87 x_user1_load2_hour4 
 + 74.5 x_user1_load2_hour5 + 64 x_user1_load2_hour6 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6  = 1.0
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
 MaxLoadConnection_user1_4: + 4 x_user1_load0_hour4 + 3 x_user1_load0_hour3 
 + 2 x_user1_load0_hour2 + 1 x_user1_load1_hour4 + 1 x_user1_load1_hour3 
 + 1 x_user1_load1_hour2 + 1 x_user1_load2_hour4 + 2 x_user1_load2_hour3 
  <= 4.0
 MaxLoadConnection_user1_5: + 4 x_user1_load0_hour5 + 3 x_user1_load0_hour4 
 + 2 x_user1_load0_hour3 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour3 + 1 x_user1_load2_hour5 + 2 x_user1_load2_hour4 
  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour5 + 2 x_user1_load0_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour4 + 1 x_user1_load2_hour6 
 + 2 x_user1_load2_hour5  <= 4.0
 MaxLoadConnection_user1_7: + 2 x_user1_load0_hour5 + 1 x_user1_load1_hour5 
 + 2 x_user1_load2_hour6  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load1_hour0 x_user1_load1_hour1 x_user1_load1_hour2
  x_user1_load1_hour3 x_user1_load1_hour4 x_user1_load1_hour5
  x_user1_load2_hour0 x_user1_load2_hour1 x_user1_load2_hour2
  x_user1_load2_hour3 x_user1_load2_hour4 x_user1_load2_hour5
  x_user1_load2_hour6
General

End