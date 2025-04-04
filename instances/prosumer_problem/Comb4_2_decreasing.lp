Minimize
 obj: + 264 x_user1_load0_hour0 + 249.5 x_user1_load0_hour1 
 + 223 x_user1_load0_hour2 + 211.5 x_user1_load0_hour3 
 + 192 x_user1_load0_hour4 + 120 x_user1_load1_hour0 
 + 115.5 x_user1_load1_hour1 + 99 x_user1_load1_hour2 
 + 97.5 x_user1_load1_hour3 + 90 x_user1_load1_hour4 
 + 85.5 x_user1_load1_hour5 + 69 x_user1_load1_hour6 
 + 61.5 x_user1_load1_hour7 + 195.5 x_user1_load2_hour0 
 + 176 x_user1_load2_hour1 + 163.5 x_user1_load2_hour2 
 + 155 x_user1_load2_hour3 + 145.5 x_user1_load2_hour4 
 + 126 x_user1_load2_hour5 + 107.5 x_user1_load2_hour6 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6  = 1.0
 MaxLoadConnection_user1_0: + 4 x_user1_load0_hour0 + 3 x_user1_load1_hour0 
 + 2 x_user1_load2_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 4 x_user1_load0_hour1 + 1 x_user1_load0_hour0 
 + 3 x_user1_load1_hour1 + 2 x_user1_load2_hour1 + 3 x_user1_load2_hour0 
  <= 4.0
 MaxLoadConnection_user1_2: + 4 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 3 x_user1_load1_hour2 + 2 x_user1_load2_hour2 
 + 3 x_user1_load2_hour1  <= 4.0
 MaxLoadConnection_user1_3: + 4 x_user1_load0_hour3 + 1 x_user1_load0_hour2 
 + 1 x_user1_load0_hour1 + 1 x_user1_load0_hour0 + 3 x_user1_load1_hour3 
 + 2 x_user1_load2_hour3 + 3 x_user1_load2_hour2  <= 4.0
 MaxLoadConnection_user1_4: + 4 x_user1_load0_hour4 + 1 x_user1_load0_hour3 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour1 + 3 x_user1_load1_hour4 
 + 2 x_user1_load2_hour4 + 3 x_user1_load2_hour3  <= 4.0
 MaxLoadConnection_user1_5: + 1 x_user1_load0_hour4 + 1 x_user1_load0_hour3 
 + 1 x_user1_load0_hour2 + 3 x_user1_load1_hour5 + 2 x_user1_load2_hour5 
 + 3 x_user1_load2_hour4  <= 4.0
 MaxLoadConnection_user1_6: + 1 x_user1_load0_hour4 + 1 x_user1_load0_hour3 
 + 3 x_user1_load1_hour6 + 2 x_user1_load2_hour6 + 3 x_user1_load2_hour5 
  <= 4.0
 MaxLoadConnection_user1_7: + 1 x_user1_load0_hour4 + 3 x_user1_load1_hour7 
 + 3 x_user1_load2_hour6  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load1_hour0
  x_user1_load1_hour1 x_user1_load1_hour2 x_user1_load1_hour3
  x_user1_load1_hour4 x_user1_load1_hour5 x_user1_load1_hour6
  x_user1_load1_hour7 x_user1_load2_hour0 x_user1_load2_hour1
  x_user1_load2_hour2 x_user1_load2_hour3 x_user1_load2_hour4
  x_user1_load2_hour5 x_user1_load2_hour6
General

End