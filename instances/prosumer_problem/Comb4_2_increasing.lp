Minimize
 obj: + 154 x_user1_load0_hour0 + 174.5 x_user1_load0_hour1 
 + 183 x_user1_load0_hour2 + 206.5 x_user1_load0_hour3 
 + 222 x_user1_load0_hour4 + 60 x_user1_load1_hour0 + 70.5 x_user1_load1_hour1 
 + 69 x_user1_load1_hour2 + 82.5 x_user1_load1_hour3 + 90 x_user1_load1_hour4 
 + 100.5 x_user1_load1_hour5 + 99 x_user1_load1_hour6 
 + 106.5 x_user1_load1_hour7 + 110.5 x_user1_load2_hour0 
 + 116 x_user1_load2_hour1 + 128.5 x_user1_load2_hour2 
 + 145 x_user1_load2_hour3 + 160.5 x_user1_load2_hour4 
 + 166 x_user1_load2_hour5 + 172.5 x_user1_load2_hour6 

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