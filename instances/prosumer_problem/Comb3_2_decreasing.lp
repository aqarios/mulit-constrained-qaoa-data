Minimize
 obj: + 120 x_user1_load0_hour0 + 114.3 x_user1_load0_hour1 
 + 96.9 x_user1_load0_hour2 + 94.2 x_user1_load0_hour3 
 + 85.8 x_user1_load0_hour4 + 80.1 x_user1_load0_hour5 
 + 62.7 x_user1_load0_hour6 + 118.1 x_user1_load1_hour0 
 + 108.5 x_user1_load1_hour1 + 96 x_user1_load1_hour2 
 + 91.4 x_user1_load1_hour3 + 83.9 x_user1_load1_hour4 
 + 74.3 x_user1_load1_hour5 + 158.1 x_user1_load2_hour0 
 + 146.6 x_user1_load2_hour1 + 128.3 x_user1_load2_hour2 
 + 122.8 x_user1_load2_hour3 + 112.5 x_user1_load2_hour4 
 + 101 x_user1_load2_hour5 

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