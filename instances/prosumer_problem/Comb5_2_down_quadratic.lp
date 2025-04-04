Minimize
 obj: + 235.4 x_user1_load0_hour0 + 295.8 x_user1_load0_hour1 
 + 328.4 x_user1_load0_hour2 + 354.8 x_user1_load0_hour3 
 + 345.4 x_user1_load0_hour4 + 311.8 x_user1_load0_hour5 
 + 82.8 x_user1_load1_hour0 + 101.6 x_user1_load1_hour1 
 + 111.8 x_user1_load1_hour2 + 118.6 x_user1_load1_hour3 
 + 112.8 x_user1_load1_hour4 + 99.6 x_user1_load1_hour5 
 + 79.6 x_user1_load2_hour0 + 95.8 x_user1_load2_hour1 
 + 110.6 x_user1_load2_hour2 + 118.8 x_user1_load2_hour3 
 + 119.6 x_user1_load2_hour4 + 105.8 x_user1_load2_hour5 
 + 86.6 x_user1_load2_hour6 

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