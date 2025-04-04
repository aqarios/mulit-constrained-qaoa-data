Minimize
 obj: + 196.5 x_user1_load0_hour0 + 218 x_user1_load0_hour1 
 + 234.5 x_user1_load0_hour2 + 267 x_user1_load0_hour3 
 + 286.5 x_user1_load0_hour4 + 304 x_user1_load0_hour5 
 + 66.5 x_user1_load1_hour0 + 74 x_user1_load1_hour1 
 + 80.5 x_user1_load1_hour2 + 91 x_user1_load1_hour3 
 + 96.5 x_user1_load1_hour4 + 102 x_user1_load1_hour5 + 67 x_user1_load2_hour0 
 + 69.5 x_user1_load2_hour1 + 78 x_user1_load2_hour2 
 + 87.5 x_user1_load2_hour3 + 97 x_user1_load2_hour4 
 + 99.5 x_user1_load2_hour5 + 104 x_user1_load2_hour6 

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